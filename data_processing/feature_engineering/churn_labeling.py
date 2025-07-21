from pyspark.sql.functions import col
from pyspark.sql import DataFrame
from datetime import datetime, timedelta
from typing import Dict

from config.pipeline_config_loader import load_pipeline_config, load_spark_config, create_spark_session

pipeline_config = load_pipeline_config("config/pipeline_config.dev.yaml")
spark_config = load_spark_config("config/spark_config.yaml")

# Create SparkSession
spark_session = create_spark_session(spark_config)

class ChurnLabelingService:
    """
    Weekly execution: Churn labeling and backfilling
    """
    
    def __init__(self, pipeline_config):
        self.spark = spark_session
        self.config = pipeline_config
    
    def generate_churn_labels(self, raw_data: Dict[str, DataFrame], snapshot_date: str) -> DataFrame:
        """
        Generate churn labels for a specific snapshot date
        
        Returns:
            DataFrame with customer_id, churn_label, churned_date
        """
        snapshot_dt = datetime.strptime(snapshot_date, '%Y-%m-%d')
        
        # Get eligible customers
        eligible_customers = self._get_eligible_customers(raw_data['customers'], snapshot_dt)
        
        # Define lookahead window
        lookahead_start = snapshot_dt + timedelta(days=1)
        lookahead_end = snapshot_dt + timedelta(days=self.config.churn_horizon_days)
        
        # Get qualifying activity in lookahead window
        qualifying_activity = self._get_qualifying_activity(
            raw_data, lookahead_start, lookahead_end
        )
        
        # Create labels
        labels_df = self._create_labels(eligible_customers, qualifying_activity, snapshot_date)
        
        # Validate churn rates
        self._validate_churn_rates(labels_df)
        
        return labels_df
    
    def backfill_true_churn_labels(self, historical_snapshots: DataFrame, raw_data: Dict[str, DataFrame]) -> DataFrame:
        """
        Backfill true churn labels for historical snapshots
        
        Returns:
            DataFrame with updated churn labels
        """
        # Get all snapshot dates that need backfilling
        snapshot_dates = historical_snapshots.select("snapshot_date").distinct().collect()
        
        updated_labels = []
        
        for row in snapshot_dates:
            snapshot_date = row['snapshot_date']
            snapshot_dt = datetime.strptime(snapshot_date, '%Y-%m-%d')
            
            # Check if enough time has passed for true label
            current_date = datetime.now()
            if (current_date - snapshot_dt).days >= self.config.churn_horizon_days:
                
                # Generate true labels for this snapshot
                true_labels = self.generate_churn_labels(raw_data, snapshot_date)
                updated_labels.append(true_labels)
        
        if updated_labels:
            return reduce(lambda df1, df2: df1.union(df2), updated_labels)
        else:
            return self.spark.createDataFrame([], schema=historical_snapshots.schema)
    
    def _get_eligible_customers(self, customers_df: DataFrame, snapshot_date: datetime) -> DataFrame:
        """Get customers eligible for churn labeling"""
        min_join_date = snapshot_date - timedelta(days=self.config.minimum_tenure_days)
        
        return customers_df.filter(col("join_date") <= min_join_date).select("customer_id")
    
    def _get_qualifying_activity(self, raw_data: Dict[str, DataFrame], start_date: datetime, end_date: datetime) -> DataFrame:
        """Get customers with qualifying activity in the lookahead window"""
        # Transaction activity
        transaction_activity = raw_data['transactions'].filter(
            (col("timestamp") >= start_date) & 
            (col("timestamp") <= end_date)
        ).select("customer_id").distinct()
        
        # Interaction activity with session duration > 0
        interaction_activity = raw_data['interactions'].filter(
            (col("timestamp") >= start_date) & 
            (col("timestamp") <= end_date) &
            (col("session_duration") > 0)
        ).select("customer_id").distinct()
        
        # Union both activity types
        return transaction_activity.union(interaction_activity).distinct()
    
    def _create_labels(self, eligible_customers: DataFrame, qualifying_activity: DataFrame, snapshot_date: str) -> DataFrame:
        """Create churn labels based on activity"""
        # Mark customers with activity as non-churned (0)
        non_churned = qualifying_activity.withColumn("churn_label", lit(0)) \
                                        .withColumn("churned_date", lit(None).cast("date"))
        
        # Mark customers without activity as churned (1)
        churned = eligible_customers.join(qualifying_activity, "customer_id", "left_anti") \
                                   .withColumn("churn_label", lit(1)) \
                                   .withColumn("churned_date", lit(snapshot_date).cast("date"))
        
        # Combine labels
        return non_churned.union(churned).withColumn("snapshot_date", lit(snapshot_date))
    
    def _validate_churn_rates(self, labels_df: DataFrame) -> None:
        """Validate that churn rates are within expected ranges"""
        total_customers = labels_df.count()
        churned_customers = labels_df.filter(col("churn_label") == 1).count()
        
        churn_rate = churned_customers / total_customers
        
        if not (0.02 <= churn_rate <= 0.15):
            print(f"WARNING: Churn rate {churn_rate:.3f} outside expected range (2-15%)")
        
        print(f"Churn rate: {churn_rate:.3f} ({churned_customers}/{total_customers})")