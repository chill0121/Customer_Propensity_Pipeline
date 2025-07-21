from pyspark.sql.functions import col, when
from pyspark.sql import DataFrame
from datetime import datetime
from typing import Dict

from config.pipeline_config_loader import load_pipeline_config, load_spark_config, create_spark_session

pipeline_config = load_pipeline_config("config/pipeline_config.dev.yaml")
spark_config = load_spark_config("config/spark_config.yaml")

# Create SparkSession
spark_session = create_spark_session(spark_config)

class FeatureEngineeringService:
    """
    Weekly execution: Feature engineering and snapshot generation
    """
    
    def __init__(self, pipeline_config):
        self.spark = spark_session
        self.config = pipeline_config
    
    def create_weekly_snapshot(self, raw_data: Dict[str, DataFrame], snapshot_date: str) -> DataFrame:
        """
        Create weekly customer snapshot with all features
        
        Returns:
            DataFrame with customer_id, snapshot_date, and all features
        """
        snapshot_dt = datetime.strptime(snapshot_date, '%Y-%m-%d')
        
        # Get eligible customers (minimum tenure)
        eligible_customers = self._get_eligible_customers(raw_data['customers'], snapshot_dt)
        
        # Get historical data for feature engineering
        historical_data = self._get_historical_data(raw_data, snapshot_dt)
        
        # Engineer features
        features_df = self.engineer_features(eligible_customers, historical_data, snapshot_dt)
        
        # Add snapshot metadata
        features_df = features_df.withColumn("snapshot_date", lit(snapshot_date))
        
        # Validate feature quality
        self._validate_feature_quality(features_df)
        
        return features_df
    
    def engineer_features(self, customers_df: DataFrame, historical_data: Dict[str, DataFrame], snapshot_date: datetime) -> DataFrame:
        """
        Generate all features efficiently in single pass
        
        Returns:
            DataFrame with engineered features
        """
        # Start with customer base
        features_df = customers_df
        
        # Add demographic features
        features_df = self._add_demographic_features(features_df, snapshot_date)
        
        # Add transaction features
        features_df = self._add_transaction_features(features_df, historical_data['transactions'], snapshot_date)
        
        # Add interaction features
        features_df = self._add_interaction_features(features_df, historical_data['interactions'], snapshot_date)
        
        # Add derived features
        features_df = self._add_derived_features(features_df)
        
        return features_df
    
    def _get_eligible_customers(self, customers_df: DataFrame, snapshot_date: datetime) -> DataFrame:
        """Get customers with minimum tenure"""
        min_join_date = snapshot_date - timedelta(days=self.config.minimum_tenure_days)
        
        return customers_df.filter(col("join_date") <= min_join_date)
    
    def _get_historical_data(self, raw_data: Dict[str, DataFrame], snapshot_date: datetime) -> Dict[str, DataFrame]:
        """Filter historical data for feature engineering"""
        lookback_start = snapshot_date - timedelta(days=self.config.feature_window_days)
        
        historical_data = {}
        
        # Filter transactions
        historical_data['transactions'] = raw_data['transactions'].filter(
            (col("timestamp") >= lookback_start) & 
            (col("timestamp") <= snapshot_date)
        )
        
        # Filter interactions
        historical_data['interactions'] = raw_data['interactions'].filter(
            (col("timestamp") >= lookback_start) & 
            (col("timestamp") <= snapshot_date)
        )
        
        return historical_data
    
    def _add_demographic_features(self, customers_df: DataFrame, snapshot_date: datetime) -> DataFrame:
        """Add demographic and tenure features"""
        return customers_df.withColumn(
            "tenure_days",
            datediff(lit(snapshot_date), col("join_date"))
        ).withColumn(
            "active_products_count",
            col("has_credit_card").cast("int") + 
            col("has_loan").cast("int") + 
            col("has_checking").cast("int") + 
            col("has_savings").cast("int")
        )
    
    def _add_transaction_features(self, customers_df: DataFrame, transactions_df: DataFrame, snapshot_date: datetime) -> DataFrame:
        """Add transaction-based features"""
        # Add days_ago for efficient time-based filtering
        transactions_with_days = transactions_df.withColumn(
            "days_ago",
            datediff(lit(snapshot_date), col("timestamp"))
        )
        
        # Single-pass aggregation with multiple time windows
        txn_features = transactions_with_days.groupBy("customer_id").agg(
            # Volume patterns
            count(when(col("days_ago") <= 30, True)).alias("txn_count_30d"),
            count(when(col("days_ago") <= 60, True)).alias("txn_count_60d"),
            count(when(col("days_ago") <= 90, True)).alias("txn_count_90d"),
            
            # Amount patterns
            sum(when(col("days_ago") <= 30, col("amount"))).alias("txn_amount_30d"),
            sum(when(col("days_ago") <= 60, col("amount"))).alias("txn_amount_60d"),
            sum(when(col("days_ago") <= 90, col("amount"))).alias("txn_amount_90d"),
            
            # Average amounts
            avg(when(col("days_ago") <= 30, col("amount"))).alias("avg_txn_amount_30d"),
            
            # Recency
            min("days_ago").alias("days_since_last_txn"),
            min(when(col("txn_type") == "deposit", col("days_ago"))).alias("days_since_last_deposit"),
            
            # Product-specific features
            count(when((col("days_ago") <= 30) & (col("product") == "checking"), True)).alias("checking_txn_count_30d"),
            count(when((col("days_ago") <= 30) & (col("product") == "credit_card"), True)).alias("credit_card_txn_count_30d"),
            count(when((col("days_ago") <= 30) & (col("product") == "savings"), True)).alias("savings_txn_count_30d"),
            
            # Transaction type patterns
            count(when((col("days_ago") <= 30) & (col("txn_type") == "deposit"), True)).alias("deposit_count_30d"),
            count(when((col("days_ago") <= 30) & (col("txn_type") == "withdrawal"), True)).alias("withdrawal_count_30d"),
            count(when((col("days_ago") <= 30) & (col("txn_type") == "fee"), True)).alias("fee_count_30d")
        )
        
        # Add trend features
        txn_features = txn_features.withColumn(
            "txn_trend_30d_vs_60d",
            when(col("txn_count_60d") > 0, 
                 col("txn_count_30d") / col("txn_count_60d")
            ).otherwise(0)
        )
        
        # Join with customers
        return customers_df.join(txn_features, "customer_id", "left")
    
    def _add_interaction_features(self, customers_df: DataFrame, interactions_df: DataFrame, snapshot_date: datetime) -> DataFrame:
        """Add digital engagement features"""
        # Add days_ago for efficient filtering
        interactions_with_days = interactions_df.withColumn(
            "days_ago",
            datediff(lit(snapshot_date), col("timestamp"))
        )
        
        # Aggregate interaction features
        interaction_features = interactions_with_days.groupBy("customer_id").agg(
            # Login patterns
            count(when((col("days_ago") <= 30) & (col("interaction_type") == "login"), True)).alias("login_count_30d"),
            count(when((col("days_ago") <= 60) & (col("interaction_type") == "login"), True)).alias("login_count_60d"),
            count(when((col("days_ago") <= 90) & (col("interaction_type") == "login"), True)).alias("login_count_90d"),
            
            # Digital recency
            min(when(col("interaction_type") == "login", col("days_ago"))).alias("days_since_last_login"),
            
            # Support interactions (risk signal)
            count(when((col("days_ago") <= 30) & (col("interaction_type") == "support"), True)).alias("support_interactions_30d"),
            
            # Channel usage
            count(when((col("days_ago") <= 30) & (col("interaction_type") == "mobile"), True)).alias("mobile_usage_30d"),
            count(when((col("days_ago") <= 30) & (col("interaction_type") == "web"), True)).alias("web_usage_30d"),
            
            # Session duration metrics
            avg(when((col("days_ago") <= 30) & (col("session_duration") > 0), col("session_duration"))).alias("avg_session_duration_30d"),
            max(when(col("days_ago") <= 30, col("session_duration"))).alias("max_session_duration_30d")
        )
        
        # Add engagement trends
        interaction_features = interaction_features.withColumn(
            "login_trend_30d_vs_60d",
            when(col("login_count_60d") > 0,
                 col("login_count_30d") / col("login_count_60d")
            ).otherwise(0)
        ).withColumn(
            "mobile_usage_ratio_30d",
            when((col("mobile_usage_30d") + col("web_usage_30d")) > 0,
                 col("mobile_usage_30d") / (col("mobile_usage_30d") + col("web_usage_30d"))
            ).otherwise(0)
        )
        
        # Join with customers
        return customers_df.join(interaction_features, "customer_id", "left")
    
    def _add_derived_features(self, features_df: DataFrame) -> DataFrame:
        """Add derived and relationship features"""
        return features_df.withColumn(
            "relationship_score",
            col("active_products_count") * 2 + 
            when(col("avg_txn_amount_30d") > 1000, 1).otherwise(0) +
            when(col("login_count_30d") > 10, 1).otherwise(0)
        ).withColumn(
            "digital_engagement_score",
            col("login_count_30d") + col("mobile_usage_30d") + col("web_usage_30d")
        ).fillna(0)  # Fill nulls with 0 for all features
    
    def _validate_feature_quality(self, features_df: DataFrame) -> None:
        """Validate feature quality"""
        total_customers = features_df.count()
        
        # Check for excessive nulls
        for column in features_df.columns:
            if column not in ["customer_id", "snapshot_date"]:
                null_count = features_df.filter(col(column).isNull()).count()
                null_ratio = null_count / total_customers
                
                if null_ratio > 0.5:
                    raise ValueError(f"Feature {column} has {null_ratio:.2%} null values")
                elif null_ratio > 0.1:
                    print(f"WARNING: Feature {column} has {null_ratio:.2%} null values")
        
        print(f"Feature quality validation passed for {total_customers} customers")