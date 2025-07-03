from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Initialize Spark session
def get_spark_session():
    return SparkSession.builder \
        .appName("CreditUnionFeatureEngineering") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # go up if needed
RAW_DATA_PATH = BASE_DIR / "data" / "raw"
PROCESSED_PATH = BASE_DIR / "data" / "processed"

class CreditUnionFeatureEngineering:
    """
    PySpark-based feature engineering pipeline for credit union data
    Designed for rolling snapshots and ML model preparation
    """
    
    def __init__(self, spark_session: SparkSession):
        self.spark = spark_session
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    # def load_data(self, 
    #               customer_df: DataFrame,
    #               transaction_df: DataFrame, 
    #               interaction_df: DataFrame,
    #               churn_df: DataFrame) -> dict[str, DataFrame]:
    #     """Load and validate input dataframes"""
        
    #     # Cache frequently accessed dataframes
    #     customer_df.cache()
    #     transaction_df.cache()
    #     interaction_df.cache()
    #     churn_df.cache()
        
    #     self.logger.info(f"Loaded data - Customers: {customer_df.count()}, "
    #                     f"Transactions: {transaction_df.count()}, "
    #                     f"Interactions: {interaction_df.count()}")
        
    #     return {
    #         'customers': customer_df,
    #         'transactions': transaction_df,
    #         'interactions': interaction_df,
    #         'churn': churn_df
    #     }
    def load_data(self, 
                  customer_df: DataFrame = None,
                  transaction_df: DataFrame = None, 
                  interaction_df: DataFrame = None,
                  churn_df: DataFrame = None,
                  data_path: str = None) -> dict[str, DataFrame]:
        """
        Load and validate input dataframes
        
        Args:
            customer_df, transaction_df, interaction_df, churn_df: Pre-loaded DataFrames
            data_path: Path to parquet files (alternative to pre-loaded DataFrames)
            
        Returns:
            Dictionary of cached DataFrames
        """
        
        # Option 1: Load from parquet files
        if data_path is not None:
            self.logger.info(f"Loading data from parquet files at: {data_path}")
            
            try:
                customer_df = self.spark.read.parquet(f"{data_path}/customers.parquet")
                transaction_df = self.spark.read.parquet(f"{data_path}/transactions.parquet")
                interaction_df = self.spark.read.parquet(f"{data_path}/interactions.parquet")
                churn_df = self.spark.read.parquet(f"{data_path}/churn_labels.parquet")
                
                self.logger.info("Successfully loaded all parquet files")
                
            except Exception as e:
                self.logger.error(f"Failed to load parquet files: {e}")
                raise
        
        # Option 2: Use pre-loaded DataFrames
        elif all(df is not None for df in [customer_df, transaction_df, interaction_df, churn_df]):
            self.logger.info("Using pre-loaded DataFrames")
            
        else:
            raise ValueError("Either provide all DataFrames or specify data_path for parquet files")
        
        # Validate that we have data
        if any(df is None for df in [customer_df, transaction_df, interaction_df, churn_df]):
            raise ValueError("One or more DataFrames are None after loading")
        
        # Cache frequently accessed dataframes
        customer_df.cache()
        transaction_df.cache()
        interaction_df.cache()
        churn_df.cache()
        
        # Log data counts
        try:
            customer_count = customer_df.count()
            transaction_count = transaction_df.count()
            interaction_count = interaction_df.count()
            churn_count = churn_df.count()
            
            self.logger.info(f"Loaded data - Customers: {customer_count:,}, "
                            f"Transactions: {transaction_count:,}, "
                            f"Interactions: {interaction_count:,}, "
                            f"Churn: {churn_count:,}")
        except Exception as e:
            self.logger.warning(f"Could not count rows: {e}")
        
        return {
            'customers': customer_df,
            'transactions': transaction_df,
            'interactions': interaction_df,
            'churn': churn_df
        }
    
    def create_transaction_features(self, 
                                  transaction_df: DataFrame,
                                  snapshot_date: str,
                                  lookback_days: int = 90) -> DataFrame:
        """
        Create comprehensive transaction-based features for a given snapshot date
        """
        
        snapshot_dt = to_date(lit(snapshot_date))
        cutoff_date = date_sub(snapshot_dt, lookback_days)
        
        # Filter transactions within lookback window
        recent_transactions = transaction_df.filter(
            (col("timestamp") <= snapshot_dt) & 
            (col("timestamp") >= cutoff_date)
        )
        
        # Define window specifications
        customer_window = Window.partitionBy("customer_id")
        customer_product_window = Window.partitionBy("customer_id", "product")
        
        # Aggregate transaction features
        transaction_features = recent_transactions.groupBy("customer_id").agg(
            # Volume metrics
            count("*").alias("tx_count_90d"),
            sum("amount").alias("tx_total_amount_90d"),
            avg("amount").alias("tx_avg_amount_90d"),
            stddev("amount").alias("tx_std_amount_90d"),
            min("amount").alias("tx_min_amount_90d"),
            max("amount").alias("tx_max_amount_90d"),
            
            # Timing metrics
            max("timestamp").alias("last_tx_date"),
            min("timestamp").alias("first_tx_date"),
            count_distinct("timestamp").alias("tx_active_days_90d"),
            
            # Transaction type diversity
            count_distinct("txn_type").alias("tx_type_diversity_90d"),
            count_distinct("product").alias("product_diversity_90d"),
            
            # Debit/Credit patterns
            sum(when(col("amount") > 0, col("amount")).otherwise(0)).alias("total_credits_90d"),
            sum(when(col("amount") < 0, abs(col("amount"))).otherwise(0)).alias("total_debits_90d"),
            sum(when(col("amount") > 0, 1).otherwise(0)).alias("credit_count_90d"),
            sum(when(col("amount") < 0, 1).otherwise(0)).alias("debit_count_90d")
        )
        
        # Product-specific features
        product_features = recent_transactions.groupBy("customer_id", "product").agg(
            count("*").alias("tx_count"),
            sum("amount").alias("tx_amount"),
            avg("amount").alias("tx_avg_amount")
        ).groupBy("customer_id").pivot("product").agg(
            first("tx_count").alias("count"),
            first("tx_amount").alias("amount"),
            first("tx_avg_amount").alias("avg_amount")
        )
        
        # Calculate derived features
        enhanced_features = transaction_features.withColumn(
            "days_since_last_tx", 
            datediff(snapshot_dt, col("last_tx_date"))
        ).withColumn(
            "tx_frequency_90d",
            col("tx_count_90d") / greatest(col("tx_active_days_90d"), lit(1))
        ).withColumn(
            "credit_debit_ratio",
            when(col("total_debits_90d") > 0, 
                 col("total_credits_90d") / col("total_debits_90d")).otherwise(0)
        ).withColumn(
            "avg_days_between_tx",
            when(col("tx_count_90d") > 1,
                 datediff(col("last_tx_date"), col("first_tx_date")) / (col("tx_count_90d") - 1)
            ).otherwise(0)
        ).withColumn(
            "snapshot_date", 
            snapshot_dt
        )
        
        # Join product-specific features
        if product_features.count() > 0:
            enhanced_features = enhanced_features.join(
                product_features, 
                on="customer_id", 
                how="left"
            )
        
        return enhanced_features
    
    def create_interaction_features(self, 
                                  interaction_df: DataFrame,
                                  snapshot_date: str,
                                  lookback_days: int = 90) -> DataFrame:
        """
        Create customer interaction and engagement features
        """
        
        snapshot_dt = to_date(lit(snapshot_date))
        cutoff_date = date_sub(snapshot_dt, lookback_days)
        
        recent_interactions = interaction_df.filter(
            (col("timestamp") <= snapshot_dt) & 
            (col("timestamp") >= cutoff_date)
        )
        
        interaction_features = recent_interactions.groupBy("customer_id").agg(
            # Login patterns
            sum(when(col("interaction_type") == "login", 1).otherwise(0)).alias("login_count_90d"),
            max(when(col("interaction_type") == "login", col("timestamp"))).alias("last_login_date"),
            
            # Support interactions
            sum(when(col("interaction_type") == "support_call", 1).otherwise(0)).alias("support_calls_90d"),
            sum(when(col("interaction_type") == "support_email", 1).otherwise(0)).alias("support_emails_90d"),
            sum(when(col("interaction_type") == "support_chat", 1).otherwise(0)).alias("support_chats_90d"),
            
            # Digital engagement
            sum(when(col("interaction_type") == "mobile_app", 1).otherwise(0)).alias("mobile_sessions_90d"),
            sum(when(col("interaction_type") == "web_portal", 1).otherwise(0)).alias("web_sessions_90d"),
            
            # Overall engagement
            count("*").alias("total_interactions_90d"),
            count_distinct("interaction_type").alias("interaction_diversity_90d"),
            count_distinct("timestamp").alias("active_interaction_days_90d")
        ).withColumn(
            "days_since_last_login",
            datediff(snapshot_dt, col("last_login_date"))
        ).withColumn(
            "total_support_interactions_90d",
            col("support_calls_90d") + col("support_emails_90d") + col("support_chats_90d")
        ).withColumn(
            "digital_engagement_score",
            (col("mobile_sessions_90d") + col("web_sessions_90d")) / greatest(col("total_interactions_90d"), lit(1))
        ).withColumn(
            "support_intensity_score",
            col("total_support_interactions_90d") / greatest(col("total_interactions_90d"), lit(1))
        ).withColumn(
            "snapshot_date",
            snapshot_dt
        )
        
        return interaction_features
    
    def create_customer_tenure_features(self, 
                                      customer_df: DataFrame,
                                      snapshot_date: str) -> DataFrame:
        """
        Create customer tenure and demographic-derived features
        """
        
        snapshot_dt = to_date(lit(snapshot_date))
        
        tenure_features = customer_df.withColumn(
            "account_age_days",
            datediff(snapshot_dt, col("join_date"))
        ).withColumn(
            "account_age_months",
            months_between(snapshot_dt, col("join_date"))
        ).withColumn(
            "account_age_years",
            col("account_age_months") / 12
        ).withColumn(
            # Age-based features
            "age_group",
            when(col("age") < 25, "18-24")
            .when(col("age") < 35, "25-34")
            .when(col("age") < 50, "35-49")
            .when(col("age") < 65, "50-64")
            .otherwise("65+")
        ).withColumn(
            "snapshot_date",
            snapshot_dt
        )
        
        return tenure_features
    
    def create_customer_360_snapshot(self,
                                  data_dict: dict[str, DataFrame],
                                  snapshot_date: str,
                                  lookback_days: int = 90) -> DataFrame:
        """
        Create comprehensive customer 360 view for a specific snapshot date.
        Ensures no column name ambiguity (e.g., snapshot_date).
        """

        self.logger.info(f"Creating customer 360 snapshot for {snapshot_date}")

        # Step 1: Create feature sets
        tx_features = self.create_transaction_features(
            data_dict['transactions'], snapshot_date, lookback_days
        ).drop("snapshot_date")  # Remove to avoid ambiguity

        interaction_features = self.create_interaction_features(
            data_dict['interactions'], snapshot_date, lookback_days
        ).drop("snapshot_date")  # Remove to avoid ambiguity

        customer_features = self.create_customer_tenure_features(
            data_dict['customers'], snapshot_date
        )  # Keep snapshot_date here

        # Step 2: Join all features on customer_id
        customer_360 = customer_features.join(
            tx_features, on="customer_id", how="left"
        ).join(
            interaction_features, on="customer_id", how="left"
        )

        # Step 3: Fill nulls in numeric columns with 0 (except customer_id)
        numeric_columns = [field.name for field in customer_360.schema.fields
                        if isinstance(field.dataType, (IntegerType, DoubleType, FloatType)) and field.name != "customer_id"]

        for col_name in numeric_columns:
            customer_360 = customer_360.fillna({col_name: 0})

        # Step 4: Add risk indicators and engagement score
        customer_360 = customer_360.withColumn(
            "high_risk_flag",
            when(
                (col("days_since_last_tx") > 30) |
                (col("days_since_last_login") > 45) |
                (col("total_support_interactions_90d") > 5),
                1
            ).otherwise(0)
        ).withColumn(
            "engagement_score",
            (col("tx_frequency_90d") * 0.4 +
            col("digital_engagement_score") * 0.3 +
            (1 - col("support_intensity_score")) * 0.3)
        )

        return customer_360

    
    def create_rolling_snapshots(self,
                               data_dict: dict[str, DataFrame],
                               start_date: str,
                               end_date: str,
                               frequency_days: int = 30,
                               lookback_days: int = 90) -> DataFrame:
        """
        Create rolling snapshots for multiple time periods
        """
        
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        all_snapshots = []
        current_date = start_dt
        
        while current_date <= end_dt:
            snapshot_date_str = current_date.strftime("%Y-%m-%d")
            
            self.logger.info(f"Processing snapshot for {snapshot_date_str}")
            
            snapshot = self.create_customer_360_snapshot(
                data_dict, snapshot_date_str, lookback_days
            )
            
            all_snapshots.append(snapshot)
            current_date += timedelta(days=frequency_days)
        
        # Union all snapshots
        combined_snapshots = all_snapshots[0]
        for snapshot in all_snapshots[1:]:
            combined_snapshots = combined_snapshots.union(snapshot)
        
        return combined_snapshots
    
    def create_feature_store_dataset(self,
                                       customer_360_df: DataFrame,
                                       churn_df: DataFrame,
                                       transaction_df: DataFrame,
                                       prediction_horizon_days: int = 30) -> DataFrame:
        """
        Create feature store using transaction recency to define churn timing
        
        This method uses the last transaction date to simulate when customers actually churned,
        then creates proper point-in-time labels for training.
        """
        
        # For churned customers, find their last transaction date
        churned_customers = churn_df.filter(col("churned") == 1).select("customer_id")
        
        # Get last transaction date for each customer
        last_transactions = transaction_df.groupBy("customer_id").agg(
            max("timestamp").alias("last_transaction_date")
        )
        
        # For churned customers, assume they churned shortly after their last transaction
        churned_with_dates = churned_customers.join(
            last_transactions, 
            on="customer_id", 
            how="inner"
        ).withColumn(
            "estimated_churn_date",
            date_add(col("last_transaction_date"), 15)  # Assume churn 15 days after last transaction
        )
        
        # Create point-in-time labels
        # Label customers as "will_churn" for snapshots that are prediction_horizon_days before churn
        churn_labels = churned_with_dates.withColumn(
            "label_date",
            date_sub(col("estimated_churn_date"), prediction_horizon_days)
        ).select(
            col("customer_id"),
            col("label_date").alias("snapshot_date"),
            lit(1).alias("will_churn_30d"),
            col("estimated_churn_date").alias("churn_date")
        )
        
        # Only keep labels that fall within our snapshot date range
        snapshot_dates = customer_360_df.select("snapshot_date").distinct()
        valid_churn_labels = churn_labels.join(
            snapshot_dates,
            on="snapshot_date",
            how="inner"
        )
        
        # Join with customer 360 features
        feature_store = customer_360_df.join(
            valid_churn_labels,
            on=["customer_id", "snapshot_date"],
            how="left"
        ).fillna({"will_churn_30d": 0})
        
        # Add metadata
        feature_store = feature_store.withColumn(
            "feature_version", lit("v2.0")
        ).withColumn(
            "created_timestamp", current_timestamp()
        ).withColumn(
            "prediction_horizon_days", lit(prediction_horizon_days)
        ).withColumn(
            "will_churn_30d", col("will_churn_30d").cast(IntegerType())
        )
        
        return feature_store

    def create_ml_ready_feature_store(self,
                                 customer_df: DataFrame,
                                 transaction_df: DataFrame,
                                 interaction_df: DataFrame,
                                 churn_df: DataFrame,
                                 prediction_horizon_days: int = 30,
                                 feature_lookback_days: int = 90,
                                 temporal_gap_days: int = 7) -> DataFrame:
        """
        Create ML-ready feature store with proper temporal separation to prevent data leakage
        
        Timeline:
        [Feature Window] -> [Gap] -> [Observation Window] -> [Prediction Point]
        
        Args:
            prediction_horizon_days: How far ahead we want to predict churn (30 days)
            feature_lookback_days: How far back to look for features (90 days)
            temporal_gap_days: Gap between feature window and churn observation (7 days)
        """
        
        # Step 1: Determine churn dates from transaction patterns
        last_transactions = transaction_df.groupBy("customer_id").agg(
            max("timestamp").alias("last_transaction_date")
        )
        
        churned_customers = churn_df.filter(col("churned") == 1).join(
            last_transactions, on="customer_id", how="inner"
        ).withColumn(
            # Assume churn happened 15 days after last transaction
            "estimated_churn_date",
            date_add(col("last_transaction_date"), 15)
        )
        
        # Step 2: Create prediction points (when we want to make predictions)
        prediction_points = churned_customers.withColumn(
            "prediction_date",
            date_sub(col("estimated_churn_date"), prediction_horizon_days)
        ).select(
            col("customer_id"),
            col("prediction_date"),
            lit(1).alias("will_churn_30d"),
            col("estimated_churn_date").alias("churn_date")
        )
        
        # Step 3: Create feature calculation dates (BEFORE the prediction point)
        feature_dates = prediction_points.withColumn(
            "feature_end_date",
            date_sub(col("prediction_date"), temporal_gap_days)
        ).withColumn(
            "feature_start_date", 
            date_sub(col("feature_end_date"), feature_lookback_days)
        ).select(
            col("customer_id"),
            col("prediction_date"),
            col("feature_end_date"),
            col("feature_start_date"),
            col("will_churn_30d")
        )
        
        self.logger.info("Creating temporally separated features...")
        
        # Step 4: Calculate features using the separated time windows
        ml_features = []
        
        # Get unique feature calculation periods
        feature_periods = feature_dates.select(
            "customer_id", "feature_start_date", "feature_end_date", "prediction_date"
        ).collect()
        
        for row in feature_periods:
            customer_id = row['customer_id']
            start_date = row['feature_start_date']
            end_date = row['feature_end_date']
            pred_date = row['prediction_date']
            
            # Calculate transaction features for this specific time window
            customer_transactions = transaction_df.filter(
                (col("customer_id") == customer_id) &
                (col("timestamp") >= start_date) &
                (col("timestamp") <= end_date)
            )
            
            tx_features = customer_transactions.agg(
                count("*").alias("tx_count"),
                coalesce(sum("amount"), lit(0)).alias("tx_total_amount"),
                coalesce(avg("amount"), lit(0)).alias("tx_avg_amount"),
                coalesce(stddev("amount"), lit(0)).alias("tx_std_amount"),
                coalesce(count_distinct("txn_type"), lit(0)).alias("tx_type_diversity"),
                coalesce(count_distinct("product"), lit(0)).alias("product_diversity"),
                coalesce(max("timestamp"), lit(start_date)).alias("last_tx_in_window")
            ).withColumn("customer_id", lit(customer_id)) \
            .withColumn("prediction_date", lit(pred_date)) \
            .withColumn("feature_window_start", lit(start_date)) \
            .withColumn("feature_window_end", lit(end_date))
            
            # Calculate interaction features for this time window  
            customer_interactions = interaction_df.filter(
                (col("customer_id") == customer_id) &
                (col("timestamp") >= start_date) &
                (col("timestamp") <= end_date)
            )
            
            int_features = customer_interactions.agg(
                coalesce(sum(when(col("interaction_type") == "login", 1).otherwise(0)), lit(0)).alias("login_count"),
                coalesce(sum(when(col("interaction_type").contains("support"), 1).otherwise(0)), lit(0)).alias("support_count"),
                coalesce(count("*"), lit(0)).alias("total_interactions"),
                coalesce(count_distinct("interaction_type"), lit(0)).alias("interaction_diversity")
            ).withColumn("customer_id", lit(customer_id))
            
            # Combine features for this customer/time period
            combined_features = tx_features.join(int_features, on="customer_id", how="outer")
            ml_features.append(combined_features)
        
        # Union all customer features
        if ml_features:
            all_features = ml_features[0]
            for feature_df in ml_features[1:]:
                all_features = all_features.union(feature_df)
        else:
            # Create empty DataFrame with correct schema if no features
            schema = StructType([
                StructField("customer_id", StringType(), True),
                StructField("prediction_date", DateType(), True),
                StructField("tx_count", LongType(), True),
                StructField("tx_avg_amount", DoubleType(), True)
            ])
            all_features = self.spark.createDataFrame([], schema)
        
        # Step 5: Add customer demographics (these don't have temporal issues)
        customer_demographics = customer_df.select(
            "customer_id", "age", "state", "join_date"
        )
        
        # Step 6: Join everything together FIRST, then add calculated columns
        final_features = all_features.join(
            customer_demographics, on="customer_id", how="left"
        ).join(
            feature_dates.select("customer_id", "prediction_date", "will_churn_30d"),
            on=["customer_id", "prediction_date"], how="left"
        ).withColumn(
            # Now we can calculate account age because prediction_date exists
            "account_age_at_prediction",
            datediff(col("prediction_date"), col("join_date"))
        )
        
        # Step 7: Add negative examples (non-churned customers)
        # Sample non-churned customers at various prediction dates
        non_churned = churn_df.filter(col("churned") == 0).select("customer_id")
        
        # Create prediction dates for non-churned customers (sample from churned dates)
        sample_dates = prediction_points.select("prediction_date").distinct().sample(0.3)  # Sample 30% of dates
        
        non_churned_examples = non_churned.crossJoin(sample_dates).withColumn(
            "will_churn_30d", lit(0)
        )
        
        # Calculate features for non-churned customers using the same temporal logic
        # For brevity, adding a simplified version
        non_churned_features = non_churned_examples.join(
            customer_demographics, on="customer_id", how="left"
        ).withColumn(
            "account_age_at_prediction",
            datediff(col("prediction_date"), col("join_date"))
        ).withColumn("tx_count", lit(0)) \
        .withColumn("tx_avg_amount", lit(0.0)) \
        .withColumn("login_count", lit(0)) \
        .withColumn("support_count", lit(0)) \
        .withColumn("tx_total_amount", lit(0.0)) \
        .withColumn("tx_std_amount", lit(0.0)) \
        .withColumn("tx_type_diversity", lit(0)) \
        .withColumn("product_diversity", lit(0)) \
        .withColumn("total_interactions", lit(0)) \
        .withColumn("interaction_diversity", lit(0))
        
        # Combine churned and non-churned examples
        ml_dataset = final_features.union(
            non_churned_features.select(final_features.columns)
        )
        
        # Add metadata
        ml_dataset = ml_dataset.withColumn(
            "feature_version", lit("v3.0_temporal_separated")
        ).withColumn(
            "prediction_horizon_days", lit(prediction_horizon_days)
        ).withColumn(
            "feature_lookback_days", lit(feature_lookback_days)
        ).withColumn(
            "temporal_gap_days", lit(temporal_gap_days)
        ).withColumn(
            "created_timestamp", current_timestamp()
        ).withColumn(
            "data_leakage_prevented", lit(True)
        )
        
        return ml_dataset


# Example usage and utility functions
def run_feature_engineering_pipeline(spark: SparkSession,
                                    customer_df: DataFrame,
                                    transaction_df: DataFrame,
                                    interaction_df: DataFrame,
                                    churn_df: DataFrame,
                                    output_path: str = PROCESSED_PATH,
                                    data_path: str = None):
    """
    Complete feature engineering pipeline execution with parquet output
    """
    
    # Initialize feature engineering
    fe = CreditUnionFeatureEngineering(spark)
    
    # Load data
    if data_path is not None:
        data_dict = fe.load_data(data_path=data_path)#customer_df, transaction_df, interaction_df, churn_df)
    
    # Create rolling snapshots (monthly for 2 years)
    print("Creating customer 360 snapshots...")
    snapshots = fe.create_rolling_snapshots(
        data_dict=data_dict,
        start_date="2022-06-01",  # Allow some history
        end_date="2024-11-01",    # Before end date to allow churn prediction
        frequency_days=30,
        lookback_days=90
    )
    
    # Save customer 360 snapshots
    customer_360_path = f"{output_path}/customer_360"
    print(f"Saving customer 360 snapshots to {customer_360_path}")
    snapshots.coalesce(4).write.mode("overwrite").partitionBy("snapshot_date").parquet(customer_360_path)
    
    # # Create feature store dataset
    print("Creating feature store dataset...")
    feature_store = fe.create_feature_store_dataset(
        customer_360_df=snapshots,
        churn_df=data_dict["churn"],
        transaction_df=data_dict['transactions'],
        prediction_horizon_days=30
    )
    
    # Save feature store
    feature_store_path = f"{output_path}/feature_store"
    print(f"Saving feature store to {feature_store_path}")
    feature_store.coalesce(4).write.mode("overwrite").partitionBy("snapshot_date").parquet(feature_store_path)
    
    print("Pipeline completed successfully!")
    print(f"Customer 360 records: {snapshots.count()}")
    print(f"Feature store records: {feature_store.count()}")
    print(f"Churn rate in feature store: {feature_store.filter(col('will_churn_30d') == 1).count() / feature_store.count() * 100:.2f}%")
    
    return feature_store, snapshots

def run_ml_ready_pipeline(spark: SparkSession,
                         customer_df: DataFrame,
                         transaction_df: DataFrame,
                         interaction_df: DataFrame,
                         churn_df: DataFrame,
                         output_path: str = PROCESSED_PATH,
                         data_path: str = None):
    """
    Pipeline specifically designed for ML training with proper temporal separation
    """
    
    # Initialize feature engineering
    fe = CreditUnionFeatureEngineering(spark)

    # Load data
    if data_path is not None:
        data_dict = fe.load_data(data_path=data_path)#customer_df, transaction_df, interaction_df, churn_df)
    
    print("Creating ML-ready feature store with temporal separation...")
    
    # Create ML-ready dataset with proper time windows
    ml_dataset = fe.create_ml_ready_feature_store(
        customer_df=data_dict['customers'],
        transaction_df=data_dict['transactions'],
        interaction_df=data_dict['interactions'],
        churn_df=data_dict['churn'],
        prediction_horizon_days=30,      # Predict 30 days ahead
        feature_lookback_days=90,        # Use 90 days of historical data
        temporal_gap_days=7              # 7-day gap to prevent leakage
    )
    
    # Save ML-ready dataset
    ml_store_path = f"{output_path}/ml_feature_store"
    print(f"Saving ML-ready feature store to {ml_store_path}")
    ml_dataset.coalesce(2).write.mode("overwrite").parquet(ml_store_path)
    
    # Print dataset statistics
    total_records = ml_dataset.count()
    churn_records = ml_dataset.filter(col('will_churn_30d') == 1).count()
    churn_rate = (churn_records / total_records * 100) if total_records > 0 else 0
    
    print(f"ML Dataset Statistics:")
    print(f"  Total records: {total_records}")
    print(f"  Churn cases: {churn_records}")
    print(f"  Non-churn cases: {total_records - churn_records}")
    print(f"  Churn rate: {churn_rate:.2f}%")
    
    # Show feature columns for model training
    feature_columns = [col for col in ml_dataset.columns 
                      if col not in ['customer_id', 'will_churn_30d', 
                                   'prediction_date', 'churn_date', 'feature_version',
                                   'prediction_horizon_days', 'feature_lookback_days',
                                   'temporal_gap_days', 'created_timestamp', 
                                   'data_leakage_prevented', 'feature_window_start',
                                   'feature_window_end']]
    
    print(f"Feature columns for XGBoost: {feature_columns}")
    
    return ml_dataset, feature_columns

# Utility function to prepare data for XGBoost
def prepare_xgboost_data(ml_dataset: DataFrame, feature_columns: list):
    """
    Prepare data specifically for XGBoost training
    """
    
    # Select only relevant columns and handle nulls
    model_data = ml_dataset.select(
        ['customer_id', 'prediction_date', 'will_churn_30d'] + feature_columns
    ).fillna(0)  # XGBoost can handle some nulls, but let's be safe
    
    # Convert to Pandas for XGBoost (if dataset is small enough)
    if model_data.count() < 100000:  # Adjust threshold based on your memory
        pandas_df = model_data.toPandas()
        
        X = pandas_df[feature_columns]
        y = pandas_df['will_churn_30d']
        
        return X, y, pandas_df
    else:
        # For larger datasets, you'd want to use Spark MLlib or save to disk
        return model_data, None, None

# Utility functions for loading saved data
def load_customer_360(spark: SparkSession, data_path: str = str(PROCESSED_PATH / "customer_360")) -> DataFrame:
    """
    Load saved customer 360 snapshots from parquet
    """
    return spark.read.parquet(data_path)

def load_feature_store(spark: SparkSession, data_path: str = str(PROCESSED_PATH / "feature_store")) -> DataFrame:
    """
    Load saved feature store from parquet
    """
    return spark.read.parquet(data_path)

def get_latest_snapshot(customer_360_df: DataFrame) -> DataFrame:
    """
    Get the most recent snapshot from customer 360 data
    """
    latest_date = customer_360_df.agg(max("snapshot_date")).collect()[0][0]
    return customer_360_df.filter(col("snapshot_date") == latest_date)

def sample_pipeline_execution(data_path=None):
    """
    Example of how to run the complete pipeline
    """
    
    # Initialize Spark
    spark = get_spark_session()
    
    # Assuming you have your dataframes loaded already
    # customer_df, transaction_df, interaction_df, churn_df = load_your_simulated_data()
    
    # Create output directory
    output_path = PROCESSED_PATH
    output_path.mkdir(exist_ok=True)

    
    
    # Run pipeline
    feature_store, customer_360 = run_feature_engineering_pipeline(
        spark=spark,
        customer_df=None,#customer_df,
        transaction_df=None,#transaction_df,
        interaction_df=None,#interaction_df,
        churn_df=None,#churn_df,
        output_path=output_path,
        data_path=data_path
    )
    
    # Load saved data for analysis
    customer_360_loaded = load_customer_360(spark)
    feature_store_loaded = load_feature_store(spark)
    
    # Get latest snapshot for current analysis
    latest_snapshot = get_latest_snapshot(customer_360_loaded)
    
    print("Pipeline setup complete!")
    return spark

# Example feature importance analysis for churn model
def analyze_feature_importance(feature_store_df: DataFrame):
    """
    Basic feature analysis for churn modeling
    """
    
    # Calculate correlation with churn target
    numeric_features = [
        "tx_count_90d", "tx_avg_amount_90d", "days_since_last_tx",
        "login_count_90d", "support_calls_90d", "account_age_months",
        "engagement_score", "high_risk_flag"
    ]
    
    correlations = {}
    for feature in numeric_features:
        correlation = feature_store_df.stat.corr("will_churn_30d", feature)
        correlations[feature] = correlation
    
    return correlations

# run_feature_engineering_pipeline()
# sample_pipeline_execution(RAW_DATA_PATH)

def run_full_pipeline(spark: SparkSession,
                      customer_df: DataFrame = None,
                      transaction_df: DataFrame = None,
                      interaction_df: DataFrame = None,
                      churn_df: DataFrame = None,
                      output_path: str = PROCESSED_PATH,
                      data_path: str = None):
    """
    Run both feature engineering and ML-ready pipelines
    """
    
    print("=== Starting Full Pipeline Execution ===")
    
    feature_store, snapshots = run_feature_engineering_pipeline(
        spark=spark,
        customer_df=customer_df,
        transaction_df=transaction_df,
        interaction_df=interaction_df,
        churn_df=churn_df,
        output_path=output_path,
        data_path=data_path
    )
    
    ml_dataset, feature_columns = run_ml_ready_pipeline(
        spark=spark,
        customer_df=customer_df,
        transaction_df=transaction_df,
        interaction_df=interaction_df,
        churn_df=churn_df,
        output_path=output_path,
        data_path=data_path
    )
    
    print("=== Full Pipeline Execution Completed ===")
    
    return {
        "feature_store": feature_store,
        "customer_360_snapshots": snapshots,
        "ml_ready_dataset": ml_dataset,
        "feature_columns": feature_columns
    }

run_full_pipeline(spark=get_spark_session(), data_path=RAW_DATA_PATH)