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
        # ).withColumn(
        #     # Income estimation (since we removed income_bracket)
        #     "estimated_income_tier",
        #     when(col("city").isin(["New York", "San Francisco", "Seattle"]), "high")
        #     .when(col("city").isin(["Los Angeles", "Chicago", "Boston"]), "medium-high")
        #     .otherwise("medium")
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
                                   prediction_horizon_days: int = 30) -> DataFrame:
        """
        Create feature store dataset with target variable for ML models
        """
        
        # Prepare churn labels with prediction horizon
        churn_labels = churn_df.withColumn(
            "churn_prediction_date",
            date_sub(col("churn_date"), prediction_horizon_days)
        ).select(
            col("customer_id"),
            col("churn_prediction_date").alias("snapshot_date"),
            lit(1).alias("will_churn_30d"),
            col("churn_date"),
            col("churn_reason")
        )
        
        # Join features with churn labels
        feature_store = customer_360_df.join(
            churn_labels,
            on=["customer_id", "snapshot_date"],
            how="left"
        ).fillna({"will_churn_30d": 0})
        
        # Add feature versioning
        feature_store = feature_store.withColumn(
            "feature_version",
            lit("v1.0")
        ).withColumn(
            "created_timestamp",
            current_timestamp()
        )
        
        # Ensure proper data types for ML
        feature_store = feature_store.withColumn(
            "will_churn_30d",
            col("will_churn_30d").cast(IntegerType())
        )
        
        return feature_store


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
    data_dict = fe.load_data(data_path=RAW_DATA_PATH)#customer_df, transaction_df, interaction_df, churn_df)
    
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
    
    # Create feature store dataset
    print("Creating feature store dataset...")
    feature_store = fe.create_feature_store_dataset(
        customer_360_df=snapshots,
        churn_df=data_dict["churn"],
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

# Utility functions for loading saved data
def load_customer_360(spark: SparkSession, data_path: str = PROCESSED_PATH / "customer_360") -> DataFrame:
    """
    Load saved customer 360 snapshots from parquet
    """
    return spark.read.parquet(data_path)

def load_feature_store(spark: SparkSession, data_path: str = PROCESSED_PATH / "feature_store") -> DataFrame:
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

def sample_pipeline_execution():
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

    
    
    # Run pipeline (uncomment when you have your data)
    feature_store, customer_360 = run_feature_engineering_pipeline(
        spark=spark,
        customer_df=None,#customer_df,
        transaction_df=None,#transaction_df,
        interaction_df=None,#interaction_df,
        churn_df=None,#churn_df,
        output_path=output_path,
        data_path=RAW_DATA_PATH
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
sample_pipeline_execution()