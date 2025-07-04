from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
from pyspark import StorageLevel 
from datetime import datetime, timedelta
from pathlib import Path
import logging
from functools import reduce

# Initialize Spark session
def get_spark_session():
    return SparkSession.builder \
        .appName("CreditUnionFeatureEngineering") \
        .config("spark.driver.memory", "20g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.adaptive.skewJoin.enabled", "true") \
        .config("spark.sql.adaptive.localShuffleReader.enabled", "true") \
        .getOrCreate()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw"
PROCESSED_PATH = BASE_DIR / "data" / "processed"

class CreditUnionFeatureEngineering:
    """
    PySpark-based feature engineering pipeline for churn prediction
    Creates Customer 360 snapshots and ML-ready feature store
    """
    
    def __init__(self, spark_session: SparkSession):
        self.spark = spark_session
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def load_data(self, data_path: str = None, **dataframes) -> dict[str, DataFrame]:
        """Load and validate input dataframes with enhanced caching strategy"""
        
        if data_path is not None:
            self.logger.info(f"Loading data from parquet files at: {data_path}")
            try:
                data_dict = {
                    'customers': self.spark.read.parquet(f"{data_path}/customers.parquet"),
                    'transactions': self.spark.read.parquet(f"{data_path}/transactions.parquet"),
                    'interactions': self.spark.read.parquet(f"{data_path}/interactions.parquet"),
                    'churn': self.spark.read.parquet(f"{data_path}/churn_labels.parquet")
                }
                self.logger.info("Successfully loaded all parquet files")
            except Exception as e:
                self.logger.error(f"Failed to load parquet files: {e}")
                raise
        
        elif all(df is not None for df in [dataframes.get('customer_df'), 
                                          dataframes.get('transaction_df'),
                                          dataframes.get('interaction_df'),
                                          dataframes.get('churn_df')]):
            data_dict = {
                'customers': dataframes['customer_df'],
                'transactions': dataframes['transaction_df'],
                'interactions': dataframes['interaction_df'],
                'churn': dataframes['churn_df']
            }
            self.logger.info("Using pre-loaded DataFrames")
        
        else:
            raise ValueError("Either provide data_path or all required DataFrames")
        
        # Standardize column names and add derived columns
        data_dict = self._standardize_data_schemas(data_dict)
        
        # Enhanced caching strategy with storage level optimization
        for name, df in data_dict.items():
            df.cache()
            if name in ['transactions', 'interactions']:  # Large datasets
                df.persist(StorageLevel.MEMORY_AND_DISK)
            try:
                count = df.count()
                self.logger.info(f"Loaded {name}: {count:,} records")
            except Exception as e:
                self.logger.warning(f"Could not count {name}: {e}")
        
        return data_dict
    
    def _standardize_data_schemas(self, data_dict: dict[str, DataFrame]) -> dict[str, DataFrame]:
        """Standardize schemas and add derived columns"""
        
        # Standardize transactions
        transactions = data_dict['transactions']
        if 'timestamp' in transactions.columns:
            transactions = transactions.withColumn('tx_date', to_date(col('timestamp')))
            transactions = transactions.withColumn('tx_datetime', col('timestamp'))
        
        # Standardize interactions  
        interactions = data_dict['interactions']
        if 'timestamp' in interactions.columns:
            interactions = interactions.withColumn('interaction_date', to_date(col('timestamp')))
            interactions = interactions.withColumn('interaction_datetime', col('timestamp'))
        
        # Add customer demographics derived fields
        customers = data_dict['customers']
        customers = customers.withColumn('customer_age_group', 
            when(col('age') < 25, 'Young')
            .when(col('age') < 35, 'Adult')
            .when(col('age') < 50, 'Middle-aged')
            .when(col('age') < 65, 'Senior')
            .otherwise('Elderly'))
        
        # Calculate tenure if join_date exists
        if 'join_date' in customers.columns:
            customers = customers.withColumn('tenure_days', 
                datediff(current_date(), col('join_date')))
        
        return {
            'customers': customers,
            'transactions': transactions,
            'interactions': interactions,
            'churn': data_dict['churn']
        }
    
    def create_customer_360_snapshot(self, data_dict: dict[str, DataFrame], 
                                   snapshot_date: datetime, 
                                   lookback_days: int = 90) -> DataFrame:
        """Create comprehensive customer 360 view for a specific snapshot date"""
        
        snapshot_ts = lit(snapshot_date)
        snapshot_str = snapshot_date.strftime("%Y-%m-%d")
        
        self.logger.info(f"Creating Customer 360 snapshot for {snapshot_str}")
        
        # Start with customer demographics
        customer_360 = data_dict['customers'].withColumn('snapshot_date', snapshot_ts)
        
        # Add transaction features
        customer_360 = self._add_transaction_features(
            customer_360, data_dict['transactions'], snapshot_date, lookback_days
        )
        
        # Add interaction features
        customer_360 = self._add_interaction_features(
            customer_360, data_dict['interactions'], snapshot_date, lookback_days
        )
        
        # Add behavioral features
        customer_360 = self._add_behavioral_features(
            customer_360, data_dict['transactions'], data_dict['interactions'], 
            snapshot_date, lookback_days
        )
        
        # Add business metrics and risk scores
        customer_360 = self._add_business_metrics(customer_360)
        
        return customer_360
    
    def _add_transaction_features(self, customer_360: DataFrame, 
                                transactions: DataFrame, 
                                snapshot_date: datetime, 
                                lookback_days: int) -> DataFrame:
        """Add comprehensive transaction features"""
        
        snapshot_ts = lit(snapshot_date)
        
        # Filter transactions within lookback window
        tx_window = transactions.filter(
            (col("tx_date") <= snapshot_ts) &
            (col("tx_date") > date_sub(snapshot_ts, lookback_days))
        )
        
        # Calculate transaction aggregations for multiple time windows
        time_windows = [7, 30, 90]
        tx_features = customer_360.select("customer_id", "snapshot_date")
        
        for window_days in time_windows:
            window_suffix = f"_{window_days}d"
            
            tx_window_filtered = tx_window.filter(
                col("tx_date") > date_sub(snapshot_ts, window_days)
            )
            
            tx_agg = tx_window_filtered.groupBy("customer_id").agg(
                # Volume metrics
                count("*").alias(f"tx_count{window_suffix}"),
                sum("amount").alias(f"tx_total_amount{window_suffix}"),
                avg("amount").alias(f"avg_tx_amount{window_suffix}"),
                stddev("amount").alias(f"std_tx_amount{window_suffix}"),
                min("amount").alias(f"min_tx_amount{window_suffix}"),
                max("amount").alias(f"max_tx_amount{window_suffix}"),
                
                # Debit/Credit patterns
                sum(when(col("amount") < 0, -col("amount")).otherwise(0)).alias(f"total_debits{window_suffix}"),
                sum(when(col("amount") > 0, col("amount")).otherwise(0)).alias(f"total_credits{window_suffix}"),
                count(when(col("amount") < 0, 1)).alias(f"debit_count{window_suffix}"),
                count(when(col("amount") > 0, 1)).alias(f"credit_count{window_suffix}"),
                
                # Diversity metrics
                count_distinct("product").alias(f"product_diversity{window_suffix}"),
                count_distinct("txn_type").alias(f"tx_type_diversity{window_suffix}"),
                count_distinct("tx_date").alias(f"tx_active_days{window_suffix}"),
                
                # Timing metrics
                max("tx_date").alias(f"last_tx_date{window_suffix}"),
                min("tx_date").alias(f"first_tx_date{window_suffix}")
            )
            
            # Calculate derived features
            tx_agg = tx_agg.withColumn(
                f"tx_frequency{window_suffix}", 
                col(f"tx_count{window_suffix}") / greatest(col(f"tx_active_days{window_suffix}"), lit(1))
            ).withColumn(
                f"credit_debit_ratio{window_suffix}",
                when(col(f"total_debits{window_suffix}") > 0, 
                     col(f"total_credits{window_suffix}") / col(f"total_debits{window_suffix}"))
                .otherwise(lit(None))
            ).withColumn(
                f"days_since_last_tx{window_suffix}",
                datediff(lit(snapshot_date), col(f"last_tx_date{window_suffix}"))
            )
            
            # Join with main features
            tx_features = tx_features.join(tx_agg, "customer_id", "left")
        
        return customer_360.join(tx_features, ["customer_id", "snapshot_date"], "left")
    
    def _add_interaction_features(self, customer_360: DataFrame, 
                                interactions: DataFrame, 
                                snapshot_date: datetime, 
                                lookback_days: int) -> DataFrame:
        """Add comprehensive interaction features"""
        
        snapshot_ts = lit(snapshot_date)
        
        # Filter interactions within lookback window
        int_window = interactions.filter(
            (col("interaction_date") <= snapshot_ts) &
            (col("interaction_date") > date_sub(snapshot_ts, lookback_days))
        )
        
        # Calculate interaction aggregations for multiple time windows
        time_windows = [7, 30, 90]
        int_features = customer_360.select("customer_id", "snapshot_date")
        
        for window_days in time_windows:
            window_suffix = f"_{window_days}d"
            
            int_window_filtered = int_window.filter(
                col("interaction_date") > date_sub(snapshot_ts, window_days)
            )
            
            int_agg = int_window_filtered.groupBy("customer_id").agg(
                # Volume metrics
                count("*").alias(f"interaction_count{window_suffix}"),
                count_distinct("interaction_date").alias(f"interaction_active_days{window_suffix}"),
                count_distinct("interaction_type").alias(f"interaction_type_diversity{window_suffix}"),
                
                # Specific interaction types
                count(when(col("interaction_type") == "login", 1)).alias(f"login_count{window_suffix}"),
                count(when(col("interaction_type") == "support", 1)).alias(f"support_count{window_suffix}"),
                count(when(col("interaction_type") == "mobile", 1)).alias(f"mobile_count{window_suffix}"),
                count(when(col("interaction_type") == "web", 1)).alias(f"web_count{window_suffix}"),
                
                # Timing metrics
                max("interaction_date").alias(f"last_interaction_date{window_suffix}"),
                min("interaction_date").alias(f"first_interaction_date{window_suffix}")
            )
            
            # Calculate derived features
            int_agg = int_agg.withColumn(
                f"interaction_frequency{window_suffix}", 
                col(f"interaction_count{window_suffix}") / greatest(col(f"interaction_active_days{window_suffix}"), lit(1))
            ).withColumn(
                f"days_since_last_interaction{window_suffix}",
                datediff(lit(snapshot_date), col(f"last_interaction_date{window_suffix}"))
            ).withColumn(
                f"digital_engagement_score{window_suffix}",
                (coalesce(col(f"mobile_count{window_suffix}"), lit(0)) + 
                 coalesce(col(f"web_count{window_suffix}"), lit(0)) + 
                 coalesce(col(f"login_count{window_suffix}"), lit(0))) / 3.0
            )
            
            # Join with main features
            int_features = int_features.join(int_agg, "customer_id", "left")
        
        return customer_360.join(int_features, ["customer_id", "snapshot_date"], "left")
    
    def _add_behavioral_features(self, customer_360: DataFrame, 
                               transactions: DataFrame, 
                               interactions: DataFrame,
                               snapshot_date: datetime, 
                               lookback_days: int) -> DataFrame:
        """Add behavioral trend features comparing different time periods"""
        
        snapshot_ts = lit(snapshot_date)
        
        # Compare recent vs older periods
        recent_period = 30  # days
        older_period = 90   # days
        
        # Transaction trends
        tx_recent = transactions.filter(
            (col("tx_date") <= snapshot_ts) &
            (col("tx_date") > date_sub(snapshot_ts, recent_period))
        ).groupBy("customer_id").agg(
            count("*").alias("tx_count_recent"),
            sum("amount").alias("tx_amount_recent"),
            avg("amount").alias("avg_tx_recent")
        )
        
        tx_older = transactions.filter(
            (col("tx_date") <= date_sub(snapshot_ts, recent_period)) &
            (col("tx_date") > date_sub(snapshot_ts, older_period))
        ).groupBy("customer_id").agg(
            count("*").alias("tx_count_older"),
            sum("amount").alias("tx_amount_older"),
            avg("amount").alias("avg_tx_older")
        )
        
        # Interaction trends
        int_recent = interactions.filter(
            (col("interaction_date") <= snapshot_ts) &
            (col("interaction_date") > date_sub(snapshot_ts, recent_period))
        ).groupBy("customer_id").agg(
            count("*").alias("int_count_recent")
        )
        
        int_older = interactions.filter(
            (col("interaction_date") <= date_sub(snapshot_ts, recent_period)) &
            (col("interaction_date") > date_sub(snapshot_ts, older_period))
        ).groupBy("customer_id").agg(
            count("*").alias("int_count_older")
        )
        
        # Combine trends
        behavioral_features = customer_360.select("customer_id", "snapshot_date") \
            .join(tx_recent, "customer_id", "left") \
            .join(tx_older, "customer_id", "left") \
            .join(int_recent, "customer_id", "left") \
            .join(int_older, "customer_id", "left")
        
        # Calculate trend indicators
        behavioral_features = behavioral_features.withColumn(
            "tx_volume_trend",
            when(col("tx_count_older") > 0, 
                 col("tx_count_recent") / col("tx_count_older"))
            .otherwise(lit(None))
        ).withColumn(
            "tx_amount_trend",
            when(col("tx_amount_older") > 0, 
                 col("tx_amount_recent") / col("tx_amount_older"))
            .otherwise(lit(None))
        ).withColumn(
            "interaction_trend",
            when(col("int_count_older") > 0, 
                 col("int_count_recent") / col("int_count_older"))
            .otherwise(lit(None))
        ).withColumn(
            "behavioral_risk_score",
            when(
                (coalesce(col("tx_volume_trend"), lit(1)) < 0.5) |
                (coalesce(col("interaction_trend"), lit(1)) < 0.5),
                1.0
            ).when(
                (coalesce(col("tx_volume_trend"), lit(1)) < 0.8) |
                (coalesce(col("interaction_trend"), lit(1)) < 0.8),
                0.6
            ).otherwise(0.0)
        )
        
        return customer_360.join(behavioral_features, ["customer_id", "snapshot_date"], "left")
    
    def _add_business_metrics(self, customer_360: DataFrame) -> DataFrame:
        """Add comprehensive business metrics and risk scores"""
        
        # Fill nulls with appropriate defaults
        customer_360 = customer_360.fillna({
            'tx_count_30d': 0,
            'tx_count_7d': 0,
            'interaction_count_30d': 0,
            'support_count_90d': 0,
            'days_since_last_tx_30d': 999,
            'days_since_last_interaction_30d': 999
        })
        
        # Enhanced business metrics
        return customer_360.withColumn(
            "dormancy_risk_score",
            when(
                (col("days_since_last_tx_30d") > 30) |
                (col("days_since_last_interaction_30d") > 45) |
                (col("tx_count_30d") == 0),
                1.0
            ).when(
                (col("days_since_last_tx_30d") > 14) |
                (col("days_since_last_interaction_30d") > 21) |
                (col("tx_count_7d") == 0),
                0.7
            ).otherwise(0.0)
        ).withColumn(
            "support_stress_score",
            when(col("support_count_90d") > 10, 1.0)
            .when(col("support_count_90d") > 5, 0.6)
            .when(col("support_count_90d") > 2, 0.3)
            .otherwise(0.0)
        ).withColumn(
            "engagement_quality_score",
            greatest(
                coalesce(col("digital_engagement_score_90d"), lit(0)) * 0.3,
                coalesce(col("interaction_frequency_90d"), lit(0)) * 0.2,
                coalesce(col("tx_frequency_90d"), lit(0)) * 0.3
            )
        ).withColumn(
            "financial_health_score",
            when(col("credit_debit_ratio_90d") > 1.2, 1.0)
            .when(col("credit_debit_ratio_90d") > 0.8, 0.8)
            .when(col("credit_debit_ratio_90d") > 0.5, 0.5)
            .otherwise(0.2)
        ).withColumn(
            "product_adoption_score",
            (coalesce(col("product_diversity_90d"), lit(0)) +
             coalesce(col("tx_type_diversity_90d"), lit(0))) / 2.0
        ).withColumn(
            "churn_risk_score",
            (coalesce(col("dormancy_risk_score"), lit(0)) * 0.3 +
             coalesce(col("support_stress_score"), lit(0)) * 0.25 +
             coalesce(col("behavioral_risk_score"), lit(0)) * 0.25 +
             (1 - coalesce(col("engagement_quality_score"), lit(0))) * 0.2)
        ).withColumn(
            "customer_value_tier",
            when(col("tx_total_amount_90d") > 50000, "Premium")
            .when(col("tx_total_amount_90d") > 20000, "Gold")
            .when(col("tx_total_amount_90d") > 5000, "Silver")
            .otherwise("Bronze")
        ).withColumn(
            "retention_priority",
            when(
                (col("customer_value_tier") == "Premium") & (col("churn_risk_score") > 0.6),
                "Critical"
            ).when(
                (col("customer_value_tier").isin(["Premium", "Gold"])) & (col("churn_risk_score") > 0.4),
                "High"
            ).when(
                col("churn_risk_score") > 0.6,
                "Medium"
            ).otherwise("Low")
        )
    
    def generate_rolling_snapshots(self, data_dict: dict[str, DataFrame],
                                 start_date: datetime, 
                                 end_date: datetime,
                                 frequency_days: int = 7,
                                 lookback_days: int = 90) -> DataFrame:
        """Generate rolling Customer 360 snapshots"""
        
        # Generate snapshot dates
        snapshot_dates = []
        current_date = start_date
        while current_date <= end_date:
            snapshot_dates.append(current_date)
            current_date += timedelta(days=frequency_days)
        
        self.logger.info(f"Generating {len(snapshot_dates)} snapshots from {start_date} to {end_date}")
        
        # Generate snapshots for each date
        all_snapshots = []
        for snapshot_date in snapshot_dates:
            snapshot = self.create_customer_360_snapshot(
                data_dict, snapshot_date, lookback_days
            )
            all_snapshots.append(snapshot)
        
        # Union all snapshots
        customer_360_rolling = reduce(DataFrame.unionByName, all_snapshots)
        
        return customer_360_rolling
    
    def create_ml_feature_store(self, customer_360: DataFrame, 
                              transactions: DataFrame,
                              prediction_horizon_days: int = 30,
                              model_name: str = "churn_prediction",
                              model_version: str = "v1.0") -> DataFrame:
        """Create ML-ready feature store with churn labels"""
        
        self.logger.info(f"Creating ML feature store for {model_name} {model_version}")
        
        # For each customer snapshot, determine if they churned in the prediction horizon
        ml_features = customer_360.alias("c360")
        
        # Find customers who had transactions after each snapshot date
        future_transactions = transactions.select("customer_id", "tx_date").alias("future_tx")
        
        # Join to find future activity
        ml_features = ml_features.join(
            future_transactions,
            (col("c360.customer_id") == col("future_tx.customer_id")) &
            (col("future_tx.tx_date") > col("c360.snapshot_date")) &
            (col("future_tx.tx_date") <= date_add(col("c360.snapshot_date"), prediction_horizon_days)),
            "left"
        )
        
        # Label churn: 1 if no future transactions, 0 if there are future transactions
        ml_features = ml_features.withColumn(
            "has_future_activity",
            when(col("future_tx.tx_date").isNotNull(), 1).otherwise(0)
        ).groupBy(
            *[col(f"c360.{c}") for c in customer_360.columns]
        ).agg(
            max("has_future_activity").alias("has_future_activity")
        ).withColumn(
            "churn_label",
            when(col("has_future_activity") == 0, 1).otherwise(0)
        )
        
        # Add comprehensive metadata
        ml_features = ml_features.withColumn("model_name", lit(model_name)) \
            .withColumn("model_version", lit(model_version)) \
            .withColumn("prediction_horizon_days", lit(prediction_horizon_days)) \
            .withColumn("feature_creation_timestamp", current_timestamp()) \
            .withColumn("data_quality_score", lit(1.0))  # Can be enhanced
        
        return ml_features.drop("has_future_activity")
    
    def get_feature_columns_for_ml(self, ml_feature_store: DataFrame) -> dict:
        """Extract feature columns suitable for ML training with categorization"""
        
        excluded_columns = {
            'customer_id', 'snapshot_date', 'churn_label', 'model_name', 
            'model_version', 'prediction_horizon_days', 'feature_creation_timestamp',
            'data_quality_score', 'join_date', 'last_tx_date_7d', 'last_tx_date_30d',
            'last_tx_date_90d', 'first_tx_date_7d', 'first_tx_date_30d', 
            'first_tx_date_90d', 'last_interaction_date_7d', 'last_interaction_date_30d',
            'last_interaction_date_90d', 'first_interaction_date_7d', 
            'first_interaction_date_30d', 'first_interaction_date_90d'
        }
        
        all_columns = ml_feature_store.columns
        feature_columns = [col for col in all_columns if col not in excluded_columns]
        
        # Categorize features for better model interpretability
        feature_categories = {
            'demographic': [col for col in feature_columns if any(x in col.lower() for x in ['age', 'gender', 'state', 'tenure'])],
            'product': [col for col in feature_columns if any(x in col.lower() for x in ['has_', 'product', 'diversity'])],
            'transaction': [col for col in feature_columns if any(x in col.lower() for x in ['tx_', 'amount', 'debit', 'credit'])],
            'interaction': [col for col in feature_columns if any(x in col.lower() for x in ['interaction', 'login', 'support', 'mobile', 'web'])],
            'behavioral': [col for col in feature_columns if any(x in col.lower() for x in ['trend', 'frequency', 'days_since'])],
            'risk_scores': [col for col in feature_columns if 'score' in col.lower()]
        }
        
        return {
            'all_features': feature_columns,
            'feature_categories': feature_categories,
            'total_features': len(feature_columns)
        }


def run_unified_pipeline(spark: SparkSession,
                        data_path: str = None,
                        start_date: str = "2023-01-01",
                        end_date: str = "2023-12-31",
                        frequency_days: int = 7,
                        lookback_days: int = 90,
                        prediction_horizon_days: int = 30,
                        output_path: str = str(PROCESSED_PATH),
                        **dataframes):
    """
    Unified pipeline for creating Customer 360 and ML feature store
    """
    
    print("=== Starting Feature Engineering Pipeline ===")
    
    # Initialize feature engineering
    fe = CreditUnionFeatureEngineering(spark)
    
    # Load data
    if data_path:
        data_dict = fe.load_data(data_path=data_path)
    else:
        data_dict = fe.load_data(**dataframes)
    
    # Convert date strings to datetime objects
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Create rolling Customer 360 snapshots
    print("Creating rolling Customer 360 snapshots...")
    customer_360_rolling = fe.generate_rolling_snapshots(
        data_dict, start_dt, end_dt, frequency_days, lookback_days
    )
    
    # Create ML feature store
    print("Creating ML feature store...")
    ml_feature_store = fe.create_ml_feature_store(
        customer_360_rolling, 
        data_dict['transactions'],
        prediction_horizon_days
    )
    
    # Get feature information
    feature_info = fe.get_feature_columns_for_ml(ml_feature_store)
    
    # Save outputs with optimized partitioning
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)
    
    print("Saving Customer 360 snapshots...")
    customer_360_rolling.write \
        .mode("overwrite") \
        .partitionBy("snapshot_date") \
        .parquet(str(output_path / "customer_360_snapshots"))
    
    print("Saving ML feature store...")
    ml_feature_store.write \
        .mode("overwrite") \
        .partitionBy("snapshot_date") \
        .parquet(str(output_path / "ml_feature_store"))
    
    # Calculate statistics
    customer_360_count = customer_360_rolling.count()
    ml_feature_count = ml_feature_store.count()
    churn_rate = ml_feature_store.agg(avg("churn_label")).collect()[0][0] * 100
    high_risk_customers = ml_feature_store.filter(col("churn_risk_score") > 0.7).count()
    avg_engagement = ml_feature_store.agg(avg("engagement_quality_score")).collect()[0][0]
    
    print(f"\n=== Pipeline Results ===")
    print(f"Customer 360 snapshots: {customer_360_count:,} records")
    print(f"ML feature store: {ml_feature_count:,} records")
    print(f"Churn rate in ML data: {churn_rate:.2f}%")
    print(f"High-risk customers: {high_risk_customers:,}")
    print(f"Average engagement score: {avg_engagement:.3f}")
    print(f"Total features for ML: {feature_info['total_features']}")
    
    # Print feature categories
    print("\n=== Feature Categories ===")
    for category, features in feature_info['feature_categories'].items():
        print(f"{category.capitalize()}: {len(features)} features")
    
    return {
        'customer_360_rolling': customer_360_rolling,
        'ml_feature_store': ml_feature_store,
        'feature_info': feature_info,
        'statistics': {
            'customer_360_count': customer_360_count,
            'ml_feature_count': ml_feature_count,
            'churn_rate': churn_rate,
            'high_risk_customers': high_risk_customers,
            'avg_engagement': avg_engagement
        }
    }


# # Example usage with monitoring
# if __name__ == "__main__":
#     spark = get_spark_session()
    
#     try:
#         # Run the pipeline
#         results = run_unified_pipeline(
#             spark=spark,
#             data_path=str(RAW_DATA_PATH)
#         )
        
#         print(" Pipeline completed successfully!")
#         print(f"Generated {results['feature_count']} features for ML")
#         print(f"Feature columns: {results['feature_columns'][:10]}...")  # Show first 10
        
#     except Exception as e:
#         print(f"Pipeline failed: {e}")
#         raise
#     finally:
#         spark.stop()                   

spark = get_spark_session()
    
try:
    # Run the pipeline
    results = run_unified_pipeline(
        spark=spark,
        data_path=str(RAW_DATA_PATH)
    )
    
    print("Pipeline completed successfully!")
    print(f"Generated {results['feature_count']} features for ML")
    print(f"Feature columns: {results['feature_columns'][:10]}...")  # Show first 10
    
except Exception as e:
    print(f"Pipeline failed: {e}")
    raise
finally:
    spark.stop()