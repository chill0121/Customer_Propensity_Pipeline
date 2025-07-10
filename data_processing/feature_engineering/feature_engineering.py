from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
from pyspark import StorageLevel 
from datetime import datetime, date, timedelta
from pathlib import Path
import logging
from functools import reduce
import builtins

# Initialize Spark session
def get_spark_session():
    return SparkSession.builder \
        .appName("CreditUnionFeatureEngineering") \
        .config("spark.driver.memory", "12g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.adaptive.skewJoin.enabled", "true") \
        .config("spark.sql.adaptive.localShuffleReader.enabled", "true") \
        .getOrCreate()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw"
PROCESSED_PATH = BASE_DIR / "data" / "processed"
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2024, 12, 31)

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
                }
                self.logger.info("Successfully loaded all parquet files")
            except Exception as e:
                self.logger.error(f"Failed to load parquet files: {e}")
                raise
        
        elif all(df is not None for df in [dataframes.get('customer_df'), 
                                          dataframes.get('transaction_df'),
                                          dataframes.get('interaction_df')]):
            data_dict = {
                'customers': dataframes['customer_df'],
                'transactions': dataframes['transaction_df'],
                'interactions': dataframes['interaction_df'],
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
        
        return {
            'customers': customers,
            'transactions': transactions,
            'interactions': interactions,
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

        # Calculate customer tenure
        if 'join_date' in customer_360.columns:
            customer_360 = customer_360.withColumn('tenure_days',
                datediff(col('snapshot_date'), col('join_date')))
        
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
        """Add comprehensive transaction features using a single aggregation pass."""

        snapshot_ts = lit(snapshot_date)

        # Filter transactions within the largest lookback window (90 days)
        tx_window = transactions.filter(
            (col("tx_date") <= snapshot_ts) &
            (col("tx_date") > date_sub(snapshot_ts, lookback_days))
        )

        # Use conditional aggregation to calculate metrics for all time windows at once
        tx_agg = tx_window.groupBy("customer_id").agg(
            # --- 90-day aggregations (on the full filtered window) ---
            count("*").alias("tx_count_90d"),
            sum("amount").alias("tx_total_amount_90d"),
            avg("amount").alias("avg_tx_amount_90d"),
            stddev("amount").alias("std_tx_amount_90d"),
            min("amount").alias("min_tx_amount_90d"),
            max("amount").alias("max_tx_amount_90d"),
            sum(when(col("amount") < 0, -col("amount")).otherwise(0)).alias("total_debits_90d"),
            sum(when(col("amount") > 0, col("amount")).otherwise(0)).alias("total_credits_90d"),
            count(when(col("amount") < 0, 1)).alias("debit_count_90d"),
            count(when(col("amount") > 0, 1)).alias("credit_count_90d"),
            count_distinct("product").alias("product_diversity_90d"),
            count_distinct("txn_type").alias("tx_type_diversity_90d"),
            count_distinct("tx_date").alias("tx_active_days_90d"),
            max("tx_date").alias("last_tx_date_90d"),

            # --- 30-day conditional aggregations ---
            count(when(col("tx_date") > date_sub(snapshot_ts, 30), 1)).alias("tx_count_30d"),
            sum(when(col("tx_date") > date_sub(snapshot_ts, 30), col("amount"))).alias("tx_total_amount_30d"),
            avg(when(col("tx_date") > date_sub(snapshot_ts, 30), col("amount"))).alias("avg_tx_amount_30d"),
            stddev(when(col("tx_date") > date_sub(snapshot_ts, 30), col("amount"))).alias("std_tx_amount_30d"),
            count_distinct(when(col("tx_date") > date_sub(snapshot_ts, 30), col("tx_date"))).alias("tx_active_days_30d"),
            max(when(col("tx_date") > date_sub(snapshot_ts, 30), col("tx_date"))).alias("last_tx_date_30d"),

            # --- 7-day conditional aggregations ---
            count(when(col("tx_date") > date_sub(snapshot_ts, 7), 1)).alias("tx_count_7d"),
            sum(when(col("tx_date") > date_sub(snapshot_ts, 7), col("amount"))).alias("tx_total_amount_7d"),
            avg(when(col("tx_date") > date_sub(snapshot_ts, 7), col("amount"))).alias("avg_tx_amount_7d"),
            count_distinct(when(col("tx_date") > date_sub(snapshot_ts, 7), col("tx_date"))).alias("tx_active_days_7d"),
            max(when(col("tx_date") > date_sub(snapshot_ts, 7), col("tx_date"))).alias("last_tx_date_7d")
        )

        # Calculate derived features for each time window
        tx_agg = tx_agg.withColumn(
            "tx_frequency_90d", col("tx_count_90d") / greatest(col("tx_active_days_90d"), lit(1))
        ).withColumn(
            "credit_debit_ratio_90d",
            when(col("total_debits_90d") > 0, col("total_credits_90d") / col("total_debits_90d")).otherwise(lit(None))
        ).withColumn(
            "days_since_last_tx_90d", datediff(snapshot_ts, col("last_tx_date_90d"))
        ).withColumn(
            "tx_frequency_30d", col("tx_count_30d") / greatest(col("tx_active_days_30d"), lit(1))
        ).withColumn(
            "days_since_last_tx_30d", datediff(snapshot_ts, col("last_tx_date_30d"))
        ).withColumn(
            "tx_frequency_7d", col("tx_count_7d") / greatest(col("tx_active_days_7d"), lit(1))
        ).withColumn(
            "days_since_last_tx_7d", datediff(snapshot_ts, col("last_tx_date_7d"))
        )

        return customer_360.join(tx_agg, "customer_id", "left")

    def _add_interaction_features(self, customer_360: DataFrame,
                                    interactions: DataFrame,
                                    snapshot_date: datetime,
                                    lookback_days: int) -> DataFrame:
        """Add comprehensive interaction features using a single aggregation pass."""

        snapshot_ts = lit(snapshot_date)

        # Filter interactions within the largest lookback window (90 days)
        int_window = interactions.filter(
            (col("interaction_date") <= snapshot_ts) &
            (col("interaction_date") > date_sub(snapshot_ts, lookback_days))
        )

        # Use conditional aggregation to calculate metrics for all time windows at once
        int_agg = int_window.groupBy("customer_id").agg(
            # --- 90-day aggregations (on the full filtered window) ---
            count("*").alias("interaction_count_90d"),
            count_distinct("interaction_date").alias("interaction_active_days_90d"),
            count_distinct("interaction_type").alias("interaction_type_diversity_90d"),
            count(when(col("interaction_type") == "login", 1)).alias("login_count_90d"),
            count(when(col("interaction_type") == "support", 1)).alias("support_count_90d"),
            count(when(col("interaction_type") == "mobile", 1)).alias("mobile_count_90d"),
            count(when(col("interaction_type") == "web", 1)).alias("web_count_90d"),
            max("interaction_date").alias("last_interaction_date_90d"),

            # --- 30-day conditional aggregations ---
            count(when(col("interaction_date") > date_sub(snapshot_ts, 30), 1)).alias("interaction_count_30d"),
            count_distinct(when(col("interaction_date") > date_sub(snapshot_ts, 30), col("interaction_date"))).alias("interaction_active_days_30d"), # CORRECTED
            max(when(col("interaction_date") > date_sub(snapshot_ts, 30), col("interaction_date"))).alias("last_interaction_date_30d"), # CORRECTED

            # --- 7-day conditional aggregations ---
            count(when(col("interaction_date") > date_sub(snapshot_ts, 7), 1)).alias("interaction_count_7d"),
            count_distinct(when(col("interaction_date") > date_sub(snapshot_ts, 7), col("interaction_date"))).alias("interaction_active_days_7d"), # CORRECTED
            max(when(col("interaction_date") > date_sub(snapshot_ts, 7), col("interaction_date"))).alias("last_interaction_date_7d") # CORRECTED
        )

        # Calculate derived features for each time window
        int_agg = int_agg.withColumn(
            "interaction_frequency_90d", col("interaction_count_90d") / greatest(col("interaction_active_days_90d"), lit(1))
        ).withColumn(
            "days_since_last_interaction_90d", datediff(snapshot_ts, col("last_interaction_date_90d"))
        ).withColumn(
            "digital_engagement_score_90d",
            (coalesce(col("mobile_count_90d"), lit(0)) +
            coalesce(col("web_count_90d"), lit(0)) +
            coalesce(col("login_count_90d"), lit(0))) / 3.0
        ).withColumn(
            "interaction_frequency_30d", col("interaction_count_30d") / greatest(col("interaction_active_days_30d"), lit(1))
        ).withColumn(
            "days_since_last_interaction_30d", datediff(snapshot_ts, col("last_interaction_date_30d"))
        ).withColumn(
            "interaction_frequency_7d", col("interaction_count_7d") / greatest(col("interaction_active_days_7d"), lit(1))
        ).withColumn(
            "days_since_last_interaction_7d", datediff(snapshot_ts, col("last_interaction_date_7d"))
        )

        return customer_360.join(int_agg, "customer_id", "left")
    
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
    
    
    def _calculate_churn_label(self, customer_360: DataFrame,
                                transactions: DataFrame,
                                interactions: DataFrame,
                                prediction_horizon_days: int) -> DataFrame:
        """
        FIXED churn label calculation that properly handles temporal consistency:
        1. Only includes customers who joined BEFORE snapshot date
        2. Calculates last_activity_date AS OF each snapshot date
        3. Implements proper eligibility criteria
        4. Handles new customers with appropriate grace periods
        """
        
        # Step 1: Create unified activity timeline
        tx_activity = transactions.select(
            col("customer_id"),
            col("tx_date").alias("activity_date"),
            lit("transaction").alias("activity_type")
        )
        
        int_activity = interactions.select(
            col("customer_id"), 
            col("interaction_date").alias("activity_date"),
            lit("interaction").alias("activity_type")
        )
        
        all_activities = tx_activity.unionByName(int_activity)
        
        # Step 2: FIXED - Only include customers who joined BEFORE snapshot date
        # This eliminates impossible scenarios where snapshots exist before customer joined
        eligible_customers = customer_360.filter(
            col("join_date") <= col("snapshot_date")
        ).filter(
            # Give customers at least 7 days to establish activity patterns
            datediff(col("snapshot_date"), col("join_date")) >= 7
        )
        
        # Step 3: FIXED - Calculate last activity AS OF each snapshot date
        # This ensures we only consider historical activity, not future activity
        snapshot_activities = all_activities.join(
            eligible_customers.select("customer_id", "snapshot_date"),
            "customer_id"
        ).filter(
            col("activity_date") <= col("snapshot_date")  # Only past/current activity
        )
        
        # Get last activity date as of each snapshot
        customer_last_activity_snapshot = snapshot_activities.groupBy(
            "customer_id", "snapshot_date"
        ).agg(
            max("activity_date").alias("last_activity_date_as_of_snapshot")
        )
        
        # Step 4: Join with eligible customers
        customer_360_with_activity = eligible_customers.join(
            customer_last_activity_snapshot,
            ["customer_id", "snapshot_date"],
            "left"
        )
        
        # Step 5: FIXED - Apply activity-based filtering properly
        # Only include customers who have been reasonably active or are genuinely new
        active_customers = customer_360_with_activity.filter(
            col("last_activity_date_as_of_snapshot").isNull() |
            (datediff(col("snapshot_date"), col("last_activity_date_as_of_snapshot")) <= 90)
        ).filter(
            (datediff(col("snapshot_date"), col("join_date")) <= 30) |
            (
                col("last_activity_date_as_of_snapshot").isNotNull() & 
                (datediff(col("snapshot_date"), col("last_activity_date_as_of_snapshot")) <= 120)
            )
        )
        
        # Step 6: FIXED - Calculate future activity properly
        # Look for activity AFTER snapshot date to determine churn
        customer_next_activity = all_activities.join(
            active_customers.select("customer_id", "snapshot_date"),
            "customer_id"
        ).filter(
            col("activity_date") > col("snapshot_date")  # Future activity
        ).groupBy("customer_id", "snapshot_date").agg(
            min("activity_date").alias("next_activity_date")
        )
        
        # Step 7: Join and calculate churn metrics
        ml_features = active_customers.join(
            customer_next_activity,
            ["customer_id", "snapshot_date"],
            "left"
        ).withColumn(
            "days_to_next_activity",
            when(col("next_activity_date").isNotNull(),
                datediff(col("next_activity_date"), col("snapshot_date")))
            .otherwise(lit(None))
        ).withColumn(
            "prediction_end_date", 
            date_add(col("snapshot_date"), prediction_horizon_days)
        ).withColumn(
            "churn_label",
            when(
                # Customer churns if no future activity within prediction horizon
                col("next_activity_date").isNull() |
                (col("days_to_next_activity") > prediction_horizon_days),
                1
            ).otherwise(0)
        )
        
        # Step 8: FIXED - Add proper business logic
        ml_features = ml_features.withColumn(
            "days_since_join_at_snapshot",
            datediff(col("snapshot_date"), col("join_date"))
        ).withColumn(
            "is_new_customer",
            when(col("days_since_join_at_snapshot") <= 30, 1).otherwise(0)
        ).withColumn(
            "days_since_last_activity",
            when(col("last_activity_date_as_of_snapshot").isNotNull(),
                datediff(col("snapshot_date"), col("last_activity_date_as_of_snapshot")))
            .otherwise(col("days_since_join_at_snapshot"))  # Use days since join if no activity
        ).withColumn(
            "has_any_historical_activity",
            when(col("last_activity_date_as_of_snapshot").isNotNull(), 1).otherwise(0)
        ).withColumn(
            "churn_label_adjusted",
            when(
                # More nuanced new customer handling
                (col("is_new_customer") == 1) & 
                (col("has_any_historical_activity") == 0) &
                (col("days_since_join_at_snapshot") <= 14),  # Very new customers
                0  # Don't label as churn yet
            ).when(
                # Customers with some activity but very new
                (col("is_new_customer") == 1) & 
                (col("has_any_historical_activity") == 1) &
                (col("days_since_last_activity") <= 45),  # Recent activity
                0  # Don't label as churn yet
            ).otherwise(col("churn_label"))
        )
        
        return ml_features
    
    def create_ml_feature_store_comprehensive(self, customer_360: DataFrame, 
                                         transactions: DataFrame,
                                         interactions: DataFrame,
                                         prediction_horizon_days: int = 30,
                                         model_name: str = "churn_prediction",
                                         model_version: str = "v1.0") -> DataFrame:
        """Most comprehensive approach considering both transactions and interactions for churn labeling."""
        
        self.logger.info(f"Creating ML feature store for {model_name} {model_version} (comprehensive churn)")
        
        # Use the unified churn logic
        ml_features = self._calculate_churn_label(
            customer_360, transactions, interactions, prediction_horizon_days
        )
        
        # Add metadata
        ml_features = ml_features.withColumn("model_name", lit(model_name)) \
            .withColumn("model_version", lit(model_version)) \
            .withColumn("prediction_horizon_days", lit(prediction_horizon_days)) \
            .withColumn("feature_creation_timestamp", current_timestamp()) \
            .withColumn("data_quality_score", lit(1.0))
        
        return ml_features
    
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

    def validate_pipeline_results(self, customer_360: DataFrame, ml_feature_store: DataFrame,
                            expected_customers: int, expected_weeks: int) -> dict:
        """Validate pipeline results for data quality"""
        
        validation_results = {}
        
        try:
            # Check record counts
            c360_count = customer_360.count()
            ml_count = ml_feature_store.count()
            expected_records = expected_customers * expected_weeks
            
            validation_results['record_count_check'] = {
                'customer_360_count': c360_count,
                'ml_feature_store_count': ml_count,
                'expected_records': expected_records,
                'count_reasonable': builtins.abs(c360_count - expected_records) < (expected_records * 0.1)
            }
            
            # Check churn rate with safe handling
            try:
                churn_rate_result = ml_feature_store.agg(avg("churn_label")).collect()
                if churn_rate_result and churn_rate_result[0] and churn_rate_result[0][0] is not None:
                    churn_rate = churn_rate_result[0][0]
                else:
                    churn_rate = 0.0
                    self.logger.warning("Could not calculate churn rate - using 0.0 as default")
            except Exception as e:
                churn_rate = 0.0
                self.logger.warning(f"Error calculating churn rate: {e} - using 0.0 as default")
            
            validation_results['churn_rate_check'] = {
                'churn_rate': churn_rate,
                'reasonable_range': 0.02 <= churn_rate <= 0.20,  # 2-20% is typical
                'warning': churn_rate > 0.30 or churn_rate < 0.01
            }
            
            # Check for data leakage
            future_features = [col for col in ml_feature_store.columns 
                            if 'future' in col.lower() or 'after' in col.lower()]
            validation_results['data_leakage_check'] = {
                'suspicious_features': future_features,
                'clean': len(future_features) == 0
            }
            
            # Check snapshot date distribution
            try:
                snapshot_dates = ml_feature_store.select("snapshot_date").distinct().count()
            except Exception as e:
                snapshot_dates = 0
                self.logger.warning(f"Error counting snapshot dates: {e}")
            
            validation_results['temporal_check'] = {
                'unique_snapshot_dates': snapshot_dates,
                'expected_dates': expected_weeks,
                'reasonable': builtins.abs(snapshot_dates - expected_weeks) <= 2
            }
            
            # Additional data quality checks
            try:
                # Check for null values in key columns
                key_columns = ['customer_id', 'snapshot_date', 'churn_label']
                null_counts = {}
                for col_name in key_columns:
                    if col_name in ml_feature_store.columns:
                        null_count = ml_feature_store.filter(col(col_name).isNull()).count()
                        null_counts[col_name] = null_count
                
                validation_results['data_quality_check'] = {
                    'null_counts': null_counts,
                    'has_nulls_in_key_columns': any(count > 0 for count in null_counts.values())
                }
            except Exception as e:
                self.logger.warning(f"Error in data quality check: {e}")
                validation_results['data_quality_check'] = {
                    'null_counts': {},
                    'has_nulls_in_key_columns': False,
                    'error': str(e)
                }
            
            # Check feature completeness
            try:
                feature_info = self.get_feature_columns_for_ml(ml_feature_store)
                validation_results['feature_completeness_check'] = {
                    'total_features': feature_info['total_features'],
                    'feature_categories': {k: len(v) for k, v in feature_info['feature_categories'].items()},
                    'has_minimum_features': feature_info['total_features'] >= 10
                }
            except Exception as e:
                self.logger.warning(f"Error in feature completeness check: {e}")
                validation_results['feature_completeness_check'] = {
                    'total_features': 0,
                    'feature_categories': {},
                    'has_minimum_features': False,
                    'error': str(e)
                }
            
        except Exception as e:
            self.logger.error(f"Critical error in validation: {e}")
            validation_results['validation_error'] = {
                'error': str(e),
                'status': 'failed'
            }
        
        return validation_results

def run_unified_pipeline_enhanced(spark: SparkSession,
                        data_path: str = None,
                        start_date: str = START_DATE,
                        end_date: str = END_DATE - timedelta(days=30),
                        frequency_days: int = 7,
                        lookback_days: int = 90,
                        prediction_horizon_days: int = 30,
                        output_path: str = str(PROCESSED_PATH),
                        validate_results: bool = True,
                        **dataframes):
    """
    Enhanced pipeline with proper date handling to avoid early-year bias
    """
    
    print("=== Starting Enhanced Feature Engineering Pipeline ===")
    
    # Initialize feature engineering
    fe = CreditUnionFeatureEngineering(spark)
    
    # Load data
    if data_path:
        data_dict = fe.load_data(data_path=data_path)
    else:
        data_dict = fe.load_data(**dataframes)
    
    # Parse strings or ensure all dates are datetime.datetime objects
    start_dt = datetime.combine(start_date, datetime.min.time()) if isinstance(start_date, date) and not isinstance(start_date, datetime) else start_date
    end_dt = datetime.combine(end_date, datetime.min.time()) if isinstance(end_date, date) and not isinstance(end_date, datetime) else end_date
    
    # CRITICAL FIX: Ensure we have enough future data for all snapshots
    # Check the actual data range in your datasets
    max_tx_date = data_dict['transactions'].agg(max("tx_date")).collect()[0][0]
    max_int_date = data_dict['interactions'].agg(max("interaction_date")).collect()[0][0]
    
    # Use the earlier of the two max dates
    actual_data_end = builtins.min(max_tx_date, max_int_date) if max_tx_date and max_int_date else (max_tx_date or max_int_date)
    
    if actual_data_end:
        # Ensure we don't create snapshots too close to the end of available data
        safe_end_date = actual_data_end - timedelta(days=prediction_horizon_days + 7)  # Extra buffer
        safe_end_date = datetime.combine(safe_end_date, datetime.min.time()) if isinstance(safe_end_date, date) and not isinstance(safe_end_date, datetime) else safe_end_date
        end_dt = builtins.min(end_dt, safe_end_date)
        
        print(f"Adjusted end date to {end_dt} to ensure {prediction_horizon_days} days of future data")
    
    # Calculate expected metrics
    expected_weeks = ((end_dt - start_dt).days // frequency_days) + 1
    expected_customers = data_dict['customers'].count()
    
    print(f"Expected: {expected_customers:,} customers × {expected_weeks} weeks = {expected_customers * expected_weeks:,} records")
    
    # Create rolling Customer 360 snapshots
    print("Creating rolling Customer 360 snapshots...")
    customer_360_rolling = fe.generate_rolling_snapshots(
        data_dict, start_dt, end_dt, frequency_days, lookback_days
    )
    
    # Create ML feature store with improved churn labeling
    print("Creating ML feature store with improved churn labeling...")
    ml_feature_store = fe.create_ml_feature_store_comprehensive(
        customer_360_rolling, 
        data_dict['transactions'],
        data_dict['interactions'],
        prediction_horizon_days
    )
    
    # DEBUGGING: Add churn rate analysis by time period
    print("\n=== Churn Rate Analysis by Time Period ===")
    churn_by_month = ml_feature_store.withColumn(
        "snapshot_month", date_format(col("snapshot_date"), "yyyy-MM")
    ).groupBy("snapshot_month").agg(
        count("*").alias("total_records"),
        sum("churn_label").alias("churned_customers"),
        avg("churn_label").alias("churn_rate")
    ).orderBy("snapshot_month")
    
    churn_by_month.show(20, False)
    
    # Additional debugging: Check data availability
    print("\n=== Data Availability Check ===")
    data_range = ml_feature_store.agg(
        min("snapshot_date").alias("min_snapshot"),
        max("snapshot_date").alias("max_snapshot"),
        min("prediction_end_date").alias("min_pred_end"),
        max("prediction_end_date").alias("max_pred_end")
    )
    data_range.show(1, False)
    
    # Validate results
    if validate_results:
        validation_results = fe.validate_pipeline_results(
            customer_360_rolling, ml_feature_store, expected_customers, expected_weeks
        )
        print("\n=== Validation Results ===")
        for check_name, results in validation_results.items():
            print(f"{check_name}: {results}")
    
    # Get feature information
    feature_info = fe.get_feature_columns_for_ml(ml_feature_store)
    
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
    
    # Save outputs if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)
        
        print("Saving outputs...")
        customer_360_rolling.write \
            .mode("overwrite") \
            .partitionBy("snapshot_date") \
            .parquet(str(output_path / "customer_360_snapshots"))
        
        ml_feature_store.write \
            .mode("overwrite") \
            .partitionBy("snapshot_date") \
            .parquet(str(output_path / "ml_feature_store"))
    
    return {
        'customer_360_rolling': customer_360_rolling,
        'ml_feature_store': ml_feature_store,
        'feature_info': feature_info,
        'validation_results': validation_results if validate_results else None,
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
#         results = run_unified_pipeline_enhanced(
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
    results = run_unified_pipeline_enhanced(
        spark=spark,
        data_path=str(RAW_DATA_PATH)
    )

    print("\nPipeline execution complete.\n")
    
except Exception as e:
    print(f"Pipeline failed: {e}")
    raise
finally:
    spark.stop()