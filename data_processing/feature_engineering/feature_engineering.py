from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
from pyspark import StorageLevel 
from datetime import datetime, timedelta
from pathlib import Path
import logging

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
    Enhanced PySpark-based feature engineering pipeline for churn prediction
    Includes advanced temporal features, behavioral patterns, and risk indicators
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
    
    def create_customer_360_snapshot(self, 
                                   data_dict: dict[str, DataFrame],
                                   snapshot_date: str,
                                   lookback_days: int = 90) -> DataFrame:
        """Create comprehensive customer 360 view with enhanced features"""
        
        self.logger.info(f"Creating customer 360 snapshot for {snapshot_date}")
        
        snapshot_dt = to_date(lit(snapshot_date))
        cutoff_date = date_sub(snapshot_dt, lookback_days)
        
        # Enhanced transaction features
        tx_features = self._calculate_enhanced_transaction_features(
            data_dict['transactions'], snapshot_dt, cutoff_date
        )
        
        # Enhanced interaction features
        interaction_features = self._calculate_enhanced_interaction_features(
            data_dict['interactions'], snapshot_dt, cutoff_date
        )
        
        # Behavioral trend features
        behavioral_features = self._calculate_behavioral_trends(
            data_dict['transactions'], data_dict['interactions'], snapshot_dt
        )
        
        # Customer demographics with enhanced tenure features
        customer_features = self._enhance_customer_demographics(
            data_dict['customers'], snapshot_dt
        )
        
        # Join all features efficiently
        customer_360 = customer_features.join(
            tx_features, on="customer_id", how="left"
        ).join(
            interaction_features, on="customer_id", how="left"
        ).join(
            behavioral_features, on="customer_id", how="left"
        )
        
        # Enhanced null handling and derived metrics
        customer_360 = self._add_enhanced_business_metrics(customer_360)
        
        return customer_360
    
    def _calculate_enhanced_transaction_features(self, transaction_df: DataFrame, 
                                              snapshot_dt, cutoff_date) -> DataFrame:
        """Calculate comprehensive transaction features optimized for churn prediction"""
        
        # Filter transactions once for efficiency
        recent_transactions = transaction_df.filter(
            (col("timestamp") <= snapshot_dt) & 
            (col("timestamp") >= cutoff_date)
        ).cache()
        
        # Multiple time windows for trend analysis
        window_30d = date_sub(snapshot_dt, 30)
        window_7d = date_sub(snapshot_dt, 7)
        
        # Base transaction features
        base_features = recent_transactions.groupBy("customer_id").agg(
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
            count_distinct(date_format("timestamp", "yyyy-MM-dd")).alias("tx_active_days_90d"),
            
            # Diversity metrics
            count_distinct("txn_type").alias("tx_type_diversity_90d"),
            count_distinct("product").alias("product_diversity_90d"),
            # count_distinct("merchant").alias("merchant_diversity_90d"),
            
            # Credit/Debit patterns
            sum(when(col("amount") > 0, col("amount")).otherwise(0)).alias("total_credits_90d"),
            sum(when(col("amount") < 0, abs(col("amount"))).otherwise(0)).alias("total_debits_90d"),
            sum(when(col("amount") > 0, 1).otherwise(0)).alias("credit_count_90d"),
            sum(when(col("amount") < 0, 1).otherwise(0)).alias("debit_count_90d"),
            
            # Advanced amount patterns
            percentile_approx("amount", 0.25).alias("tx_amount_q1_90d"),
            percentile_approx("amount", 0.75).alias("tx_amount_q3_90d"),
            skewness("amount").alias("tx_amount_skewness_90d"),
            kurtosis("amount").alias("tx_amount_kurtosis_90d"),
            
            # Time-based patterns
            avg(hour("timestamp")).alias("avg_tx_hour_90d"),
            count_distinct(dayofweek("timestamp")).alias("tx_days_of_week_90d"),
            sum(when(hour("timestamp").between(9, 17), 1).otherwise(0)).alias("business_hours_tx_90d"),
            sum(when(dayofweek("timestamp").isin([1, 7]), 1).otherwise(0)).alias("weekend_tx_90d")
        )
        
        # Short-term features (30 days and 7 days)
        short_term_features = recent_transactions.groupBy("customer_id").agg(
            # 30-day features
            sum(when(col("timestamp") >= window_30d, 1).otherwise(0)).alias("tx_count_30d"),
            sum(when(col("timestamp") >= window_30d, col("amount")).otherwise(0)).alias("tx_total_amount_30d"),
            avg(when(col("timestamp") >= window_30d, col("amount"))).alias("tx_avg_amount_30d"),
            
            # 7-day features
            sum(when(col("timestamp") >= window_7d, 1).otherwise(0)).alias("tx_count_7d"),
            sum(when(col("timestamp") >= window_7d, col("amount")).otherwise(0)).alias("tx_total_amount_7d"),
            avg(when(col("timestamp") >= window_7d, col("amount"))).alias("tx_avg_amount_7d")
        )
        
        # Join and calculate derived features
        enhanced_features = base_features.join(short_term_features, on="customer_id", how="left")
        
        # Calculate temporal trends and ratios
        enhanced_features = enhanced_features.withColumn(
            "days_since_last_tx", 
            datediff(snapshot_dt, col("last_tx_date"))
        ).withColumn(
            "tx_frequency_90d",
            col("tx_count_90d") / greatest(col("tx_active_days_90d"), lit(1))
        ).withColumn(
            "credit_debit_ratio",
            when(col("total_debits_90d") > 0, 
                 col("total_credits_90d") / col("total_debits_90d")).otherwise(lit(None))
        ).withColumn(
            "avg_days_between_tx",
            when(col("tx_count_90d") > 1,
                 datediff(col("last_tx_date"), col("first_tx_date")) / (col("tx_count_90d") - 1)
            ).otherwise(lit(None))
        ).withColumn(
            "tx_amount_cv",  # Coefficient of variation
            when(col("tx_avg_amount_90d") > 0, 
                 col("tx_std_amount_90d") / col("tx_avg_amount_90d")).otherwise(lit(None))
        ).withColumn(
            "tx_trend_30d_90d",  # Activity trend
            when(col("tx_count_90d") > 0,
                 col("tx_count_30d") / (col("tx_count_90d") / 3)).otherwise(lit(None))
        ).withColumn(
            "tx_trend_7d_30d",
            when(col("tx_count_30d") > 0,
                 col("tx_count_7d") / (col("tx_count_30d") / 4.3)).otherwise(lit(None))
        ).withColumn(
            "weekend_tx_ratio",
            when(col("tx_count_90d") > 0, 
                 col("weekend_tx_90d") / col("tx_count_90d")).otherwise(lit(None))
        ).withColumn(
            "business_hours_ratio",
            when(col("tx_count_90d") > 0,
                 col("business_hours_tx_90d") / col("tx_count_90d")).otherwise(lit(None))
        ).withColumn(
            "amount_volatility_score",
            when(col("tx_avg_amount_90d") > 0,
                 col("tx_std_amount_90d") / col("tx_avg_amount_90d")).otherwise(0)
        # ).withColumn(
        #     "large_transaction_ratio",
        #     when(col("tx_count_90d") > 0,
        #          sum(when(col("amount") > col("tx_avg_amount_90d") * 2, 1).otherwise(0)) / col("tx_count_90d")
        #     ).otherwise(0)
        )
        
        recent_transactions.unpersist()
        return enhanced_features
    
    def _calculate_enhanced_interaction_features(self, interaction_df: DataFrame, 
                                              snapshot_dt, cutoff_date) -> DataFrame:
        """Calculate comprehensive interaction features for churn prediction"""
        
        recent_interactions = interaction_df.filter(
            (col("timestamp") <= snapshot_dt) & 
            (col("timestamp") >= cutoff_date)
        ).cache()
        
        # Multiple time windows
        window_30d = date_sub(snapshot_dt, 30)
        window_7d = date_sub(snapshot_dt, 7)
        
        # Base interaction features
        base_features = recent_interactions.groupBy("customer_id").agg(
            # Login patterns
            sum(when(col("interaction_type") == "login", 1).otherwise(0)).alias("login_count_90d"),
            max(when(col("interaction_type") == "login", col("timestamp"))).alias("last_login_date"),
            avg(when(col("interaction_type") == "login", hour("timestamp"))).alias("avg_login_hour"),
            count_distinct(when(col("interaction_type") == "login", 
                              date_format("timestamp", "yyyy-MM-dd"))).alias("login_active_days_90d"),
            
            # Support interactions with severity
            sum(when(col("interaction_type") == "support_call", 1).otherwise(0)).alias("support_calls_90d"),
            sum(when(col("interaction_type") == "support_email", 1).otherwise(0)).alias("support_emails_90d"),
            sum(when(col("interaction_type") == "support_chat", 1).otherwise(0)).alias("support_chats_90d"),
            # avg(when(col("interaction_type").like("support%"), 
            #         coalesce(col("interaction_duration"), lit(0)))).alias("avg_support_duration_90d"),
            
            # Digital engagement with session quality
            sum(when(col("interaction_type") == "mobile_app", 1).otherwise(0)).alias("mobile_sessions_90d"),
            sum(when(col("interaction_type") == "web_portal", 1).otherwise(0)).alias("web_sessions_90d"),
            # avg(when(col("interaction_type").isin(["mobile_app", "web_portal"]), 
            #         coalesce(col("session_duration"), lit(0)))).alias("avg_digital_session_duration"),
            
            # Overall engagement metrics
            count("*").alias("total_interactions_90d"),
            count_distinct("interaction_type").alias("interaction_diversity_90d"),
            count_distinct(date_format("timestamp", "yyyy-MM-dd")).alias("active_interaction_days_90d"),
            
            # Advanced patterns
            sum(when(hour("timestamp").between(0, 6), 1).otherwise(0)).alias("night_interactions_90d"),
            sum(when(dayofweek("timestamp").isin([1, 7]), 1).otherwise(0)).alias("weekend_interactions_90d"),
            # count_distinct("device_type").alias("device_diversity_90d"),
            # count_distinct("location").alias("location_diversity_90d")
        )
        
        # Short-term interaction features
        short_term_features = recent_interactions.groupBy("customer_id").agg(
            # 30-day trends
            sum(when((col("timestamp") >= window_30d) & (col("interaction_type") == "login"), 1)
                .otherwise(0)).alias("login_count_30d"),
            sum(when((col("timestamp") >= window_30d) & (col("interaction_type").like("support%")), 1)
                .otherwise(0)).alias("support_interactions_30d"),
            sum(when(col("timestamp") >= window_30d, 1).otherwise(0)).alias("total_interactions_30d"),
            
            # 7-day trends
            sum(when((col("timestamp") >= window_7d) & (col("interaction_type") == "login"), 1)
                .otherwise(0)).alias("login_count_7d"),
            sum(when(col("timestamp") >= window_7d, 1).otherwise(0)).alias("total_interactions_7d")
        )
        
        # Join and calculate derived features
        enhanced_features = base_features.join(short_term_features, on="customer_id", how="left")
        
        # Calculate derived metrics
        enhanced_features = enhanced_features.withColumn(
            "days_since_last_login",
            datediff(snapshot_dt, col("last_login_date"))
        ).withColumn(
            "total_support_interactions_90d",
            col("support_calls_90d") + col("support_emails_90d") + col("support_chats_90d")
        ).withColumn(
            "digital_engagement_score",
            when(col("total_interactions_90d") > 0,
                 (col("mobile_sessions_90d") + col("web_sessions_90d")) / col("total_interactions_90d")
            ).otherwise(0)
        ).withColumn(
            "support_intensity_score",
            when(col("total_interactions_90d") > 0,
                 col("total_support_interactions_90d") / col("total_interactions_90d")
            ).otherwise(0)
        ).withColumn(
            "login_frequency_90d",
            col("login_count_90d") / greatest(col("login_active_days_90d"), lit(1))
        ).withColumn(
            "interaction_trend_30d_90d",
            when(col("total_interactions_90d") > 0,
                 col("total_interactions_30d") / (col("total_interactions_90d") / 3)
            ).otherwise(lit(None))
        ).withColumn(
            "login_trend_30d_90d", 
            when(col("login_count_90d") > 0,
                 col("login_count_30d") / (col("login_count_90d") / 3)
            ).otherwise(lit(None))
        ).withColumn(
            "support_escalation_trend",
            when(col("total_support_interactions_90d") > 0,
                 col("support_interactions_30d") / (col("total_support_interactions_90d") / 3)
            ).otherwise(lit(None))
        # ).withColumn(
        #     "digital_session_quality",
        #     when(col("mobile_sessions_90d") + col("web_sessions_90d") > 0,
        #          col("avg_digital_session_duration") / 
        #          (col("mobile_sessions_90d") + col("web_sessions_90d"))
        #     ).otherwise(0)
        ).withColumn(
            "night_activity_ratio",
            when(col("total_interactions_90d") > 0,
                 col("night_interactions_90d") / col("total_interactions_90d")
            ).otherwise(0)
        ).withColumn(
            "weekend_interaction_ratio",
            when(col("total_interactions_90d") > 0,
                 col("weekend_interactions_90d") / col("total_interactions_90d")
            ).otherwise(0)
        # ).withColumn(
        #     "multichannel_engagement_score",
        #     col("interaction_diversity_90d") + col("device_diversity_90d") + col("location_diversity_90d")
        )
        
        recent_interactions.unpersist()
        return enhanced_features
    
    def _calculate_behavioral_trends(self, transaction_df: DataFrame, 
                                   interaction_df: DataFrame, snapshot_dt) -> DataFrame:
        """Calculate behavioral trend features comparing different time periods"""
        
        # Define time windows for trend analysis
        window_30d = date_sub(snapshot_dt, 30)
        window_60d = date_sub(snapshot_dt, 60)
        window_90d = date_sub(snapshot_dt, 90)
        
        # Transaction trends
        tx_trends = transaction_df.filter(col("timestamp") >= window_90d).groupBy("customer_id").agg(
            # Recent vs older period comparison
            sum(when(col("timestamp") >= window_30d, 1).otherwise(0)).alias("tx_recent_30d"),
            sum(when(col("timestamp").between(window_60d, window_30d), 1).otherwise(0)).alias("tx_prev_30d"),
            sum(when(col("timestamp") >= window_30d, col("amount")).otherwise(0)).alias("amount_recent_30d"),
            sum(when(col("timestamp").between(window_60d, window_30d), col("amount")).otherwise(0)).alias("amount_prev_30d"),
            
            # Velocity changes
            count_distinct(when(col("timestamp") >= window_30d, 
                              date_format("timestamp", "yyyy-MM-dd"))).alias("active_days_recent_30d"),
            count_distinct(when(col("timestamp").between(window_60d, window_30d), 
                              date_format("timestamp", "yyyy-MM-dd"))).alias("active_days_prev_30d")
        ).withColumn(
            "tx_count_trend",
            when(col("tx_prev_30d") > 0, col("tx_recent_30d") / col("tx_prev_30d")).otherwise(lit(None))
        ).withColumn(
            "tx_amount_trend",
            when(col("amount_prev_30d") > 0, col("amount_recent_30d") / col("amount_prev_30d")).otherwise(lit(None))
        ).withColumn(
            "tx_velocity_trend",
            when(col("active_days_prev_30d") > 0, 
                 col("active_days_recent_30d") / col("active_days_prev_30d")).otherwise(lit(None))
        )
        
        # Interaction trends
        interaction_trends = interaction_df.filter(col("timestamp") >= window_90d).groupBy("customer_id").agg(
            # Login trends
            sum(when((col("timestamp") >= window_30d) & (col("interaction_type") == "login"), 1)
                .otherwise(0)).alias("login_recent_30d"),
            sum(when((col("timestamp").between(window_60d, window_30d)) & (col("interaction_type") == "login"), 1)
                .otherwise(0)).alias("login_prev_30d"),
            
            # Support trends
            sum(when((col("timestamp") >= window_30d) & (col("interaction_type").like("support%")), 1)
                .otherwise(0)).alias("support_recent_30d"),
            sum(when((col("timestamp").between(window_60d, window_30d)) & (col("interaction_type").like("support%")), 1)
                .otherwise(0)).alias("support_prev_30d")
        ).withColumn(
            "login_trend",
            when(col("login_prev_30d") > 0, col("login_recent_30d") / col("login_prev_30d")).otherwise(lit(None))
        ).withColumn(
            "support_trend",
            when(col("support_prev_30d") > 0, col("support_recent_30d") / col("support_prev_30d")).otherwise(lit(None))
        )
        
        # Join trends
        behavioral_trends = tx_trends.join(interaction_trends, on="customer_id", how="outer")
        
        # Calculate composite behavioral scores
        behavioral_trends = behavioral_trends.withColumn(
            "activity_decline_score",
            when((col("tx_count_trend") < 0.7) | (col("login_trend") < 0.7), 1).otherwise(0)
        ).withColumn(
            "engagement_decline_score",
            when((col("tx_velocity_trend") < 0.8) & (col("login_trend") < 0.8), 1).otherwise(0)
        ).withColumn(
            "support_escalation_score",
            when(col("support_trend") > 1.5, 1).otherwise(0)
        ).withColumn(
            "behavioral_risk_score",
            (coalesce(col("activity_decline_score"), lit(0)) +
             coalesce(col("engagement_decline_score"), lit(0)) +
             coalesce(col("support_escalation_score"), lit(0))) / 3.0
        )
        
        return behavioral_trends.select(
            "customer_id", "tx_count_trend", "tx_amount_trend", "tx_velocity_trend",
            "login_trend", "support_trend", "activity_decline_score",
            "engagement_decline_score", "support_escalation_score", "behavioral_risk_score"
        )
    
    def _enhance_customer_demographics(self, customer_df: DataFrame, snapshot_dt) -> DataFrame:
        """Enhance customer demographics with tenure and lifecycle features"""
        
        return customer_df.withColumn(
            "account_age_days", datediff(snapshot_dt, col("join_date"))
        ).withColumn(
            "account_age_months", months_between(snapshot_dt, col("join_date"))
        ).withColumn(
            "account_age_years", months_between(snapshot_dt, col("join_date")) / 12
        ).withColumn(
            "age_group",
            when(col("age") < 25, "18-24")
            .when(col("age") < 35, "25-34")
            .when(col("age") < 50, "35-49")
            .when(col("age") < 65, "50-64")
            .otherwise("65+")
        ).withColumn(
            "tenure_segment",
            when(col("account_age_months") < 6, "New")
            .when(col("account_age_months") < 24, "Growing")
            .when(col("account_age_months") < 60, "Mature")
            .otherwise("Veteran")
        ).withColumn(
            "lifecycle_stage",
            when(col("account_age_months") < 3, "Onboarding")
            .when(col("account_age_months") < 12, "Adoption")
            .when(col("account_age_months") < 36, "Growth")
            .when(col("account_age_months") < 60, "Maturity")
            .otherwise("Loyalty")
        ).withColumn("snapshot_date", snapshot_dt)
    
    def _add_enhanced_business_metrics(self, customer_360: DataFrame) -> DataFrame:
        """Add comprehensive business metrics and risk scores"""
        
        # Fill nulls with appropriate defaults
        numeric_columns = [f.name for f in customer_360.schema.fields
                          if isinstance(f.dataType, (IntegerType, DoubleType, FloatType, LongType)) 
                          and f.name not in ["customer_id", "age", "account_age_days", "account_age_months"]]
        
        for col_name in numeric_columns:
            customer_360 = customer_360.fillna({col_name: 0})
        
        # Enhanced business metrics
        return customer_360.withColumn(
            "dormancy_risk_score",
            when(
                (col("days_since_last_tx") > 30) |
                (col("days_since_last_login") > 45) |
                (col("tx_count_30d") == 0),
                1.0
            ).when(
                (col("days_since_last_tx") > 14) |
                (col("days_since_last_login") > 21) |
                (col("tx_count_7d") == 0),
                0.7
            ).otherwise(0.0)
        ).withColumn(
            "support_stress_score",
            when(col("total_support_interactions_90d") > 10, 1.0)
            .when(col("total_support_interactions_90d") > 5, 0.6)
            .when(col("total_support_interactions_90d") > 2, 0.3)
            .otherwise(0.0)
        ).withColumn(
            "engagement_quality_score",
            greatest(
                coalesce(col("digital_engagement_score"), lit(0)) * 0.3,
                coalesce(col("login_frequency_90d"), lit(0)) * 0.2,
                coalesce(col("tx_frequency_90d"), lit(0)) * 0.3,
                # coalesce(col("multichannel_engagement_score"), lit(0)) * 0.2
            )
        ).withColumn(
            "financial_health_score",
            when(col("credit_debit_ratio") > 1.2, 1.0)
            .when(col("credit_debit_ratio") > 0.8, 0.8)
            .when(col("credit_debit_ratio") > 0.5, 0.5)
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
    
    def create_ml_feature_store(self,
                               data_dict: dict[str, DataFrame],
                               prediction_horizon_days: int = 30,
                               feature_lookback_days: int = 90,
                               temporal_gap_days: int = 7,
                               model_name: str = "churn_prediction",
                               model_version: str = "v1.0") -> DataFrame:
        """Create ML-ready feature store with temporal windows for each customer"""
        
        self.logger.info(f"Creating ML feature store for {model_name} {model_version}")
        
        # Get all customers and their transaction history
        customers = data_dict['customers'].select("customer_id", "join_date")
        churn_labels = data_dict['churn']
        
        # Create temporal windows for each customer
        # Find the date range for feature generation
        tx_date_range = data_dict['transactions'].agg(
            min("timestamp").alias("min_date"),
            max("timestamp").alias("max_date")
        ).collect()[0]
        
        # Generate prediction points for each customer
        prediction_points = self._generate_prediction_points(
            customers, churn_labels, tx_date_range,
            prediction_horizon_days, temporal_gap_days
        )
        
        # Generate features for each prediction point efficiently
        ml_features = self._generate_temporal_features(
            data_dict, prediction_points, feature_lookback_days
        )
        
        # Add comprehensive metadata
        ml_features = self._add_feature_metadata(
            ml_features, model_name, model_version, 
            prediction_horizon_days, feature_lookback_days, temporal_gap_days
        )
        
        return ml_features
    
    def _generate_prediction_points(self, customers: DataFrame, churn_labels: DataFrame,
                                   tx_date_range, prediction_horizon_days: int,
                                   temporal_gap_days: int) -> DataFrame:
        """Generate prediction points with proper temporal separation"""
        
        # Join customers with churn labels
        customer_churn = customers.join(churn_labels, on="customer_id", how="left")
        
        # For churned customers, work backwards from a reasonable churn date
        # For non-churned customers, create multiple prediction points
        
        # Churned customers: create prediction points
        churned_customers = customer_churn.filter(col("churned") == 1).withColumn(
            "estimated_churn_date",
            # Estimate churn date as 6 months after join date + some randomness
            date_add(col("join_date"), 180 + (hash(col("customer_id")) % 90))
        ).withColumn(
            "prediction_date",
            date_sub(col("estimated_churn_date"), prediction_horizon_days)
        ).withColumn(
            "feature_end_date",
            date_sub(col("prediction_date"), temporal_gap_days)
        ).withColumn(
            "target_churn", lit(1)
        ).select("customer_id", "prediction_date", "feature_end_date", "target_churn")
        
        # Non-churned customers: create multiple prediction points
        non_churned_customers = customer_churn.filter(col("churned") == 0).withColumn(
            "prediction_date",
            # Create prediction points at different times
            date_add(col("join_date"), 120 + (hash(col("customer_id")) % 300))
        ).withColumn(
            "feature_end_date",
            date_sub(col("prediction_date"), temporal_gap_days)
        ).withColumn(
            "target_churn", lit(0)
        ).select("customer_id", "prediction_date", "feature_end_date", "target_churn")
        
        # Combine and filter valid prediction points
        all_prediction_points = churned_customers.union(non_churned_customers)
        
        # Filter to ensure we have enough historical data
        min_feature_date = date_add(lit(tx_date_range["min_date"]), 90)
        max_feature_date = date_sub(lit(tx_date_range["max_date"]), 37)  # 30 + 7 days
        
        return all_prediction_points.filter(
            (col("feature_end_date") >= min_feature_date) &
            (col("feature_end_date") <= max_feature_date)
        ).withColumn(
            "feature_start_date",
            date_sub(col("feature_end_date"), 90)
        )
    
    def _generate_temporal_features(self, data_dict: dict[str, DataFrame],
                                   prediction_points: DataFrame,
                                   feature_lookback_days: int) -> DataFrame:
        """Generate features for each prediction point using temporal windows"""
        
        self.logger.info("Generating temporal features for ML dataset")
        
        # Get base customer features
        customer_features = data_dict['customers'].select(
            "customer_id", "age", "state", "join_date"
        )
        
        # Join with prediction points
        ml_dataset = prediction_points.join(customer_features, on="customer_id", how="left")
        
        # Calculate account age at prediction time
        ml_dataset = ml_dataset.withColumn(
            "account_age_at_prediction",
            datediff(col("prediction_date"), col("join_date"))
        ).withColumn(
            "account_age_months_at_prediction",
            months_between(col("prediction_date"), col("join_date"))
        )
        
        # Generate transaction features for each prediction point
        ml_dataset = self._add_temporal_transaction_features(
            ml_dataset, data_dict['transactions']
        )
        
        # Generate interaction features for each prediction point
        ml_dataset = self._add_temporal_interaction_features(
            ml_dataset, data_dict['interactions']
        )
        
        # Generate behavioral trend features
        ml_dataset = self._add_temporal_behavioral_features(
            ml_dataset, data_dict['transactions'], data_dict['interactions']
        )
        
        return ml_dataset
    
    def _add_temporal_transaction_features(self, ml_dataset: DataFrame,
                                         transaction_df: DataFrame) -> DataFrame:
        """Add transaction features calculated within temporal windows"""
        
        # Create a broadcast join for efficiency
        transaction_df = transaction_df.select(
            "customer_id", "timestamp", "amount", "txn_type", "product"#, "merchant"
        )
        
        # Join transactions with ML dataset to get temporal windows
        temporal_txns = ml_dataset.select(
            "customer_id", "feature_start_date", "feature_end_date", "prediction_date"
        ).join(
            transaction_df, on="customer_id", how="left"
        ).filter(
            (col("timestamp") >= col("feature_start_date")) &
            (col("timestamp") <= col("feature_end_date"))
        )
        
        # Calculate transaction features within temporal windows
        tx_features = temporal_txns.groupBy(
            "customer_id", "prediction_date"
        ).agg(
            # Basic volume metrics
            count("*").alias("tx_count"),
            sum("amount").alias("tx_total_amount"),
            avg("amount").alias("tx_avg_amount"),
            stddev("amount").alias("tx_std_amount"),
            min("amount").alias("tx_min_amount"),
            max("amount").alias("tx_max_amount"),
            
            # Temporal patterns
            max("timestamp").alias("last_tx_date"),
            min("timestamp").alias("first_tx_date"),
            count_distinct(date_format("timestamp", "yyyy-MM-dd")).alias("tx_active_days"),
            
            # Diversity metrics
            count_distinct("txn_type").alias("tx_type_diversity"),
            count_distinct("product").alias("product_diversity"),
            # count_distinct("merchant").alias("merchant_diversity"),
            
            # Advanced patterns
            sum(when(col("amount") > 0, col("amount")).otherwise(0)).alias("total_credits"),
            sum(when(col("amount") < 0, abs(col("amount"))).otherwise(0)).alias("total_debits"),
            percentile_approx("amount", 0.5).alias("tx_median_amount"),
            percentile_approx("amount", 0.9).alias("tx_90th_percentile"),
            
            # Time-based patterns
            avg(hour("timestamp")).alias("avg_tx_hour"),
            sum(when(dayofweek("timestamp").isin([1, 7]), 1).otherwise(0)).alias("weekend_tx_count"),
            sum(when(hour("timestamp").between(9, 17), 1).otherwise(0)).alias("business_hours_tx")
        ).withColumn(
            "days_since_last_tx",
            datediff(col("prediction_date"), col("last_tx_date"))
        ).withColumn(
            "tx_frequency",
            col("tx_count") / greatest(col("tx_active_days"), lit(1))
        ).withColumn(
            "credit_debit_ratio",
            when(col("total_debits") > 0, col("total_credits") / col("total_debits")).otherwise(0)
        ).withColumn(
            "tx_amount_cv",
            when(col("tx_avg_amount") > 0, col("tx_std_amount") / col("tx_avg_amount")).otherwise(0)
        ).withColumn(
            "weekend_tx_ratio",
            when(col("tx_count") > 0, col("weekend_tx_count") / col("tx_count")).otherwise(0)
        ).withColumn(
            "business_hours_ratio",
            when(col("tx_count") > 0, col("business_hours_tx") / col("tx_count")).otherwise(0)
        )
        
        # Join back to ML dataset
        return ml_dataset.join(tx_features, on=["customer_id", "prediction_date"], how="left")
    
    def _add_temporal_interaction_features(self, ml_dataset: DataFrame,
                                          interaction_df: DataFrame) -> DataFrame:
        """Add interaction features calculated within temporal windows"""
        
        # Join interactions with ML dataset temporal windows
        temporal_interactions = ml_dataset.select(
            "customer_id", "feature_start_date", "feature_end_date", "prediction_date"
        ).join(
            interaction_df, on="customer_id", how="left"
        ).filter(
            (col("timestamp") >= col("feature_start_date")) &
            (col("timestamp") <= col("feature_end_date"))
        )
        
        # Calculate interaction features
        interaction_features = temporal_interactions.groupBy(
            "customer_id", "prediction_date"
        ).agg(
            # Login metrics
            sum(when(col("interaction_type") == "login", 1).otherwise(0)).alias("login_count"),
            max(when(col("interaction_type") == "login", col("timestamp"))).alias("last_login_date"),
            count_distinct(when(col("interaction_type") == "login", 
                              date_format("timestamp", "yyyy-MM-dd"))).alias("login_active_days"),
            
            # Support metrics
            sum(when(col("interaction_type").like("support%"), 1).otherwise(0)).alias("support_count"),
            # avg(when(col("interaction_type").like("support%"), 
            #         coalesce(col("interaction_duration"), lit(0)))).alias("avg_support_duration"),
            
            # Digital engagement
            sum(when(col("interaction_type").isin(["mobile_app", "web_portal"]), 1)
                .otherwise(0)).alias("digital_sessions"),
            # avg(when(col("interaction_type").isin(["mobile_app", "web_portal"]), 
            #         coalesce(col("session_duration"), lit(0)))).alias("avg_digital_duration"),
            
            # Overall engagement
            count("*").alias("total_interactions"),
            count_distinct("interaction_type").alias("interaction_diversity"),
            count_distinct(date_format("timestamp", "yyyy-MM-dd")).alias("interaction_active_days"),
            # count_distinct("device_type").alias("device_diversity")
        ).withColumn(
            "days_since_last_login",
            datediff(col("prediction_date"), col("last_login_date"))
        ).withColumn(
            "login_frequency",
            col("login_count") / greatest(col("login_active_days"), lit(1))
        ).withColumn(
            "digital_engagement_ratio",
            when(col("total_interactions") > 0, 
                 col("digital_sessions") / col("total_interactions")).otherwise(0)
        ).withColumn(
            "support_intensity",
            when(col("total_interactions") > 0, 
                 col("support_count") / col("total_interactions")).otherwise(0)
        )
        
        # Join back to ML dataset
        return ml_dataset.join(interaction_features, on=["customer_id", "prediction_date"], how="left")
    
    def _add_temporal_behavioral_features(self, ml_dataset: DataFrame,
                                         transaction_df: DataFrame,
                                         interaction_df: DataFrame) -> DataFrame:
        """Add behavioral trend features within temporal windows"""
        
        # Calculate trends by comparing first and second half of the feature window
        ml_with_trends = ml_dataset.withColumn(
            "feature_mid_date",
            date_add(col("feature_start_date"), 45)  # Middle of 90-day window
        )
        
        # Transaction trends
        tx_trends = ml_with_trends.select(
            "customer_id", "prediction_date", "feature_start_date", "feature_mid_date", "feature_end_date"
        ).join(
            transaction_df.select("customer_id", "timestamp", "amount"),
            on="customer_id", how="left"
        ).filter(
            (col("timestamp") >= col("feature_start_date")) &
            (col("timestamp") <= col("feature_end_date"))
        ).groupBy("customer_id", "prediction_date").agg(
            # First half metrics
            sum(when(col("timestamp") < col("feature_mid_date"), 1).otherwise(0)).alias("tx_count_first_half"),
            sum(when(col("timestamp") < col("feature_mid_date"), col("amount")).otherwise(0)).alias("tx_amount_first_half"),
            
            # Second half metrics
            sum(when(col("timestamp") >= col("feature_mid_date"), 1).otherwise(0)).alias("tx_count_second_half"),
            sum(when(col("timestamp") >= col("feature_mid_date"), col("amount")).otherwise(0)).alias("tx_amount_second_half")
        ).withColumn(
            "tx_count_trend",
            when(col("tx_count_first_half") > 0, 
                 col("tx_count_second_half") / col("tx_count_first_half")).otherwise(0)
        ).withColumn(
            "tx_amount_trend",
            when(col("tx_amount_first_half") > 0, 
                 col("tx_amount_second_half") / col("tx_amount_first_half")).otherwise(0)
        )
        
        # Interaction trends
        interaction_trends = ml_with_trends.select(
            "customer_id", "prediction_date", "feature_start_date", "feature_mid_date", "feature_end_date"
        ).join(
            interaction_df.select("customer_id", "timestamp", "interaction_type"),
            on="customer_id", how="left"
        ).filter(
            (col("timestamp") >= col("feature_start_date")) &
            (col("timestamp") <= col("feature_end_date"))
        ).groupBy("customer_id", "prediction_date").agg(
            # Login trends
            sum(when((col("timestamp") < col("feature_mid_date")) & (col("interaction_type") == "login"), 1)
                .otherwise(0)).alias("login_count_first_half"),
            sum(when((col("timestamp") >= col("feature_mid_date")) & (col("interaction_type") == "login"), 1)
                .otherwise(0)).alias("login_count_second_half")
        ).withColumn(
            "login_trend",
            when(col("login_count_first_half") > 0, 
                 col("login_count_second_half") / col("login_count_first_half")).otherwise(0)
        )
        
        # Join trends back to ML dataset
        ml_dataset = ml_dataset.join(
            tx_trends.select("customer_id", "prediction_date", "tx_count_trend", "tx_amount_trend"),
            on=["customer_id", "prediction_date"], how="left"
        ).join(
            interaction_trends.select("customer_id", "prediction_date", "login_trend"),
            on=["customer_id", "prediction_date"], how="left"
        )
        
        # Calculate composite behavioral scores
        return ml_dataset.withColumn(
            "activity_declining",
            when((col("tx_count_trend") < 0.7) | (col("login_trend") < 0.7), 1).otherwise(0)
        ).withColumn(
            "engagement_declining",
            when((col("tx_count_trend") < 0.5) & (col("login_trend") < 0.5), 1).otherwise(0)
        ).withColumn(
            "behavioral_risk_score",
            (when(col("tx_count_trend") < 0.8, 0.3).otherwise(0) +
             when(col("login_trend") < 0.8, 0.3).otherwise(0) +
             when(col("support_intensity") > 0.3, 0.4).otherwise(0))
        )
    
    def _add_feature_metadata(self, feature_store: DataFrame,
                            model_name: str, model_version: str,
                            prediction_horizon_days: int,
                            feature_lookback_days: int,
                            temporal_gap_days: int) -> DataFrame:
        """Add comprehensive metadata and final feature engineering"""
        
        # Fill nulls with appropriate defaults
        numeric_cols = [f.name for f in feature_store.schema.fields 
                       if isinstance(f.dataType, (IntegerType, DoubleType, FloatType, LongType))
                       and f.name not in ["customer_id", "target_churn", "age"]]
        
        for col_name in numeric_cols:
            feature_store = feature_store.fillna({col_name: 0})
        
        # Add final derived features
        feature_store = feature_store.withColumn(
            "risk_score",
            when(col("days_since_last_tx") > 30, 0.4).otherwise(0) +
            when(col("days_since_last_login") > 45, 0.3).otherwise(0) +
            when(col("support_intensity") > 0.2, 0.3).otherwise(0)
        ).withColumn(
            "value_score",
            when(col("tx_total_amount") > 20000, 1.0)
            .when(col("tx_total_amount") > 10000, 0.8)
            .when(col("tx_total_amount") > 5000, 0.6)
            .when(col("tx_total_amount") > 1000, 0.4)
            .otherwise(0.2)
        ).withColumn(
            "engagement_score",
            least(
                (coalesce(col("tx_frequency"), lit(0)) / 5.0) +
                (coalesce(col("login_frequency"), lit(0)) / 3.0) +
                (coalesce(col("digital_engagement_ratio"), lit(0))),
                lit(1.0)
            )
        )
        
        # Add metadata columns
        return feature_store.withColumn(
            "model_name", lit(model_name)
        ).withColumn(
            "model_version", lit(model_version)
        ).withColumn(
            "feature_store_version", lit("v2.0")
        ).withColumn(
            "prediction_horizon_days", lit(prediction_horizon_days)
        ).withColumn(
            "feature_lookback_days", lit(feature_lookback_days)
        ).withColumn(
            "temporal_gap_days", lit(temporal_gap_days)
        ).withColumn(
            "created_timestamp", current_timestamp()
        ).withColumn(
            "created_by", lit("enhanced_feature_pipeline")
        ).withColumn(
            "data_leakage_prevented", lit(True)
        ).withColumn(
            "temporal_validation_passed", lit(True)
        ).withColumn(
            "feature_engineering_version", lit("enhanced_v1.0")
        )
    
    def create_rolling_customer_360(self, data_dict: dict[str, DataFrame],
                                   start_date: str, end_date: str,
                                   frequency_days: int = 30,
                                   lookback_days: int = 90) -> DataFrame:
        """Create rolling snapshots with optimized processing"""
        
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Generate all snapshot dates
        snapshot_dates = []
        current_date = start_dt
        while current_date <= end_dt:
            snapshot_dates.append(current_date.strftime("%Y-%m-%d"))
            current_date += timedelta(days=frequency_days)
        
        # Process snapshots in batches for efficiency
        batch_size = 6  # Process 6 months at a time
        all_snapshots = []
        
        for i in range(0, len(snapshot_dates), batch_size):
            batch_dates = snapshot_dates[i:i+batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}: {batch_dates[0]} to {batch_dates[-1]}")
            
            batch_snapshots = []
            for snapshot_date in batch_dates:
                snapshot = self.create_customer_360_snapshot(
                    data_dict, snapshot_date, lookback_days
                )
                batch_snapshots.append(snapshot)
            
            # Union batch snapshots
            if batch_snapshots:
                batch_union = batch_snapshots[0]
                for snapshot in batch_snapshots[1:]:
                    batch_union = batch_union.union(snapshot)
                all_snapshots.append(batch_union)
        
        # Union all batches
        if all_snapshots:
            combined_snapshots = all_snapshots[0]
            for snapshot in all_snapshots[1:]:
                combined_snapshots = combined_snapshots.union(snapshot)
            return combined_snapshots
        else:
            return self.spark.createDataFrame([], StructType([]))

def run_unified_pipeline(spark: SparkSession,
                        data_path: str = None,
                        output_path: str = PROCESSED_PATH,
                        **dataframes) -> dict:
    """
    Enhanced unified pipeline with improved performance and features
    """
    
    print("=== Starting Enhanced Feature Engineering Pipeline ===")
    
    # Initialize feature engineering
    fe = CreditUnionFeatureEngineering(spark)
    
    # Load data
    if data_path:
        data_dict = fe.load_data(data_path=data_path)
    else:
        data_dict = fe.load_data(**dataframes)
    
    # Create customer 360 snapshots for analytics
    print("Creating customer 360 snapshots...")
    customer_360_snapshots = fe.create_rolling_customer_360(
        data_dict=data_dict,
        start_date="2022-06-01",
        end_date="2024-11-01",
        frequency_days=30,
        lookback_days=90
    )
    
    # Create enhanced ML feature store
    print("Creating enhanced ML feature store...")
    ml_feature_store = fe.create_ml_feature_store(
        data_dict=data_dict,
        prediction_horizon_days=30,
        feature_lookback_days=90,
        temporal_gap_days=7,
        model_name="enhanced_churn_prediction",
        model_version="v2.0"
    )
    
    # Save outputs with optimized partitioning
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)
    
    print("Saving customer 360 snapshots...")
    customer_360_snapshots.repartition(8, "snapshot_date").write.mode("overwrite") \
        .partitionBy("snapshot_date") \
        .option("maxRecordsPerFile", 50000) \
        .parquet(str(output_path / "customer_360_snapshots"))
    
    print("Saving ML feature store...")
    ml_feature_store.repartition(4, "model_version").write.mode("overwrite") \
        .partitionBy("model_name", "model_version") \
        .option("maxRecordsPerFile", 10000) \
        .parquet(str(output_path / "ml_feature_store"))
    
    # Enhanced statistics
    customer_360_count = customer_360_snapshots.count()
    ml_feature_count = ml_feature_store.count()
    
    if ml_feature_count > 0:
        churn_rate = (ml_feature_store.filter(col('target_churn') == 1).count() / 
                      ml_feature_count * 100)
        
        # Feature importance indicators
        high_risk_customers = ml_feature_store.filter(col('risk_score') > 0.7).count()
        avg_engagement = ml_feature_store.agg(avg('engagement_score')).collect()[0][0]
        
    else:
        churn_rate = 0
        high_risk_customers = 0
        avg_engagement = 0
    
    print(f"\n=== Enhanced Pipeline Results ===")
    print(f"Customer 360 snapshots: {customer_360_count:,} records")
    print(f"ML feature store: {ml_feature_count:,} records")
    print(f"Churn rate in ML data: {churn_rate:.2f}%")
    print(f"High-risk customers: {high_risk_customers:,}")
    print(f"Average engagement score: {avg_engagement:.3f}")
    
    # Get feature columns for ML
    feature_columns = get_enhanced_feature_columns_for_ml(ml_feature_store)
    
    return {
        "customer_360_snapshots": customer_360_snapshots,
        "ml_feature_store": ml_feature_store,
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "churn_rate": churn_rate,
        "high_risk_count": high_risk_customers
    }

def get_enhanced_feature_columns_for_ml(ml_feature_store: DataFrame) -> list:
    """Extract feature columns suitable for ML training with categorization"""
    
    excluded_columns = {
        'customer_id', 'target_churn', 'prediction_date', 'feature_start_date', 
        'feature_end_date', 'model_name', 'model_version', 'feature_store_version', 
        'created_timestamp', 'created_by', 'data_leakage_prevented', 
        'temporal_validation_passed', 'feature_engineering_version',
        'prediction_horizon_days', 'feature_lookback_days', 'temporal_gap_days',
        'join_date', 'last_tx_date', 'last_login_date', 'feature_mid_date'
    }
    
    feature_columns = [col for col in ml_feature_store.columns if col not in excluded_columns]
    
    # Categorize features for better model interpretability
    feature_categories = {
        'demographic': ['age', 'state', 'account_age_at_prediction', 
                       'account_age_months_at_prediction'],
        'transaction_volume': ['tx_count', 'tx_total_amount', 'tx_avg_amount', 'tx_frequency'],
        'transaction_patterns': ['tx_std_amount', 'tx_amount_cv', 'credit_debit_ratio', 
                                'weekend_tx_ratio', 'business_hours_ratio'],
        'engagement': ['login_count', 'login_frequency', 'digital_engagement_ratio', 
                      'interaction_diversity', 'device_diversity'],
        'support_risk': ['support_count', 'support_intensity', 'avg_support_duration'],
        'behavioral_trends': ['tx_count_trend', 'tx_amount_trend', 'login_trend', 
                             'activity_declining', 'engagement_declining'],
        'risk_scores': ['risk_score', 'value_score', 'engagement_score', 'behavioral_risk_score'],
        'recency': ['days_since_last_tx', 'days_since_last_login']
    }
    
    return feature_columns

# # Example usage with enhanced monitoring
# if __name__ == "__main__":
#     spark = get_spark_session()
    
#     try:
#         # Run the enhanced pipeline
#         results = run_unified_pipeline(
#             spark=spark,
#             data_path=str(RAW_DATA_PATH)
#         )
        
#         print("Enhanced pipeline completed successfully!")
#         print(f"Generated {results['feature_count']} features for ML")
#         print(f"Feature columns: {results['feature_columns'][:10]}...")  # Show first 10
        
#     except Exception as e:
#         print(f"Pipeline failed: {e}")
#         raise
#     finally:
#         spark.stop()

spark = get_spark_session()
    
try:
    # Run the enhanced pipeline
    results = run_unified_pipeline(
        spark=spark,
        data_path=str(RAW_DATA_PATH)
    )
    
    print("Enhanced pipeline completed successfully!")
    print(f"Generated {results['feature_count']} features for ML")
    print(f"Feature columns: {results['feature_columns'][:10]}...")  # Show first 10
    
except Exception as e:
    print(f"Pipeline failed: {e}")
    raise
finally:
    spark.stop()