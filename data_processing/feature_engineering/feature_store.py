from pyspark.sql.functions import col
from pyspark.sql import DataFrame

from config.pipeline_config_loader import load_pipeline_config, load_spark_config, create_spark_session

pipeline_config = load_pipeline_config("config/pipeline_config.dev.yaml")
spark_config = load_spark_config("config/spark_config.yaml")

# Create SparkSession
spark_session = create_spark_session(spark_config)

class FeatureStoreService:
    """
    Weekly execution: Feature storage and retrieval
    """
    
    def __init__(self, pipeline_config):
        self.spark = spark_session
        self.config = pipeline_config
    
    def store_features(self, features_df: DataFrame, snapshot_date: str) -> None:
        """Store features in feature store"""
        feature_path = f"{self.config.feature_store_path}/processed/feature_store/snapshot_date={snapshot_date}"
        
        features_df.write.mode("overwrite").parquet(feature_path)
        
        print(f"Stored {features_df.count()} customer features for {snapshot_date}")
    
    def store_labels(self, labels_df: DataFrame, snapshot_date: str) -> None:
        """Store churn labels in feature store"""
        labels_path = f"{self.config.feature_store_path}/processed/feature_store/labels/snapshot_date={snapshot_date}"
        
        labels_df.write.mode("overwrite").parquet(labels_path)
        
        print(f"Stored {labels_df.count()} churn labels for {snapshot_date}")
    
    def load_features(self, snapshot_date: str) -> DataFrame:
        """Load features from feature store"""
        feature_path = f"{self.config.feature_store_path}/processed/feature_store/snapshot_date={snapshot_date}"
        
        return self.spark.read.parquet(feature_path)
    
    def load_labels(self, snapshot_date: str) -> DataFrame:
        """Load labels from feature store"""
        labels_path = f"{self.config.feature_store_path}/processed/feature_store/labels/snapshot_date={snapshot_date}"
        
        return self.spark.read.parquet(labels_path)
    
    def load_features_date_range(self, start_date: str, end_date: str) -> DataFrame:
        """Load features for a date range"""
        feature_path = f"{self.config.feature_store_path}/processed/feature_store"
        
        return self.spark.read.parquet(feature_path) \
                  .filter((col("snapshot_date") >= start_date) & 
                         (col("snapshot_date") <= end_date))
    
    def load_labels_date_range(self, start_date: str, end_date: str) -> DataFrame:
        """Load labels for a date range"""
        labels_path = f"{self.config.feature_store_path}/processed/feature_store/labels"
        
        return self.spark.read.parquet(labels_path) \
                  .filter((col("snapshot_date") >= start_date) & 
                         (col("snapshot_date") <= end_date))