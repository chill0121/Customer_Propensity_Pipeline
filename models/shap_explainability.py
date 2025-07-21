from pyspark.sql.functions import col, monotonically_increasing_id, avg
from pyspark.sql import DataFrame
import shap
import pandas as pd
import numpy as np

from typing import List, Tuple

from config.pipeline_config_loader import load_pipeline_config, load_spark_config, create_spark_session

pipeline_config = load_pipeline_config("config/pipeline_config.dev.yaml")
spark_config = load_spark_config("config/spark_config.yaml")

# Create SparkSession
spark_session = create_spark_session(spark_config)

class SHAPExplainabilityService:
    """
    Monthly execution: SHAP explanations for model interpretability
    """
    
    def __init__(self, pipeline_config):
        self.spark = spark_session
        self.config = pipeline_config
    
    def generate_shap_explanations(self, model, features_df: DataFrame) -> DataFrame:
        """
        Generate SHAP explanations for model interpretability
        
        Returns:
            DataFrame with SHAP values for sampled customers
        """
        # Sample customers for computational efficiency
        sample_df = features_df.sample(
            False, 
            self.config.shap_sample_size / features_df.count(), 
            seed=42
        )
        
        # Prepare data for SHAP
        X_sample = self._prepare_shap_data(sample_df)
        
        # Generate SHAP values
        shap_values, expected_value = self._calculate_shap_values(model, X_sample)
        
        # Create explanations DataFrame
        explanations_df = self._create_shap_explanations(
            sample_df, shap_values, expected_value, X_sample.columns
        )
        
        return explanations_df
    
    def generate_global_feature_importance(self, model, features_df: DataFrame) -> DataFrame:
        """
        Generate global feature importance using SHAP
        
        Returns:
            DataFrame with feature importance rankings
        """
        # Generate SHAP explanations
        explanations_df = self.generate_shap_explanations(model, features_df)
        
        # Calculate mean absolute SHAP values per feature
        feature_columns = [col for col in explanations_df.columns 
                          if col.startswith("shap_")]
        
        importance_data = []
        for feature_col in feature_columns:
            feature_name = feature_col.replace("shap_", "")
            
            mean_abs_shap = explanations_df.agg(
                avg(abs(col(feature_col))).alias("importance")
            ).collect()[0]["importance"]
            
            importance_data.append({
                "feature_name": feature_name,
                "importance": mean_abs_shap
            })
        
        # Create importance DataFrame and rank
        importance_df = self.spark.createDataFrame(importance_data)
        
        return importance_df.orderBy(col("importance").desc()) \
                           .withColumn("rank", monotonically_increasing_id() + 1)
    
    def _prepare_shap_data(self, sample_df: DataFrame) -> pd.DataFrame:
        """Prepare data for SHAP calculation"""
        # Select feature columns
        feature_columns = [col for col in sample_df.columns 
                          if col not in ["customer_id", "snapshot_date"]]
        
        # Convert to pandas and handle missing values
        X_sample = sample_df.select(feature_columns).toPandas()
        X_sample = X_sample.fillna(0)
        
        return X_sample
    
    def _calculate_shap_values(self, model, X_sample: pd.DataFrame) -> Tuple[np.ndarray, float]:
        """Calculate SHAP values using TreeExplainer"""
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        return shap_values, explainer.expected_value
    
    def _create_shap_explanations(self, sample_df: DataFrame, shap_values: np.ndarray, 
                                 expected_value: float, feature_names: List[str]) -> DataFrame:
        """Create SHAP explanations DataFrame"""
        # Get customer IDs
        customer_ids = sample_df.select("customer_id").toPandas()["customer_id"].values
        
        # Create SHAP DataFrame
        shap_data = {
            "customer_id": customer_ids,
            "expected_value": [expected_value] * len(customer_ids)
        }
        
        # Add SHAP values for each feature
        for i, feature_name in enumerate(feature_names):
            shap_data[f"shap_{feature_name}"] = shap_values[:, i]
        
        # Convert to Spark DataFrame
        shap_pandas_df = pd.DataFrame(shap_data)
        
        return self.spark.createDataFrame(shap_pandas_df)