from pyspark.sql.functions import col
from pyspark.sql import DataFrame
from datetime import datetime

from config.pipeline_config_loader import load_pipeline_config, load_spark_config, create_spark_session

pipeline_config = load_pipeline_config("config/pipeline_config.dev.yaml")
spark_config = load_spark_config("config/spark_config.yaml")

# Create SparkSession
spark_session = create_spark_session(spark_config)

class ModelInferenceService:
    """
    Weekly execution: Model inference and predictions
    """
    
    def __init__(self, pipeline_config):
        self.spark = spark_session
        self.config = pipeline_config
    
    def generate_predictions(self, model, features_df: DataFrame) -> DataFrame:
        """
        Generate churn predictions for customers
        
        Returns:
            DataFrame with customer_id, churn_probability, churn_prediction
        """
        # Prepare features for inference
        X_inference = self._prepare_inference_data(features_df)
        
        # Generate predictions
        probabilities = model.predict_proba(X_inference)
        predictions = model.predict(X_inference)
        
        # Create results DataFrame
        customer_ids = features_df.select("customer_id").toPandas()["customer_id"].values
        
        results_pandas = pd.DataFrame({
            "customer_id": customer_ids,
            "churn_probability": probabilities[:, 1],  # Probability of churn (class 1)
            "churn_prediction": predictions,
            "prediction_date": datetime.now().strftime('%Y-%m-%d')
        })
        
        # Convert back to Spark DataFrame
        return self.spark.createDataFrame(results_pandas)
    
    def _prepare_inference_data(self, features_df: DataFrame) -> pd.DataFrame:
        """Prepare features for model inference"""
        # Select feature columns (exclude metadata)
        feature_columns = [col for col in features_df.columns 
                          if col not in ["customer_id", "snapshot_date"]]
        
        # Convert to pandas and handle missing values
        X = features_df.select(feature_columns).toPandas()
        X = X.fillna(0)  # Fill NaN with 0
        
        return X
    
    def score_customers(self, model, features_df: DataFrame, top_n: int = 1000) -> DataFrame:
        """
        Score customers and return top N highest risk customers
        
        Returns:
            DataFrame with top N customers ranked by churn probability
        """
        predictions_df = self.generate_predictions(model, features_df)
        
        # Rank customers by churn probability
        ranked_customers = predictions_df.orderBy(col("churn_probability").desc()) \
                                        .limit(top_n) \
                                        .withColumn("churn_rank", monotonically_increasing_id() + 1)
        
        return ranked_customers