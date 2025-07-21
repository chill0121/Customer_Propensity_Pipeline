from pyspark.sql import DataFrame
from typing import Dict

from config.pipeline_config_loader import load_pipeline_config, load_spark_config, create_spark_session

pipeline_config = load_pipeline_config("config/pipeline_config.dev.yaml")
spark_config = load_spark_config("config/spark_config.yaml")

# Create SparkSession
spark_session = create_spark_session(spark_config)

class ModelEvaluationService:
    """
    Weekly execution: Model evaluation and monitoring
    """
    
    def __init__(self, pipeline_config):
        self.spark = spark_session
        self.config = pipeline_config
    
    def evaluate_model_performance(self, model, features_df: DataFrame, labels_df: DataFrame) -> Dict[str, float]:
        """
        Evaluate model performance on validation data
        
        Returns:
            Dictionary with performance metrics
        """
        # Join features with labels
        evaluation_df = features_df.join(labels_df, ["customer_id", "snapshot_date"], "inner")
        
        # Generate predictions
        inference_service = ModelInferenceService(self.config)
        predictions_df = inference_service.generate_predictions(model, evaluation_df)
        
        # Join predictions with true labels
        evaluation_data = predictions_df.join(
            labels_df.select("customer_id", "churn_label"), 
            "customer_id", 
            "inner"
        ).toPandas()
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            evaluation_data["churn_label"], 
            evaluation_data["churn_prediction"],
            evaluation_data["churn_probability"]
        )
        
        # Validate performance thresholds
        self._validate_performance_thresholds(metrics)
        
        return metrics
    
    def monitor_data_drift(self, current_features: DataFrame, reference_features: DataFrame) -> Dict[str, float]:
        """
        Monitor for data drift between current and reference features
        
        Returns:
            Dictionary with drift metrics for each feature
        """
        drift_metrics = {}
        
        feature_columns = [col for col in current_features.columns 
                          if col not in ["customer_id", "snapshot_date"]]
        
        for feature in feature_columns:
            # Calculate basic statistics for drift detection
            current_stats = current_features.agg(
                avg(feature).alias("mean"),
                stddev(feature).alias("std")
            ).collect()[0]
            
            reference_stats = reference_features.agg(
                avg(feature).alias("mean"),
                stddev(feature).alias("std")
            ).collect()[0]
            
            # Calculate drift score (normalized difference in means)
            if reference_stats["std"] and reference_stats["std"] > 0:
                drift_score = abs(current_stats["mean"] - reference_stats["mean"]) / reference_stats["std"]
            else:
                drift_score = 0.0
            
            drift_metrics[feature] = drift_score
        
        return drift_metrics
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: pd.Series, y_prob: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics"""
        from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
        
        # Calculate precision at target recall
        precision_at_recall = self._calculate_precision_at_recall(y_true, y_prob, self.config.target_recall)
        
        return {
            "auc": roc_auc_score(y_true, y_prob),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "precision_at_target_recall": precision_at_recall
        }
    
    def _calculate_precision_at_recall(self, y_true: pd.Series, y_prob: pd.Series, target_recall: float) -> float:
        """Calculate precision at target recall level"""
        from sklearn.metrics import precision_recall_curve
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        
        # Find threshold closest to target recall
        recall_diff = np.abs(recall - target_recall)
        closest_idx = np.argmin(recall_diff)
        
        return precision[closest_idx]
    
    def _validate_performance_thresholds(self, metrics: Dict[str, float]) -> None:
        """Validate that model meets performance thresholds"""
        if metrics["auc"] < self.config.min_auc_threshold:
            print(f"WARNING: AUC {metrics['auc']:.3f} below threshold {self.config.min_auc_threshold}")
        
        if metrics["precision_at_target_recall"] < self.config.min_precision_threshold:
            print(f"WARNING: Precision at {self.config.target_recall:.1%} recall is {metrics['precision_at_target_recall']:.3f}, below threshold {self.config.min_precision_threshold}")
        
        print(f"Model Performance: AUC={metrics['auc']:.3f}, Precision@{self.config.target_recall:.1%}={metrics['precision_at_target_recall']:.3f}")