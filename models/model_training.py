from pyspark.sql.functions import col
from pyspark.sql import DataFrame
from datetime import datetime
import xgboost as xgb
import pandas as pd
from typing import Tuple

from config.pipeline_config_loader import load_pipeline_config, load_spark_config, create_spark_session

pipeline_config = load_pipeline_config("config/pipeline_config.dev.yaml")
spark_config = load_spark_config("config/spark_config.yaml")

# Create SparkSession
spark_session = create_spark_session(spark_config)

class ModelTrainingService:
    """
    Monthly execution: Model training with temporal validation
    """
    
    def __init__(self, pipeline_config):
        self.spark = spark_session
        self.config = pipeline_config
    
    def train_model(self, features_df: DataFrame, labels_df: DataFrame):
        """
        Train XGBoost model with temporal validation
        
        Returns:
            Trained model object
        """
        # Join features with labels
        training_data = features_df.join(labels_df, ["customer_id", "snapshot_date"], "inner")
        
        # Perform customer-level + temporal split
        train_df, val_df = self._perform_data_split(training_data)
        
        # Prepare training data
        X_train, y_train = self._prepare_training_data(train_df)
        X_val, y_val = self._prepare_training_data(val_df)
        
        # Train XGBoost model
        model = self._train_xgboost_model(X_train, y_train)
        
        # Validate model performance
        self._validate_model_performance(model, X_val, y_val)
        
        # Save model
        self._save_model(model)
        
        return model
    
    def _perform_data_split(self, training_data: DataFrame) -> Tuple[DataFrame, DataFrame]:
        """Perform customer-level + temporal data split"""
        # Step 1: Customer-level split
        unique_customers = training_data.select("customer_id").distinct()
        train_customers, val_customers = self._customer_level_split(unique_customers)
        
        # Step 2: Temporal split within customer groups
        temporal_split_date = self.config.temporal_split_date
        
        train_df = training_data.join(train_customers, "customer_id", "inner") \
                               .filter(col("snapshot_date") <= temporal_split_date)
        
        val_df = training_data.join(val_customers, "customer_id", "inner") \
                             .filter(col("snapshot_date") > temporal_split_date)
        
        # Validate splits
        self._validate_data_splits(train_customers, val_customers, train_df, val_df)
        
        return train_df, val_df
    
    def _customer_level_split(self, unique_customers: DataFrame) -> Tuple[DataFrame, DataFrame]:
        """Split customers to prevent data leakage"""
        train_customers = unique_customers.sample(False, self.config.train_ratio, seed=42)
        val_customers = unique_customers.subtract(train_customers)
        
        return train_customers, val_customers
    
    def _validate_data_splits(self, train_customers: DataFrame, val_customers: DataFrame, 
                            train_df: DataFrame, val_df: DataFrame) -> None:
        """Validate data splits for no leakage"""
        # Check customer overlap
        overlap = train_customers.intersect(val_customers)
        overlap_count = overlap.count()
        
        if overlap_count > 0:
            raise ValueError(f"Customer overlap detected: {overlap_count} customers in both train and val")
        
        # Check temporal ordering
        max_train_date = train_df.agg(max("snapshot_date")).collect()[0][0]
        min_val_date = val_df.agg(min("snapshot_date")).collect()[0][0]
        
        if max_train_date and min_val_date and max_train_date > min_val_date:
            raise ValueError("Temporal ordering violated: validation data before training data")
        
        print(f"Data split validation passed:")
        print(f"  Train customers: {train_customers.count()}")
        print(f"  Val customers: {val_customers.count()}")
        print(f"  Customer overlap: {overlap_count} (should be 0)")
        print(f"  Train samples: {train_df.count()}")
        print(f"  Val samples: {val_df.count()}")
    
    def _prepare_training_data(self, df: DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for model training"""
        # Select feature columns
        feature_columns = [col for col in df.columns 
                          if col not in ["customer_id", "snapshot_date", "churn_label", "churned_date"]]
        
        # Convert to pandas
        pandas_df = df.select(feature_columns + ["churn_label"]).toPandas()
        
        # Prepare features and target
        X = pandas_df[feature_columns].fillna(0)
        y = pandas_df["churn_label"]
        
        return X, y
    
    def _train_xgboost_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train XGBoost model"""
        model = xgb.XGBClassifier(
            max_depth=6,
            n_estimators=100,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        
        model.fit(X_train, y_train)
        
        return model
    
    def _validate_model_performance(self, model, X_val: pd.DataFrame, y_val: pd.Series) -> None:
        """Validate model performance on validation set"""
        # Generate predictions
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        evaluation_service = ModelEvaluationService(self.config)
        metrics = evaluation_service._calculate_metrics(y_val, y_pred, y_prob)
        
        # Validate thresholds
        evaluation_service._validate_performance_thresholds(metrics)
    
    def _save_model(self, model) -> None:
        """Save trained model"""
        import pickle
        
        model_path = f"{self.config.model_store_path}/churn_model_{datetime.now().strftime('%Y%m%d')}.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"Model saved to: {model_path}")