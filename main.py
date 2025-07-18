'''
# Customer Churn Prediction Pipeline

## Architecture Components

### Weekly (Data Processing) Components
DataIngestionService: Raw data ingestion and validation
FeatureEngineeringService: Feature creation and snapshot generation
ChurnLabelingService: Label generation and backfilling
FeatureStoreService: Feature storage and retrieval

### Weekly (Inference) Components
ModelInferenceService: Real-time churn predictions
ModelEvaluationService: Performance monitoring and validation

### Monthly (Training) Components
ModelTrainingService: Model training with temporal validation
SHAPExplainabilityService: Model interpretability generation
'''

# =============================================================================
# MAIN USAGE EXAMPLE
# =============================================================================

def main():
    """
    Example of how to use the refactored pipeline components
    """
    
    # Initialize configuration
    spark = SparkSession.builder.appName("ChurnPipeline").getOrCreate()
    
    config = PipelineConfig(
        spark_session=spark,
        raw_data_path="/data/raw",
        feature_store_path="/data/feature_store",
        model_store_path="/data/models",
        temporal_split_date="2024-01-01"
    )
    
    # Initialize services
    data_ingestion = DataIngestionService(config)
    feature_engineering = FeatureEngineeringService(config)
    churn_labeling = ChurnLabelingService(config)
    feature_store = FeatureStoreService(config)
    model_training = ModelTrainingService(config)
    model_inference = ModelInferenceService(config)
    model_evaluation = ModelEvaluationService(config)
    shap_explainability = SHAPExplainabilityService(config)
    
    # Example weekly processing workflow
    snapshot_date = "2024-01-15"
    
    print("=== WEEKLY DATA PROCESSING ===")
    
    # 1. Data ingestion and quality checks
    raw_data = data_ingestion.ingest_raw_data(snapshot_date)
    clean_data = data_ingestion.perform_data_quality_checks(raw_data)
    
    # 2. Feature engineering
    features_df = feature_engineering.create_weekly_snapshot(clean_data, snapshot_date)
    
    # 3. Churn labeling
    labels_df = churn_labeling.generate_churn_labels(clean_data, snapshot_date)
    
    # 4. Store in feature store
    feature_store.store_features(features_df, snapshot_date)
    feature_store.store_labels(labels_df, snapshot_date)
    
    print("=== WEEKLY INFERENCE ===")
    
    # 5. Load model and generate predictions (assuming model exists)
    # model = load_model()  # Implementation depends on storage format
    # predictions_df = model_inference.generate_predictions(model, features_df)
    
    print("=== MONTHLY TRAINING ===")
    
    # 6. Load historical data for training
    start_date = "2024-01-01"
    end_date = "2024-01-31"
    
    historical_features = feature_store.load_features_date_range(start_date, end_date)
    historical_labels = feature_store.load_labels_date_range(start_date, end_date)
    
    # 7. Train model
    model = model_training.train_model(historical_features, historical_labels)
    
    # 8. Generate SHAP explanations
    shap_explanations = shap_explainability.generate_shap_explanations(model, historical_features)
    feature_importance = shap_explainability.generate_global_feature_importance(model, historical_features)
    
    print("Pipeline execution completed successfully!")


if __name__ == "__main__":
    main()