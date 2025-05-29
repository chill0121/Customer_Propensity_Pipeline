# Customer Propensity

Repository for a customer propensity modeling pipeline leveraging Apache Spark (PySpark), Amazon SageMaker, and API endpoints to deliver actionable business insights.
The initial focus is on churn prediction, using SHAP values to explain model outputs and inform customer segmentation. These enriched customer profiles will support the development of a next-best-offer model.

This end-to-end, data-driven workflow enables more effective decision-making by optimizing resource allocation toward high-propensity customers and deepening insights into customer behavior and preferences.

## Repository File Structure
```
Customer_Propensity/
│
├── 1_data_ingestion/
│   ├── raw/                     # Raw unprocessed data
│   ├── external/                # 3rd party data (economic indicators, etc.)
│   ├── processed/               # Cleaned & transformed datasets
│   ├── config/
│   │   └── db_connections.json  # Database connection information
│   └── scripts/
│       └── data_ingest.py       # Scripts to load from databases, APIs, etc.
│
├── 2_data_processing/
│   ├── clean_merge.py/          # Aggregated account info, demographics
│   ├── feature_engineering.py/  # Lagged, rolling stats, Fourier transforms, etc.
│   └── eda_notebooks/           # Exploratory Data Analysis
│       ├── churn_eda.ipynb
│       └── segmentation_eda.ipynb
│
├── 3_modeling/
│   ├── churn_prediction/
│   │   ├── train_xgboost.py     # Baseline churn model
│   │   ├── evaluate.py          # Metrics output
│   │   └── metrics_report.py    # Logging metrics report, ROC curves, lift charts - to ./reports/
│   │
│   ├── segmentation/
│   │   ├── embedding.py/        # Dimensionality-reduced feature space (e.g., UMAP)
│   │   ├── clustering.py/       # KMeans, HDBSCAN, etc.
│   │   ├── interpretation.py/   # SHAP values, decision trees for cluster explanation
│   │   └── segment_profiles.md  # Segment definitions and characteristics
│   │
│   └── scripts/
│       ├── data_split.py
│       └── train_models.py
│
├── 4_api_service/
│   ├── app/                     # FastAPI app
│   │   ├── main.py              # API endpoints
│   │   ├── models/              # ML model loading logic
│   │   └── utils/               # Input validation, logging, etc.
│   └── tests/
│       └── test_api.py
│
├── 5_reports/                   # Reports generated from ./modeling/
│   ├── model_performance_churn.pdf
│   └── segment_summary_report.pdf
│
├── 6_dashboard/
│   ├── dashboards/              # Dash/Streamlit UI or Tableau embed
│   └── summary_stats/           # Segment-level insights, churn trends
│
├── 7_sagemaker_container/       # Container setup and scripts
│   ├── Dockerfile
│   ├── train_script.py
│   ├── inference_script.py
│   └── requirements.txt
│
├── 8_utils/
│   ├── config.json              # Global parameters, environment setup
│   └── logger.py                # Logging for test cases and model tracking
│
└── README.md
```