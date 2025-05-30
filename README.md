# Customer Propensity Pipeline

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

## Project Workflow

| Stage                | Description                                                                 | Key Tools / Scripts / Directories                           |
|----------------------|-----------------------------------------------------------------------------|--------------------------------------------------------------|
| **1. Data Ingestion**| Collect raw and external data from databases, APIs, and third-party sources using PySpark for distributed loading and schema inference | `1_data_ingestion/`<br>- `data_ingest.py` (uses PySpark)<br>- `db_connections.json` |
| **2. Data Processing**| Clean, merge, and engineer features (lagged stats, transforms) using PySpark DataFrames and SQL functions for scalable prep | `2_data_processing/`<br>- `clean_merge.py` (PySpark)<br>- `feature_engineering.py` (PySpark)<br>- `eda_notebooks/` |
| **3. Modeling - Churn**| Train and evaluate churn prediction model with explainability (SHAP)       | `3_modeling/churn_prediction/`<br>- `train_xgboost.py`<br>- `evaluate.py`<br>- `metrics_report.py` |
| **4. Modeling - Segmentation**| Segment customers based on model outputs and engineered features | `3_modeling/segmentation/`<br>- `embedding.py`<br>- `clustering.py`<br>- `interpretation.py` |
| **5. API Service**    | Deploy model predictions and insights via FastAPI endpoints                | `4_api_service/app/`<br>- `main.py`<br>- `models/`<br>- `utils/` |
| **6. Reporting**      | Generate performance reports and segment summaries                         | `5_reports/`<br>- `model_performance_churn.pdf`<br>- `segment_summary_report.pdf` |
| **7. Dashboard**      | Visualize trends, segments, and key statistics in UI                       | `6_dashboard/`<br>- Dash/Streamlit or Tableau embedded apps |
| **8. Deployment (SageMaker)**| Train and deploy models in scalable environments                     | `7_sagemaker_container/`<br>- `train_script.py`<br>- `inference_script.py`<br>- `Dockerfile` |
| **9. Utilities & Configs**| Manage global settings and logging infrastructure   

## Tech Stack (Tentative)

This project combines scalable data processing, machine learning, and model deployment in a modular pipeline.

### Data Layer
- **Apache Spark (PySpark)** – Distributed data ingestion, cleaning, and feature engineering
- **PostgreSQL / S3** – Structured and unstructured data sources (pending)
- **JSON Configs** – Database credentials, environment flags

### Modeling & ML
- **XGBoost** – Baseline churn prediction model
- **SHAP** – Model explainability and feature attribution
- **UMAP** – Dimensionality reduction for customer embeddings
- **HDBSCAN** – Customer segmentation
- **Scikit-learn** – Utilities for preprocessing, evaluation

### Deployment & Serving
- **Amazon SageMaker** – Training and scalable deployment of ML models
- **FastAPI** – REST API for exposing predictions and customer segment insights
- **Docker** – Containerized model and inference environments

### Visualization & Reporting
- **Dash** – Interactive dashboards
- **Matplotlib / Seaborn / Plotly** – EDA and report visuals
- **PDF Reports** – Auto-generated summaries for stakeholders

### Development & Utilities
- **Pytest** – API and model test coverage
- **Logging & Monitoring** – Custom `logger.py`, integration with SageMaker logging
- **VS Code / Jupyter Notebooks** – Development and exploration

> _Note: This stack is expected to evolve based on project needs (e.g., cloud data warehouses, CI/CD tooling, feature store integration)._
