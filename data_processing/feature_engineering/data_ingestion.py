from pyspark.sql.functions import col, max as spark_max, when
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, 
    BooleanType, DateType, DoubleType, TimestampType
)
from pyspark.sql import DataFrame
from datetime import datetime
from typing import Dict

from config.pipeline_config_loader import load_pipeline_config, load_spark_config, create_spark_session

pipeline_config = load_pipeline_config("config/pipeline_config.dev.yaml")
spark_config = load_spark_config("config/spark_config.yaml")

# Create SparkSession
spark_session = create_spark_session(spark_config)

# =============================================================================
# WEEKLY DATA PROCESSING COMPONENTS
# =============================================================================

class DataIngestionService:
    """
    Weekly execution: Raw data ingestion, schema validation, and data quality checks
    """
    
    def __init__(self, pipeline_config):
        self.spark = spark_session
        self.config = pipeline_config
        self.expected_schemas = self._define_expected_schemas()
    
    def ingest_raw_data(self, data_date: str) -> Dict[str, DataFrame]:
        """
        Ingest raw data for a specific date
        
        Returns:
            Dict containing customers, transactions, and interactions DataFrames
        """
        raw_data = {
            'customers': self._load_customers_data(data_date),
            'transactions': self._load_transactions_data(data_date),
            'interactions': self._load_interactions_data(data_date)
        }
        
        # Validate schemas
        self._validate_schemas(raw_data)
        
        return raw_data
    
    def perform_data_quality_checks(self, raw_data: Dict[str, DataFrame]) -> Dict[str, DataFrame]:
        """
        Perform comprehensive data quality checks
        
        Returns:
            Cleaned and validated data
        """
        cleaned_data = {}
        
        for table_name, df in raw_data.items():
            print(f"Performing quality checks on {table_name}...")
            
            # Check for duplicates
            self._check_duplicates(df, table_name)
            
            # Check for null values in critical columns
            self._check_null_values(df, table_name)
            
            # Check data freshness
            self._check_data_freshness(df, table_name)
            
            # Apply data cleaning
            cleaned_data[table_name] = self._clean_data(df, table_name)
        
        return cleaned_data
    
    def _define_expected_schemas(self) -> Dict[str, StructType]:
        """Define expected schemas for validation"""
        return {
            'customers': StructType([
                StructField("customer_id", StringType(), False),
                StructField("age", IntegerType(), True),
                StructField("state", StringType(), True),
                StructField("has_credit_card", BooleanType(), True),
                StructField("has_loan", BooleanType(), True),
                StructField("has_checking", BooleanType(), True),
                StructField("has_savings", BooleanType(), True),
                StructField("join_date", DateType(), False),
                StructField("gender", StringType(), True)
            ]),
            'transactions': StructType([
                StructField("customer_id", StringType(), False),
                StructField("product", StringType(), False),
                StructField("amount", DoubleType(), False),
                StructField("txn_type", StringType(), False),
                StructField("timestamp", TimestampType(), False)
            ]),
            'interactions': StructType([
                StructField("customer_id", StringType(), False),
                StructField("interaction_type", StringType(), False),
                StructField("timestamp", TimestampType(), False),
                StructField("session_duration", IntegerType(), True)
            ])
        }
    
    def _load_customers_data(self, data_date: str) -> DataFrame:
        """Load customers data for specific date"""
        return self.spark.read.parquet(f"{self.config.raw_data_path}/customers/{data_date}")
    
    def _load_transactions_data(self, data_date: str) -> DataFrame:
        """Load transactions data for specific date"""
        return self.spark.read.parquet(f"{self.config.raw_data_path}/transactions/{data_date}")
    
    def _load_interactions_data(self, data_date: str) -> DataFrame:
        """Load interactions data for specific date"""
        return self.spark.read.parquet(f"{self.config.raw_data_path}/interactions/{data_date}")
    
    def _validate_schemas(self, raw_data: Dict[str, DataFrame]) -> None:
        """Validate that data conforms to expected schemas"""
        for table_name, df in raw_data.items():
            expected_schema = self.expected_schemas[table_name]
            actual_schema = df.schema
            
            # Check if schemas match (simplified validation)
            expected_fields = {field.name: field.dataType for field in expected_schema.fields}
            actual_fields = {field.name: field.dataType for field in actual_schema.fields}
            
            missing_fields = set(expected_fields.keys()) - set(actual_fields.keys())
            if missing_fields:
                raise ValueError(f"Missing fields in {table_name}: {missing_fields}")
    
    def _check_duplicates(self, df: DataFrame, table_name: str) -> None:
        """Check for duplicate records"""
        total_count = df.count()
        distinct_count = df.distinct().count()
        
        if total_count != distinct_count:
            duplicate_count = total_count - distinct_count
            print(f"WARNING: {table_name} has {duplicate_count} duplicate records")
    
    def _check_null_values(self, df: DataFrame, table_name: str) -> None:
        """Check for null values in critical columns"""
        critical_columns = {
            'customers': ['customer_id', 'join_date'],
            'transactions': ['customer_id', 'amount', 'timestamp'],
            'interactions': ['customer_id', 'timestamp']
        }
        
        for col_name in critical_columns.get(table_name, []):
            null_count = df.filter(col(col_name).isNull()).count()
            if null_count > 0:
                raise ValueError(f"Critical column {col_name} in {table_name} has {null_count} null values")
    
    def _check_data_freshness(self, df: DataFrame, table_name: str) -> None:
        """Check if data is fresh enough for processing"""
        if table_name in ['transactions', 'interactions']:
            max_timestamp = df.agg(spark_max("timestamp")).collect()[0][0]
            current_time = datetime.now()
            
            if max_timestamp and (current_time - max_timestamp).days > 7:
                print(f"WARNING: {table_name} data is {(current_time - max_timestamp).days} days old")
    
    def _clean_data(self, df: DataFrame, table_name: str) -> DataFrame:
        """Apply data cleaning transformations"""
        # Remove duplicates
        df_clean = df.distinct()
        
        # Handle specific cleaning by table
        if table_name == 'transactions':
            # Filter out invalid amounts
            df_clean = df_clean.filter(col("amount") > 0)
            
        elif table_name == 'interactions':
            # Handle negative session durations
            df_clean = df_clean.withColumn(
                "session_duration", 
                when(col("session_duration") < 0, 0).otherwise(col("session_duration"))
            )
        
        return df_clean