from dataclasses import dataclass
from typing import Dict, Any
import yaml
import os
import re
from pyspark.sql import SparkSession

# Ensure the environment variable PROJECT_ROOT is set
if 'PROJECT_ROOT' not in os.environ:
    raise EnvironmentError("Environment variable 'PROJECT_ROOT' is not set. Please set it to the project root directory, e.g., 'export PROJECT_ROOT=/path/to/project'.")

@dataclass
class PipelineConfig:
    data_paths: Dict[str, str]
    business_logic: Dict[str, int]
    modeling: Dict[str, float]
    performance_thresholds: Dict[str, float]
    processing: Dict[str, int]

@dataclass
class SparkConfig:
    app_name: str
    configs: Dict[str, Any]

def substitute_env_vars(obj):
    """Recursively substitute environment variables in strings."""
    if isinstance(obj, dict):
        return {k: substitute_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [substitute_env_vars(i) for i in obj]
    elif isinstance(obj, str):
        # Substitute ${VAR} with environment variable
        return re.sub(r'\${(\w+)}', lambda m: os.environ.get(m.group(1), m.group(0)), obj)
    else:
        return obj

def _safe_dict(section):
    return section if isinstance(section, dict) else {}

def load_pipeline_config(config_path: str) -> PipelineConfig:
    """Load pipeline config YAML into PipelineConfig dataclass, substituting env vars."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    config_dict = substitute_env_vars(config_dict)
    if not isinstance(config_dict, dict):
        raise ValueError("Config file must contain a top-level dictionary.")
    return PipelineConfig(
        data_paths=_safe_dict(config_dict.get('data_paths')),
        business_logic=_safe_dict(config_dict.get('business_logic')),
        modeling=_safe_dict(config_dict.get('modeling')),
        performance_thresholds=_safe_dict(config_dict.get('performance_thresholds')),
        processing=_safe_dict(config_dict.get('processing'))
    )

def load_spark_config(config_path: str) -> SparkConfig:
    """Load Spark config YAML into SparkConfig dataclass."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    if not isinstance(config_dict, dict):
        raise ValueError("Spark config file must contain a top-level dictionary.")
    
    return SparkConfig(
        app_name=config_dict.get('app_name', 'SparkApp'),
        configs=_safe_dict(config_dict.get('configs'))
    )

def create_spark_session(spark_config: SparkConfig) -> SparkSession:
    """Create a SparkSession from SparkConfig dataclass."""
    builder = SparkSession.builder.appName(spark_config.app_name)
    
    for key, value in spark_config.configs.items():
        builder = builder.config(key, value)
    
    return builder.getOrCreate()

def load_all_configs(pipeline_config_path: str = "config/pipeline_config.dev.yaml", 
                    spark_config_path: str = "config/spark_config.yaml") -> tuple[PipelineConfig, SparkConfig]:
    """Load both pipeline and Spark configs in one call."""
    pipeline_config = load_pipeline_config(pipeline_config_path)
    spark_config = load_spark_config(spark_config_path)
    return pipeline_config, spark_config

# Usage example:
# # Load individual configs
# from config.pipeline_config_loader import load_pipeline_config, load_spark_config

# pipeline_config = load_pipeline_config("config/pipeline_config.dev.yaml")
# spark_config = load_spark_config("config/spark_config.yaml")

# # Load both configs at once
# pipeline_config, spark_config = load_all_configs()

# # Create SparkSession
# spark_session = create_spark_session(spark_config)

# # Access config values
# print(pipeline_config.data_paths['raw_data_path'])
# print(spark_config.app_name)