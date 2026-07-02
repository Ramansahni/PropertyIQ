import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset directories & paths
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

APPARTMENTS_DATA_PATH = os.path.join(RAW_DATA_DIR, 'appartments.csv')
PROPERTIES_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'gurgaon_properties_post_feature_selection_v2.csv')
LATLONG_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'latlong.csv')

# Database path
DB_PATH = os.getenv('DB_PATH', os.path.join(BASE_DIR, 'database', 'database.db'))

# Model pickle directories & paths
MODELS_DIR = os.path.join(BASE_DIR, 'models')
TRAINED_MODELS_DIR = os.path.join(MODELS_DIR, 'trained')
SIMILARITY_MODELS_DIR = os.path.join(MODELS_DIR, 'similarity')
METADATA_MODELS_DIR = os.path.join(MODELS_DIR, 'metadata')
METRICS_MODELS_DIR = os.path.join(MODELS_DIR, 'metrics')

PIPELINE_PATH = os.path.join(TRAINED_MODELS_DIR, 'pipeline.pkl')
DF_PATH = os.path.join(METADATA_MODELS_DIR, 'df.pkl')
LOCATION_DF_PATH = os.path.join(METADATA_MODELS_DIR, 'location_df_normalized.pkl')

COSINE_SIM1_PATH = os.path.join(SIMILARITY_MODELS_DIR, 'cosine_sim1.pkl')
COSINE_SIM2_PATH = os.path.join(SIMILARITY_MODELS_DIR, 'cosine_sim2.pkl')
COSINE_SIM3_PATH = os.path.join(SIMILARITY_MODELS_DIR, 'cosine_sim3.pkl')
