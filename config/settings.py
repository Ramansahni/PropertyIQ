import os

# General project configuration settings

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 500

APP_ENV = os.getenv('APP_ENV', 'development')
DEBUG = os.getenv('DEBUG', 'True').lower() in ('true', '1', 't')
BACKEND_API_URL = os.getenv('BACKEND_API_URL', "http://127.0.0.1:8000")

