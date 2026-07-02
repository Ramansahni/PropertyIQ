import sys
from pathlib import Path
# Add project root path to system path for clean relative imports
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import logging.config
import pickle
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.constants import (
    PIPELINE_PATH,
    DF_PATH,
    COSINE_SIM1_PATH,
    COSINE_SIM2_PATH,
    COSINE_SIM3_PATH,
    LOCATION_DF_PATH,
)
from backend.routers import prediction, recommendation, analytics, health

# Production-style logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "backend": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("backend")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Safely manages application initialization and teardown of machine learning model cache."""
    logger.info("Initializing FastAPI Backend Lifespan...")
    logger.info("Loading machine learning models and similarity matrices once at startup...")
    
    try:
        # Load Pipeline
        logger.info(f"Loading pipeline from {PIPELINE_PATH}")
        with open(PIPELINE_PATH, 'rb') as f:
            app.state.pipeline = pickle.load(f)
            
        # Load metadata dataframe
        logger.info(f"Loading metadata dataframe from {DF_PATH}")
        with open(DF_PATH, 'rb') as f:
            app.state.df = pickle.load(f)
            
        # Load similarity matrices
        logger.info(f"Loading facility similarity matrix from {COSINE_SIM1_PATH}")
        with open(COSINE_SIM1_PATH, 'rb') as f:
            app.state.cosine_sim1 = pickle.load(f)
            
        logger.info(f"Loading price similarity matrix from {COSINE_SIM2_PATH}")
        with open(COSINE_SIM2_PATH, 'rb') as f:
            app.state.cosine_sim2 = pickle.load(f)
            
        logger.info(f"Loading location similarity matrix from {COSINE_SIM3_PATH}")
        with open(COSINE_SIM3_PATH, 'rb') as f:
            app.state.cosine_sim3 = pickle.load(f)
            
        # Load location normalized metadata
        logger.info(f"Loading location coordinates metadata from {LOCATION_DF_PATH}")
        with open(LOCATION_DF_PATH, 'rb') as f:
            app.state.location_df_normalized = pickle.load(f)
            
        logger.info("All model artifacts and similarity data loaded successfully!")
    except Exception as e:
        logger.exception("Failed to load machine learning models during startup!")
        # Load as None so that dependencies check and raise HTTP 503 instead of application hard crash
        app.state.pipeline = None
        app.state.df = None
        app.state.cosine_sim1 = None
        app.state.cosine_sim2 = None
        app.state.cosine_sim3 = None
        app.state.location_df_normalized = None

    yield
    # Shutdown actions
    logger.info("Cleaning up backend lifespan resources...")
    app.state.pipeline = None
    app.state.df = None
    app.state.cosine_sim1 = None
    app.state.cosine_sim2 = None
    app.state.cosine_sim3 = None
    app.state.location_df_normalized = None
    logger.info("Lifespan clean up complete.")

# Initialize application
app = FastAPI(
    title="PropertyIQ API Backend",
    version="1.0.0",
    description="Production-grade API endpoints for property pricing predictions, sector comparisons, and similarity recommendations in Gurgaon real estate.",
    lifespan=lifespan,
)

# CORS middleware config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
@app.get("/")
async def root():
    return {
        "service": "PropertyIQ Backend",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

app.include_router(health.router)
app.include_router(prediction.router)
app.include_router(recommendation.router)
app.include_router(analytics.router)

