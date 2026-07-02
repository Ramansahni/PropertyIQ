import pandas as pd
import logging
from config.constants import PROPERTIES_DATA_PATH
from backend.schemas.analytics import AnalyticsResponse
from utils.data_loader import load_analytics_data

logger = logging.getLogger("backend")

def get_market_analytics() -> AnalyticsResponse:
    """Retrieves and calculates high-level market metrics, fallbacking to direct load to bypass UI cache runtimes."""
    logger.info("Computing market analytics summaries...")
    try:
        df = load_analytics_data()
    except Exception as e:
        logger.warning(f"Failed loading analytics data via data_loader: {e}. Falling back to direct load.")
        df = None
        
    if df is None:
        df = pd.read_csv(PROPERTIES_DATA_PATH)
        df['furnishing_type'] = df['furnishing_type'].replace({0.0: 'unfurnished', 1.0: 'semifurnished', 2.0: 'furnished'})

    # Compute stats
    total_properties = len(df)
    median_price = float(df['price'].median())
    sectors_count = int(df['sector'].nunique())
    median_built_up_area = float(df['built_up_area'].median())
    
    # Property type breakdown
    prop_counts = df['property_type'].value_counts().to_dict()
    property_type_distribution = {str(k): int(v) for k, v in prop_counts.items()}
    
    # Furnishing type breakdown
    furn_counts = df['furnishing_type'].value_counts().to_dict()
    furnishing_type_distribution = {str(k): int(v) for k, v in furn_counts.items()}
    
    return AnalyticsResponse(
        total_properties=total_properties,
        median_price_cr=round(median_price, 2),
        sectors_count=sectors_count,
        median_built_up_area_sqft=median_built_up_area,
        property_type_distribution=property_type_distribution,
        furnishing_type_distribution=furnishing_type_distribution
    )
