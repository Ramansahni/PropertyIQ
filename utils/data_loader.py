import pandas as pd
import streamlit as st
import re
from config.constants import PROPERTIES_DATA_PATH, LATLONG_DATA_PATH

@st.cache_data
def load_analytics_data():
    """Loads raw CSV data for analytics dashboard."""
    try:
        df = pd.read_csv(PROPERTIES_DATA_PATH)
        df['furnishing_type'] = df['furnishing_type'].replace({0.0:'unfurnished',1.0:'semifurnished',2.0:'furnished'})
        return df
    except FileNotFoundError:
        st.error("Data file not found. Ensure raw CSV is in the data/processed/ directory.")
        return None

@st.cache_data
def load_latlong_data():
    try:
        latlong = pd.read_csv(LATLONG_DATA_PATH)
        # Parse "28.3663° N, 76.9456° E" into floats
        def extract_coord(val):
            try:
                parts = val.split(',')
                lat = float(re.findall(r"[-+]?\d*\.\d+|\d+", parts[0])[0])
                lon = float(re.findall(r"[-+]?\d*\.\d+|\d+", parts[1])[0])
                return lat, lon
            except:
                return None, None
        
        latlong[['lat', 'lon']] = latlong['coordinates'].apply(lambda x: pd.Series(extract_coord(x)))
        return latlong
    except:
        return None
