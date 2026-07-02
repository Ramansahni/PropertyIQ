import streamlit as st
import pickle
import pandas as pd
import numpy as np
from config.constants import PIPELINE_PATH, DF_PATH

@st.cache_resource
def load_model_and_data():
    """Loads the ML pipeline and sample dataframe securely from reorganized paths."""
    try:
        with open(PIPELINE_PATH, 'rb') as file:
            pipeline = pickle.load(file)
            
        with open(DF_PATH, 'rb') as file:
            df = pickle.load(file)
            
        return pipeline, df
    except FileNotFoundError:
        st.error("Model files not found. Please ensure export scripts have been run.")
        return None, None

def predict_price(pipeline, input_data):
    """Generates price prediction from the input list/dictionary."""
    # Convert input to DataFrame matching training data
    columns = ['property_type', 'sector', 'bedRoom', 'bathroom', 'balcony',
               'agePossession', 'built_up_area', 'servant room', 'store room',
               'furnishing_type', 'luxury_category', 'floor_category']
    
    input_df = pd.DataFrame([input_data], columns=columns)
    
    # Predict and transform back from log scale
    pred_log = pipeline.predict(input_df)
    pred_price = np.expm1(pred_log)[0]
    
    return pred_price
