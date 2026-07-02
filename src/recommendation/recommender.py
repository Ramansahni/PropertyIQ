import streamlit as st
import pickle
import pandas as pd
import json
import re
from config.constants import COSINE_SIM1_PATH, COSINE_SIM2_PATH, COSINE_SIM3_PATH, LOCATION_DF_PATH

@st.cache_resource
def load_recommender_data():
    """Loads similarity matrices and data for recommendation from new models structure."""
    try:
        with open(COSINE_SIM1_PATH, 'rb') as f:
            cosine_sim1 = pickle.load(f)
        with open(COSINE_SIM2_PATH, 'rb') as f:
            cosine_sim2 = pickle.load(f)
        with open(COSINE_SIM3_PATH, 'rb') as f:
            cosine_sim3 = pickle.load(f)
        with open(LOCATION_DF_PATH, 'rb') as f:
            location_df_normalized = pickle.load(f)
        
        return cosine_sim1, cosine_sim2, cosine_sim3, location_df_normalized
    except FileNotFoundError:
        st.error("Recommender models not found. Ensure export scripts were run.")
        return None, None, None, None

def get_recommendations(property_name, similarity_matrix, location_df_normalized, top_n=5):
    """Helper function to get top N recommendations from a given similarity matrix."""
    try:
        idx = location_df_normalized.index.get_loc(property_name)
    except KeyError:
        return pd.DataFrame() # Property not found
    
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    top_indices = [i[0] for i in sorted_scores[1:top_n+1]]
    top_scores = [i[1] for i in sorted_scores[1:top_n+1]]
    
    top_properties = location_df_normalized.index[top_indices].tolist()
    
    recommendations_df = pd.DataFrame({
        'PropertyName': top_properties,
        'SimilarityScore': top_scores
    })
    
    return recommendations_df

def recommend_by_facilities(property_name, cosine_sim1, location_df_normalized, top_n=5):
    return get_recommendations(property_name, cosine_sim1, location_df_normalized, top_n)

def recommend_by_price(property_name, cosine_sim2, location_df_normalized, top_n=5):
    return get_recommendations(property_name, cosine_sim2, location_df_normalized, top_n)

def recommend_by_location(property_name, cosine_sim3, location_df_normalized, top_n=5):
    return get_recommendations(property_name, cosine_sim3, location_df_normalized, top_n)

# Helpers for recommendation generation used during training / exports
def refined_parse_modified_v2(detail_str):
    try:
        details = json.loads(detail_str.replace("'", "\""))
    except:
        return {}

    extracted = {}
    for bhk, detail in details.items():
        extracted[f'building type_{bhk}'] = detail.get('building_type')
        area = detail.get('area', '')
        area_parts = area.split('-')
        if len(area_parts) == 1:
            try:
                value = float(area_parts[0].replace(',', '').replace(' sq.ft.', '').strip())
                extracted[f'area low {bhk}'] = value
                extracted[f'area high {bhk}'] = value
            except:
                extracted[f'area low {bhk}'] = None
                extracted[f'area high {bhk}'] = None
        elif len(area_parts) == 2:
            try:
                extracted[f'area low {bhk}'] = float(area_parts[0].replace(',', '').replace(' sq.ft.', '').strip())
                extracted[f'area high {bhk}'] = float(area_parts[1].replace(',', '').replace(' sq.ft.', '').strip())
            except:
                extracted[f'area low {bhk}'] = None
                extracted[f'area high {bhk}'] = None

        price_range = detail.get('price-range', '')
        price_parts = price_range.split('-')
        if len(price_parts) == 2:
            try:
                extracted[f'price low {bhk}'] = float(price_parts[0].replace('₹', '').replace(' Cr', '').replace(' L', '').strip())
                extracted[f'price high {bhk}'] = float(price_parts[1].replace('₹', '').replace(' Cr', '').replace(' L', '').strip())
                if 'L' in price_parts[0]:
                    extracted[f'price low {bhk}'] /= 100
                if 'L' in price_parts[1]:
                    extracted[f'price high {bhk}'] /= 100
            except:
                extracted[f'price low {bhk}'] = None
                extracted[f'price high {bhk}'] = None
    return extracted

def distance_to_meters(distance_str):
    try:
        if 'Km' in distance_str or 'KM' in distance_str:
            return float(distance_str.split()[0]) * 1000
        elif 'Meter' in distance_str or 'meter' in distance_str:
            return float(distance_str.split()[0])
        else:
            return None
    except:
        return None
