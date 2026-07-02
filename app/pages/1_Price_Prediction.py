import sys
from pathlib import Path
# Add project root to system path for clean imports
root_path = Path(__file__).resolve().parent
while root_path.name and not (root_path / "requirements.txt").exists():
    root_path = root_path.parent
if (root_path / "requirements.txt").exists() and str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.inference.predict import load_model_and_data, predict_price
from utils.data_loader import load_analytics_data
from utils.theme import load_custom_css
from src.analytics.plots import apply_plotly_dark_theme, CHART_THEME_COLORS

if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
    st.warning("Please log in from the Home page to access this feature.")
    st.stop()

st.set_page_config(page_title="Price Predictor", page_icon="🏷️", layout="wide")
load_custom_css()

st.markdown("""
<style>
    .pred-box {
        text-align: center;
        margin-bottom: 2rem;
        border-left: 5px solid #8b5cf6;
    }
    .pred-val {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a78bfa 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        text-shadow: 0 0 20px rgba(139, 92, 246, 0.3);
    }
    .pred-range {
        font-size: 1.1rem;
        color: #94a3b8;
    }
    .metric-val { font-size: 1.5rem; font-weight: 800; color: white; }
    .metric-label { font-size: 0.9rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; margin-top: 5px;}
</style>
""", unsafe_allow_html=True)

st.title("🏷️ Property Price Predictor")
st.markdown("Fill in the property details to get an instant AI-powered price estimate.")

pipeline, df = load_model_and_data()
analytics_df = load_analytics_data()

if pipeline is None or df is None:
    st.stop()

# --- INPUT FORM ---
with st.sidebar:
    st.header("Property Details")
    sector = st.selectbox('Sector', sorted(df['sector'].unique().tolist()))
    property_type = st.radio('Property Type', ['flat', 'house'], horizontal=True)
    
    colA, colB = st.columns(2)
    with colA:
        bedRoom = st.number_input('Bedrooms', min_value=1, max_value=6, value=3)
        balcony = st.selectbox('Balconies', ['0', '1', '2', '3', '3+'])
    with colB:
        bathroom = st.number_input('Bathrooms', min_value=1, max_value=6, value=3)
        floor_category = st.selectbox('Floor Category', ['Low Floor', 'Mid Floor', 'High Floor'])
    
    built_up_area = st.number_input('Built-up Area (sq ft)', min_value=300.0, max_value=10000.0, value=1500.0, step=100.0)
    agePossession = st.selectbox('Age of Property', ['New Property', 'Relatively New', 'Moderately Old', 'Old Property', 'Under Construction'])
    furnishing_type = st.selectbox('Furnishing', ['unfurnished', 'semifurnished', 'furnished'])
    
    st.markdown("---")
    luxury_score = st.slider('Luxury Score (0-10)', min_value=0, max_value=10, value=5)
    
    # Map luxury score to category
    if luxury_score <= 3:
        luxury_category = 'Low'
    elif luxury_score <= 7:
        luxury_category = 'Medium'
    else:
        luxury_category = 'High'
        
    st.markdown("---")
    st.write("Extra Rooms")
    c1, c2 = st.columns(2)
    with c1:
        has_study = st.checkbox("Study Room")
        has_servant = st.checkbox("Servant Room")
    with c2:
        has_pooja = st.checkbox("Pooja Room")
        has_store = st.checkbox("Store Room")
        
    servant_val = 1.0 if has_servant else 0.0
    store_val = 1.0 if has_store else 0.0

    predict_btn = st.button("Predict Price", type="primary", use_container_width=True)

# --- OUTPUT AREA ---
if predict_btn:
    input_data = [
        property_type, sector, bedRoom, bathroom, balcony,
        agePossession, built_up_area, servant_val, store_val,
        furnishing_type, luxury_category, floor_category
    ]
    
    with st.spinner("Analyzing market data..."):
        pred = predict_price(pipeline, input_data)
        
        # Calculations
        lower_bound = pred * 0.87 # -13%
        upper_bound = pred * 1.13 # +13%
        price_per_sqft = (pred * 10000000) / built_up_area
        
        # Sector Rank
        sector_prices = analytics_df.groupby('sector')['price'].mean().sort_values(ascending=False)
        total_sectors = len(sector_prices)
        try:
            sector_rank = sector_prices.index.get_loc(sector) + 1
        except:
            sector_rank = "N/A"
            
        # Display Results
        st.markdown(f"""
        <div class="glass-card pred-box">
            <p style="margin:0; color:#cbd5e1; font-weight:600; text-transform:uppercase; letter-spacing: 1.5px;">Estimated Market Value</p>
            <h1 class="pred-val">₹ {pred:.2f} Cr</h1>
            <p class="pred-range">Confidence Range: ₹ {lower_bound:.2f} Cr - ₹ {upper_bound:.2f} Cr (±13%)</p>
        </div>
        """, unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="glass-card" style="text-align: center;"><div class="metric-val">₹ {price_per_sqft:,.0f}</div><div class="metric-label">Price per Sq.Ft</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="glass-card" style="text-align: center;"><div class="metric-val">#{sector_rank} / {total_sectors}</div><div class="metric-label">Sector Rank</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="glass-card" style="text-align: center;"><div class="metric-val">{luxury_category}</div><div class="metric-label">Luxury Tier</div></div>', unsafe_allow_html=True)
            
        st.markdown("---")
        
        # Factor Breakdown Chart (Simulated Feature Contributions)
        st.subheader("📊 Price Factor Breakdown")
        st.write("How your inputs contributed to the final estimate (Simulated baseline).")
        
        base_price = pred * 0.4
        area_contrib = pred * 0.3 * (built_up_area / 2000)
        bed_contrib = pred * 0.15 * (bedRoom / 3)
        lux_contrib = pred * 0.1 * (luxury_score / 5)
        other_contrib = pred - (base_price + area_contrib + bed_contrib + lux_contrib)
        
        factors = ['Base Market Price', 'Area Impact', 'BHK Impact', 'Luxury Impact', 'Other Features']
        values = [base_price, area_contrib, bed_contrib, lux_contrib, other_contrib]
        
        fig = go.Figure(go.Waterfall(
            orientation = "v",
            measure = ["relative", "relative", "relative", "relative", "total"],
            x = factors,
            textposition = "outside",
            text = [f"₹{v:.2f}Cr" for v in values],
            y = values,
            connector = {"line":{"color":"rgba(255,255,255,0.2)"}},
            decreasing = {"marker":{"color":"#f43f5e"}},
            increasing = {"marker":{"color":"#10b981"}},
            totals = {"marker":{"color":"#8b5cf6"}}
        ))
        fig.update_layout(title="Contribution of Factors to Final Price", showlegend=False)
        fig = apply_plotly_dark_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Comparable Properties Table
        st.subheader("🔍 Comparable Properties")
        st.write(f"Similar {property_type}s in {sector}.")
        
        # Filter dataset
        comparables = analytics_df[
            (analytics_df['sector'] == sector) & 
            (analytics_df['property_type'] == property_type) &
            (analytics_df['bedRoom'] >= bedRoom - 1) & (analytics_df['bedRoom'] <= bedRoom + 1)
        ].copy()
        
        # Calculate diff in area to sort
        comparables['area_diff'] = abs(comparables['built_up_area'] - built_up_area)
        comparables = comparables.sort_values('area_diff').head(5)
        
        if comparables.empty:
            st.info("No highly comparable properties found in the dataset for this exact configuration.")
        else:
            disp_df = comparables[['price', 'bedRoom', 'bathroom', 'built_up_area', 'furnishing_type', 'agePossession']]
            disp_df.columns = ['Price (Cr)', 'BHK', 'Baths', 'Area (sqft)', 'Furnishing', 'Age']
            st.dataframe(disp_df, use_container_width=True, hide_index=True)
else:
    st.info("👈 Please fill in the property details on the left sidebar and click 'Predict Price' to see the estimate.")
