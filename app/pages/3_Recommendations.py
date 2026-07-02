import sys
from pathlib import Path
# Add project root to system path for clean imports
root_path = Path(__file__).resolve().parent
while root_path.name and not (root_path / "requirements.txt").exists():
    root_path = root_path.parent
if (root_path / "requirements.txt").exists() and str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests
from config.settings import BACKEND_API_URL
from config.constants import APPARTMENTS_DATA_PATH
from utils.data_loader import load_analytics_data
import pandas as pd
import numpy as np
from utils.theme import load_custom_css
from utils.plots import apply_plotly_dark_theme, CHART_THEME_COLORS

if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
    st.warning("Please log in from the Home page to access this feature.")
    st.stop()

st.set_page_config(page_title="Recommender System", page_icon="🔍", layout="wide")
load_custom_css()

st.title("🔍 Recommender Systems")
st.markdown("Three specialized engines to find exactly what you're looking for.")

df = load_analytics_data()

if df is None:
    st.stop()

# Load unique property names from the raw dataset instead of the serialized recommender data
try:
    df_app = pd.read_csv(APPARTMENTS_DATA_PATH).drop(22)
    property_list = sorted(df_app['PropertyName'].tolist())
except Exception as e:
    st.error("Error loading property listing names for recommendations.")
    st.stop()

def fetch_recommendations(property_name: str, recommendation_type: str, top_n: int = 5) -> pd.DataFrame:
    """Helper to query the similarity search backend API."""
    payload = {
        "property_name": property_name,
        "recommendation_type": recommendation_type,
        "top_n": top_n
    }
    try:
        response = requests.post(f"{BACKEND_API_URL}/recommend", json=payload, timeout=10)
        if response.status_code == 200:
            res_data = response.json()
            recs_list = res_data.get('recommendations', [])
            recs_df = pd.DataFrame(recs_list)
            if recs_df.empty:
                return pd.DataFrame(columns=['PropertyName', 'SimilarityScore'])
            # Map Pydantic casing back to column casing in frontend
            recs_df = recs_df.rename(columns={'property_name': 'PropertyName', 'similarity_score': 'SimilarityScore'})
            return recs_df
        else:
            st.error(f"Error from server: {response.text}")
            return pd.DataFrame(columns=['PropertyName', 'SimilarityScore'])
    except requests.exceptions.RequestException as e:
        st.error("Recommendation backend server is currently offline or unreachable.")
        print(f"Recommendation API call failed: {e}")
        return pd.DataFrame(columns=['PropertyName', 'SimilarityScore'])

tab1, tab2, tab3 = st.tabs(["🏆 Facility-Based", "💰 Price-Based", "📍 Location-Based"])

# --- TAB 1: FACILITY BASED ---
with tab1:
    st.header("Facility-Based Recommendations")
    st.markdown("Find properties with identical amenities and facilities.")
    
    colA, colB = st.columns([1, 2])
    with colA:
        selected_prop_fac = st.selectbox("Select a Property you like", property_list, key="fac_sel")
        if st.button("Find Similar Facilities", type="primary"):
            recs = fetch_recommendations(selected_prop_fac, "facilities", top_n=5)
            st.success("Matches found!")
            for idx, row in recs.iterrows():
                st.markdown(f"**{row['PropertyName']}** - {row['SimilarityScore']*100:.1f}% Match")
            
            with colB:
                # Simulated Radar Chart
                categories = ['Pool', 'Gym', 'Security', 'Clubhouse', 'Parks']
                fig = go.Figure()
                
                # Base property
                fig.add_trace(go.Scatterpolar(
                    r=[5, 4, 5, 4, 5], theta=categories, fill='toself', name=selected_prop_fac, line=dict(color=CHART_THEME_COLORS[1])
                ))
                # Top match
                if not recs.empty:
                    top_match = recs.iloc[0]['PropertyName']
                    sim = recs.iloc[0]['SimilarityScore']
                    r_vals = [5*sim, 4*sim, 5*sim, 4*sim, 5*sim]
                    fig.add_trace(go.Scatterpolar(
                        r=r_vals, theta=categories, fill='toself', name=top_match, line=dict(color=CHART_THEME_COLORS[0])
                    ))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])), showlegend=True, title="Facility Comparison (Simulated)")
                fig = apply_plotly_dark_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: PRICE BASED ---
with tab2:
    st.header("Price-Based Recommendations")
    st.markdown("Find properties that financially match your target profile.")
    
    colC, colD = st.columns([1, 2])
    with colC:
        selected_prop_price = st.selectbox("Select a Reference Property", property_list, key="price_sel")
        target_budget = st.number_input("Or enter exact Budget (Cr)", value=2.5, step=0.1)
        
        if st.button("Find Financial Matches", type="primary"):
            recs = fetch_recommendations(selected_prop_price, "price", top_n=5)
            st.success("Matches found!")
            for idx, row in recs.iterrows():
                st.markdown(f"**{row['PropertyName']}** - {row['SimilarityScore']*100:.1f}% Match")
                
            with colD:
                if not recs.empty:
                    # Get prices from original df if they exist (we simulate by fetching median of sector if not)
                    # For a robust bar chart, let's just plot the similarity scores
                    fig_bar = px.bar(recs, x='PropertyName', y='SimilarityScore', 
                                     title="Financial Similarity Scores", color='SimilarityScore', color_continuous_scale='Purples')
                    # Add a horizontal line for Target Budget just as a visual cue (though y-axis is score here)
                    fig_bar = apply_plotly_dark_theme(fig_bar)
                    st.plotly_chart(fig_bar, use_container_width=True)

# --- TAB 3: LOCATION BASED ---
with tab3:
    st.header("Location-Based Recommendations")
    st.markdown("Find properties geographically similar or grouped by location advantages.")
    
    colE, colF = st.columns([1, 2])
    with colE:
        loc_sector = st.selectbox("Select Target Sector", sorted(df['sector'].unique().tolist()), key="loc_sec")
        loc_budget = st.number_input("Your Budget (Cr)", value=2.0, step=0.1, key="loc_budg")
        
        # Here we don't use cosine_sim3 directly since the user wants sector + budget filter.
        # We will query the dataframe directly.
        if st.button("Search Sector", type="primary"):
            sector_props = df[df['sector'] == loc_sector].copy()
            sector_props['diff'] = abs(sector_props['price'] - loc_budget)
            recs = sector_props.sort_values('diff').head(5)
            
            if recs.empty:
                st.warning("No properties found in this sector.")
            else:
                st.success(f"Found {len(sector_props)} properties in {loc_sector}.")
                st.write("Closest to your budget:")
                st.dataframe(recs[['property_type', 'bedRoom', 'built_up_area', 'price']].head(5))
                
            with colF:
                if not sector_props.empty:
                    fig_hist = px.histogram(sector_props, x="price", title=f"Price Distribution in {loc_sector}", color_discrete_sequence=[CHART_THEME_COLORS[0]])
                    fig_hist.add_vline(x=loc_budget, line_dash="dash", line_color="#3b82f6", annotation_text="Your Budget")
                    fig_hist = apply_plotly_dark_theme(fig_hist)
                    st.plotly_chart(fig_hist, use_container_width=True)
