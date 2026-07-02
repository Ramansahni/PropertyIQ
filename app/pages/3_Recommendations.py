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
from src.recommendation.recommender import load_recommender_data, recommend_by_facilities, recommend_by_price, recommend_by_location
from utils.data_loader import load_analytics_data
import pandas as pd
import numpy as np
from utils.theme import load_custom_css
from src.analytics.plots import apply_plotly_dark_theme, CHART_THEME_COLORS

if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
    st.warning("Please log in from the Home page to access this feature.")
    st.stop()

st.set_page_config(page_title="Recommender System", page_icon="🔍", layout="wide")
load_custom_css()

st.title("🔍 Recommender Systems")
st.markdown("Three specialized engines to find exactly what you're looking for.")

cosine_sim1, cosine_sim2, cosine_sim3, location_df_normalized = load_recommender_data()
df = load_analytics_data()

if location_df_normalized is None or df is None:
    st.stop()

property_list = sorted(location_df_normalized.index.tolist())

tab1, tab2, tab3 = st.tabs(["🏆 Facility-Based", "💰 Price-Based", "📍 Location-Based"])

# --- TAB 1: FACILITY BASED ---
with tab1:
    st.header("Facility-Based Recommendations")
    st.markdown("Find properties with identical amenities and facilities.")
    
    colA, colB = st.columns([1, 2])
    with colA:
        selected_prop_fac = st.selectbox("Select a Property you like", property_list, key="fac_sel")
        if st.button("Find Similar Facilities", type="primary"):
            recs = recommend_by_facilities(selected_prop_fac, cosine_sim1, location_df_normalized, top_n=5)
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
            recs = recommend_by_price(selected_prop_price, cosine_sim2, location_df_normalized, top_n=5)
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
