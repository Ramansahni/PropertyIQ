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
from utils.data_loader import load_analytics_data, load_latlong_data
import pandas as pd
from utils.theme import load_custom_css
from src.analytics.plots import apply_plotly_dark_theme, CHART_THEME_COLORS

if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
    st.warning("Please log in from the Home page to access this feature.")
    st.stop()

st.set_page_config(page_title="Market Analytics", page_icon="📊", layout="wide")
load_custom_css()

st.markdown("""
<style>
    .metric-val { font-size: 2.5rem; font-weight: 800; color: white; }
    .metric-label { font-size: 1rem; color: #94a3b8; font-weight: 500; text-transform: uppercase; letter-spacing: 1px;}
</style>
""", unsafe_allow_html=True)

st.title("📊 Gurgaon Real Estate Analytics")
st.markdown("Interactive dashboards to understand the market deeply.")

df = load_analytics_data()
latlong = load_latlong_data()

if df is None:
    st.stop()

# --- TOP STATS ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="glass-card" style="text-align: center;"><div class="metric-val">{len(df)}</div><div class="metric-label">Properties</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="glass-card" style="text-align: center;"><div class="metric-val">₹ {df["price"].median():.2f} Cr</div><div class="metric-label">Median Price</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="glass-card" style="text-align: center;"><div class="metric-val">{df["sector"].nunique()}</div><div class="metric-label">Sectors</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="glass-card" style="text-align: center;"><div class="metric-val">{int(df["built_up_area"].median())}</div><div class="metric-label">Median Sq.Ft</div></div>', unsafe_allow_html=True)

# --- TABS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺️ Sector Price Map", 
    "📈 Price Distribution", 
    "🛏️ BHK Analysis", 
    "🏗️ Property Type", 
    "✨ Luxury vs Price", 
    "🛋️ Furnishing Impact"
])

with tab1:
    st.header("Sector Price Map")
    if latlong is not None:
        # Merge median prices with latlong
        sector_prices = df.groupby('sector')['price'].median().reset_index()
        map_df = pd.merge(sector_prices, latlong, on='sector', how='inner')
        
        fig_map = px.scatter_mapbox(
            map_df, lat="lat", lon="lon", size="price", color="price",
            hover_name="sector", hover_data={"lat":False, "lon":False, "price":':.2f'},
            color_continuous_scale="Purples", size_max=20,
            zoom=10.5, mapbox_style="carto-darkmatter", title="Gurgaon Sector Median Prices (in Cr)"
        )
        fig_map.update_layout(height=600)
        fig_map = apply_plotly_dark_theme(fig_map)
        st.plotly_chart(fig_map, use_container_width=True)
        
        st.subheader("Top 10 Most Expensive Sectors")
        top_10 = map_df.sort_values('price', ascending=False).head(10)
        fig_bar = px.bar(top_10, x='sector', y='price', text_auto='.2f', color='price', color_continuous_scale='Purples')
        fig_bar.update_layout(xaxis_title="Sector", yaxis_title="Median Price (Cr)")
        fig_bar = apply_plotly_dark_theme(fig_bar)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.error("Latitude/Longitude data missing.")

with tab2:
    st.header("Price Distribution")
    c1, c2 = st.columns(2)
    with c1:
        fig_hist = px.histogram(df, x="price", nbins=50, title="Market Price Distribution", color_discrete_sequence=[CHART_THEME_COLORS[0]])
        fig_hist.add_vline(x=df['price'].median(), line_dash="dash", line_color="#3b82f6", annotation_text="Median")
        fig_hist = apply_plotly_dark_theme(fig_hist)
        st.plotly_chart(fig_hist, use_container_width=True)
    with c2:
        fig_box = px.box(df, x="property_type", y="price", color="property_type", title="Flats vs Houses Spread", color_discrete_sequence=CHART_THEME_COLORS)
        fig_box = apply_plotly_dark_theme(fig_box)
        st.plotly_chart(fig_box, use_container_width=True)
        
    sample_df = df.sample(n=min(1000, len(df)))
    fig_scatter = px.scatter(sample_df, x="built_up_area", y="price", color="bedRoom", size="price", title="Area vs Price (Sampled 1000 properties)", hover_data=['sector'], color_continuous_scale='Purples')
    fig_scatter = apply_plotly_dark_theme(fig_scatter)
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:
    st.header("BHK Analysis")
    bhk_counts = df['bedRoom'].value_counts().reset_index()
    bhk_counts.columns = ['BHK', 'Count']
    
    c1, c2 = st.columns(2)
    with c1:
        fig_pie = px.pie(bhk_counts, names='BHK', values='Count', title="Market Share by BHK", hole=0.4, color_discrete_sequence=CHART_THEME_COLORS)
        fig_pie = apply_plotly_dark_theme(fig_pie)
        st.plotly_chart(fig_pie, use_container_width=True)
    with c2:
        bhk_price = df.groupby('bedRoom')['price'].median().reset_index()
        fig_bar2 = px.bar(bhk_price, x='bedRoom', y='price', title="Median Price per BHK", color='price', color_continuous_scale='Purples')
        fig_bar2 = apply_plotly_dark_theme(fig_bar2)
        st.plotly_chart(fig_bar2, use_container_width=True)
        
    fig_box_bhk = px.box(df, x="bedRoom", y="price", color="property_type", title="Price Spread for each BHK (Split by Type)", color_discrete_sequence=CHART_THEME_COLORS)
    fig_box_bhk = apply_plotly_dark_theme(fig_box_bhk)
    st.plotly_chart(fig_box_bhk, use_container_width=True)

with tab4:
    st.header("Property Type Breakdown")
    c1, c2 = st.columns(2)
    with c1:
        type_counts = df['property_type'].value_counts().reset_index()
        fig_pie2 = px.pie(type_counts, names='property_type', values='count', title="Flats vs Houses", color_discrete_sequence=CHART_THEME_COLORS)
        fig_pie2 = apply_plotly_dark_theme(fig_pie2)
        st.plotly_chart(fig_pie2, use_container_width=True)
    with c2:
        furn_counts = df['furnishing_type'].value_counts().reset_index()
        fig_bar_furn = px.bar(furn_counts, x='furnishing_type', y='count', title="Furnishing Type Distribution", color_discrete_sequence=[CHART_THEME_COLORS[1]])
        fig_bar_furn = apply_plotly_dark_theme(fig_bar_furn)
        st.plotly_chart(fig_bar_furn, use_container_width=True)
        
    age_price = df.groupby('agePossession')['price'].median().reset_index()
    fig_bar_age = px.bar(age_price, x='agePossession', y='price', title="Median Price by Property Age", color_discrete_sequence=[CHART_THEME_COLORS[2]])
    fig_bar_age = apply_plotly_dark_theme(fig_bar_age)
    st.plotly_chart(fig_bar_age, use_container_width=True)

with tab5:
    st.header("Luxury vs Price")
    c1, c2 = st.columns(2)
    with c1:
        # Luxury category mapping: Low -> 1, Medium -> 2, High -> 3 for trendline
        temp_df = df.copy()
        temp_df['lux_num'] = temp_df['luxury_category'].map({'Low':1, 'Medium':2, 'High':3})
        fig_scatter_lux = px.scatter(temp_df, x="lux_num", y="price", trendline="ols", title="Luxury Tier vs Price Correlation", color_discrete_sequence=[CHART_THEME_COLORS[0]])
        fig_scatter_lux.update_xaxes(tickvals=[1,2,3], ticktext=['Low', 'Medium', 'High'])
        fig_scatter_lux = apply_plotly_dark_theme(fig_scatter_lux)
        st.plotly_chart(fig_scatter_lux, use_container_width=True)
    with c2:
        fig_hist_lux = px.histogram(df, x="luxury_category", title="Distribution of Luxury Scores", color_discrete_sequence=[CHART_THEME_COLORS[1]])
        fig_hist_lux = apply_plotly_dark_theme(fig_hist_lux)
        st.plotly_chart(fig_hist_lux, use_container_width=True)
        
    lux_price = df.groupby('luxury_category')['price'].median().reset_index()
    fig_bar_lux = px.bar(lux_price, x='luxury_category', y='price', title="Median Price by Luxury Tier", color='price', color_continuous_scale='Purples')
    fig_bar_lux = apply_plotly_dark_theme(fig_bar_lux)
    st.plotly_chart(fig_bar_lux, use_container_width=True)

with tab6:
    st.header("Furnishing Impact")
    c1, c2 = st.columns(2)
    with c1:
        fig_box_furn = px.box(df, x="furnishing_type", y="price", color="furnishing_type", title="Furnishing Effect on Price Spread", color_discrete_sequence=CHART_THEME_COLORS)
        fig_box_furn = apply_plotly_dark_theme(fig_box_furn)
        st.plotly_chart(fig_box_furn, use_container_width=True)
    with c2:
        furn_share = df['furnishing_type'].value_counts().reset_index()
        fig_pie3 = px.pie(furn_share, names='furnishing_type', values='count', title="Market Share of Furnishing", hole=0.5, color_discrete_sequence=CHART_THEME_COLORS)
        fig_pie3 = apply_plotly_dark_theme(fig_pie3)
        st.plotly_chart(fig_pie3, use_container_width=True)
        
    furn_bhk = df.groupby(['bedRoom', 'furnishing_type'])['price'].median().reset_index()
    fig_bar_furn_bhk = px.bar(furn_bhk, x="bedRoom", y="price", color="furnishing_type", barmode="group", title="Median Price by Furnishing broken down by BHK", color_discrete_sequence=CHART_THEME_COLORS)
    fig_bar_furn_bhk = apply_plotly_dark_theme(fig_bar_furn_bhk)
    st.plotly_chart(fig_bar_furn_bhk, use_container_width=True)
