import sys
from pathlib import Path
# Add project root to system path for clean imports
root_path = Path(__file__).resolve().parent
while root_path.name and not (root_path / "requirements.txt").exists():
    root_path = root_path.parent
if (root_path / "requirements.txt").exists() and str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import streamlit as st
from database.db_manager import init_db, create_user, verify_user
from utils.auth import display_captcha, reset_captcha
from utils.theme import load_custom_css
import time

st.set_page_config(
    page_title="PropertyIQ | Automate Real Estate",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load global SaaS styles
load_custom_css()

init_db()

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'user' not in st.session_state:
    st.session_state['user'] = None

# Custom CSS for Home page specific elements (Hero, Nav, Property Cards)
st.markdown("""
<style>
    /* Top Navigation Bar */
    .nav-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.5rem 2rem;
        background: transparent;
        margin-top: -3rem;
        margin-bottom: 2rem;
    }
    .logo {
        font-size: 1.75rem;
        font-weight: 800;
        text-decoration: none;
        color: white;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .nav-links {
        display: flex;
        gap: 2rem;
        font-size: 0.95rem;
        font-weight: 500;
        color: #cbd5e1;
    }
    
    /* Hero Section */
    .hero {
        background: radial-gradient(circle at 50% 50%, rgba(139, 92, 246, 0.15) 0%, rgba(15, 23, 42, 0) 50%), url('https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop') center/cover;
        position: relative;
        padding: 8rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 3rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.6) 0%, rgba(15, 23, 42, 0.95) 100%);
        z-index: 1;
    }
    .hero-content {
        position: relative;
        z-index: 2;
    }
    .hero h1 {
        color: white;
        font-size: 4.5rem;
        font-weight: 800;
        margin-bottom: 1rem;
        letter-spacing: -1px;
        line-height: 1.1;
    }
    .hero p {
        color: #94a3b8;
        font-size: 1.25rem;
        margin-bottom: 2.5rem;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
    }
    
    /* Stats Strip */
    .stats-strip {
        display: flex;
        justify-content: space-around;
        padding: 2.5rem;
        margin-bottom: 4rem;
    }
    .stat-item {
        text-align: center;
    }
    .stat-num {
        font-size: 3rem;
        font-weight: 800;
        color: white;
        text-shadow: 0 0 20px rgba(139, 92, 246, 0.4);
    }
    .stat-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 0.5rem;
    }

    /* Property Cards */
    .property-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 2rem;
        margin-bottom: 3rem;
    }
    .prop-card {
        padding: 0; /* Override glass-card padding for image flush */
        overflow: hidden;
    }
    .prop-img {
        height: 220px;
        position: relative;
    }
    .prop-badge {
        position: absolute;
        top: 15px;
        left: 15px;
        background: rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(4px);
        color: white;
        padding: 6px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 6px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .prop-content {
        padding: 1.5rem;
    }
    .prop-price {
        font-size: 1.75rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.25rem;
    }
    .prop-title {
        font-size: 1.1rem;
        color: #e2e8f0;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    .prop-location {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-bottom: 1.25rem;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .prop-meta {
        display: flex;
        justify-content: space-between;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        padding-top: 1.25rem;
        font-size: 0.85rem;
        color: #cbd5e1;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #64748b;
        font-size: 0.9rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        margin-top: 4rem;
    }
</style>
""", unsafe_allow_html=True)

# Top Navigation
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.markdown('<a href="#" class="logo"><span class="text-gradient">✦</span> PropertyIQ</a>', unsafe_allow_html=True)
with col2:
    if st.session_state['logged_in']:
        st.markdown('<div class="nav-links" style="justify-content: center;"><span>Dashboard</span> | <span>Analytics</span> | <span>Settings</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="nav-links" style="justify-content: center;"><span>Features</span> | <span>Pricing</span> | <span>Resources</span></div>', unsafe_allow_html=True)

# Login Modal Logic
if not st.session_state['logged_in']:
    with col3:
        with st.popover("Access Portal", use_container_width=True):
            tab1, tab2 = st.tabs(["Login", "Sign Up"])
            with tab1:
                login_email = st.text_input("Email", key="login_email")
                login_password = st.text_input("Password", type="password", key="login_password")
                login_captcha_passed = display_captcha("login")
                if st.button("Login to Dashboard", type="primary", use_container_width=True):
                    if not login_captcha_passed:
                        st.error("Incorrect CAPTCHA.")
                    else:
                        user = verify_user(login_email, login_password)
                        if user:
                            st.session_state['logged_in'] = True
                            st.session_state['user'] = user
                            st.success("Authentication successful!")
                            reset_captcha("login")
                            st.rerun()
                        else:
                            st.error("Invalid credentials.")
            with tab2:
                signup_name = st.text_input("Name")
                signup_email = st.text_input("Email", key="signup_email")
                signup_password = st.text_input("Password", type="password", key="signup_password")
                signup_captcha_passed = display_captcha("signup")
                if st.button("Create Account", type="primary", use_container_width=True):
                    if not signup_captcha_passed:
                        st.error("Incorrect CAPTCHA.")
                    else:
                        if create_user(signup_name, signup_email, signup_password):
                            st.success("Created! Please log in.")
                            reset_captcha("signup")
                        else:
                            st.error("Email exists.")
else:
    with col3:
        if st.button(f"Logged in as {st.session_state['user']['name']} (Sign Out)"):
            st.session_state['logged_in'] = False
            st.session_state['user'] = None
            st.rerun()

# Hero Section
st.markdown("""
<div class="hero">
    <div class="hero-content">
        <h1>Automate Your <span class="text-gradient">Real Estate</span> Smartly</h1>
        <p>Harness the power of AI to predict market trends, analyze investments, and find your next premium property with unparalleled accuracy.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Quick Links
if st.session_state['logged_in']:
    st.markdown("<h3 style='color: white; margin-bottom: 1rem;'>System Modules</h3>", unsafe_allow_html=True)
    is_admin = st.session_state['user'].get('is_admin') == 1
    
    if is_admin:
        colA, colB, colC, colD = st.columns(4)
    else:
        colA, colB, colC = st.columns(3)
        
    with colA:
        if st.button("🏷️ Price AI Engine", use_container_width=True, type="primary"):
            st.switch_page("pages/1_Price_Prediction.py")
    with colB:
        if st.button("📊 Data Analytics", use_container_width=True):
            st.switch_page("pages/2_Analytics.py")
    with colC:
        if st.button("🔍 Smart Recommender", use_container_width=True):
            st.switch_page("pages/3_Recommendations.py")
            
    if is_admin:
        with colD:
            if st.button("⚙️ Admin Console", use_container_width=True, type="secondary"):
                st.switch_page("pages/4_Admin_Dashboard.py")

# Stats Strip
st.markdown("""
<div class="stats-strip glass-card">
    <div class="stat-item"><div class="stat-num">3.5k+</div><div class="stat-label">Data Points</div></div>
    <div class="stat-item"><div class="stat-num">98%</div><div class="stat-label">Model Accuracy</div></div>
    <div class="stat-item"><div class="stat-num">11</div><div class="stat-label">Neural Pipelines</div></div>
    <div class="stat-item"><div class="stat-num">&lt;1s</div><div class="stat-label">Inference Time</div></div>
</div>
""", unsafe_allow_html=True)

# Featured Properties
st.markdown("<h3 style='color: white; margin-bottom: 1.5rem;'>Premium Listings</h3>", unsafe_allow_html=True)
st.markdown("""
<div class="property-grid">
    <div class="glass-card prop-card">
        <div class="prop-img" style="background: url('https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?w=500&auto=format&fit=crop') center/cover;">
            <div class="prop-badge">Verified Deal</div>
        </div>
        <div class="prop-content">
            <div class="prop-price">₹ 5.50 Cr</div>
            <div class="prop-title">4 BHK Flat in DLF The Arbour</div>
            <div class="prop-location">📍 Sector 63, Cyber City</div>
            <div class="prop-meta">
                <span>🛏️ 4 Beds</span>
                <span>📐 3950 Sq.Ft.</span>
                <span>✨ Premium</span>
            </div>
        </div>
    </div>
    <div class="glass-card prop-card">
        <div class="prop-img" style="background: url('https://images.unsplash.com/photo-1600585154340-be6161a56a0c?w=500&auto=format&fit=crop') center/cover;">
            <div class="prop-badge">High Yield</div>
        </div>
        <div class="prop-content">
            <div class="prop-price">₹ 2.85 Cr</div>
            <div class="prop-title">3 BHK Flat in M3M Golf Estate</div>
            <div class="prop-location">📍 Sector 65, Golf Course Ext</div>
            <div class="prop-meta">
                <span>🛏️ 3 Beds</span>
                <span>📐 1975 Sq.Ft.</span>
                <span>✨ Luxury</span>
            </div>
        </div>
    </div>
    <div class="glass-card prop-card">
        <div class="prop-img" style="background: url('https://images.unsplash.com/photo-1600047509807-ba8f99d2cdde?w=500&auto=format&fit=crop') center/cover;">
            <div class="prop-badge">Trending</div>
        </div>
        <div class="prop-content">
            <div class="prop-price">₹ 0.82 Cr</div>
            <div class="prop-title">2 BHK Flat in Signature Global</div>
            <div class="prop-location">📍 Sector 36, Sohna Road</div>
            <div class="prop-meta">
                <span>🛏️ 2 Beds</span>
                <span>📐 850 Sq.Ft.</span>
                <span>✨ Standard</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    © 2026 PropertyIQ SaaS. Powered by Deep Learning & Predictive Analytics.
</div>
""", unsafe_allow_html=True)
