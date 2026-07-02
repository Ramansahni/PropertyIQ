import streamlit as st

def load_custom_css():
    st.markdown("""
    <style>
        /* Modern Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        /* App Background Override (Soft Dark) */
        .stApp {
            background-color: #0f172a;
        }

        /* Hide Top Anchor Links from Streamlit */
        .stApp header {
            background-color: transparent !important;
        }
        
        /* Glassmorphism Classes */
        .glass-card {
            background: rgba(30, 41, 59, 0.65);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }
        .glass-card:hover {
            border: 1px solid rgba(139, 92, 246, 0.3);
            box-shadow: 0 8px 32px 0 rgba(139, 92, 246, 0.15);
            transform: translateY(-4px);
        }

        /* Buttons Styling Overrides */
        div.stButton > button:first-child {
            border-radius: 8px;
            transition: all 0.3s ease;
            font-weight: 600;
        }
        /* Primary Button Style overrides for native streamlit primary */
        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
            border: none;
            color: white;
        }
        div.stButton > button[kind="primary"]:hover {
            box-shadow: 0 0 15px rgba(139, 92, 246, 0.6);
            transform: translateY(-2px);
        }
        /* Secondary Button Style */
        div.stButton > button[kind="secondary"] {
            background: rgba(30, 41, 59, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #e2e8f0;
        }
        div.stButton > button[kind="secondary"]:hover {
            background: rgba(30, 41, 59, 0.8);
            border: 1px solid rgba(139, 92, 246, 0.5);
            color: white;
            box-shadow: 0 0 10px rgba(139, 92, 246, 0.2);
            transform: translateY(-2px);
        }

        /* Text Gradients */
        .text-gradient {
            background: linear-gradient(135deg, #a78bfa 0%, #818cf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
    </style>
    """, unsafe_allow_html=True)


