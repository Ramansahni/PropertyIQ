import sys
from pathlib import Path
# Add project root to system path for clean imports
root_path = Path(__file__).resolve().parent
while root_path.name and not (root_path / "requirements.txt").exists():
    root_path = root_path.parent
if (root_path / "requirements.txt").exists() and str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import streamlit as st
import time

if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
    st.warning("Please log in from the Home page to access this feature.")
    st.stop()

st.set_page_config(page_title="User Profile", page_icon="👤", layout="centered")

st.title("👤 Your Profile")

user = st.session_state['user']

st.markdown(f"""
<div style="background-color: white; padding: 2rem; border-radius: 10px; margin-bottom: 2rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); text-align: center;">
    <div style="width: 100px; height: 100px; background-color: #4F46E5; color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 40px; font-weight: bold; margin: 0 auto 1rem auto;">
        {user['name'][0].upper()}
    </div>
    <h2 style="margin: 0;">{user['name']}</h2>
    <p style="color: #6B7280; font-size: 1.1rem;">{user['email']}</p>
</div>
""", unsafe_allow_html=True)

st.subheader("Account Actions")

if st.button("Logout", type="primary", use_container_width=True):
    st.session_state['logged_in'] = False
    st.session_state['user'] = None
    st.success("You have been logged out successfully. Redirecting...")
    time.sleep(1)
    st.switch_page("Home.py")
