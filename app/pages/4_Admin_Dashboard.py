import sys
from pathlib import Path
# Add project root to system path for clean imports
root_path = Path(__file__).resolve().parent
while root_path.name and not (root_path / "requirements.txt").exists():
    root_path = root_path.parent
if (root_path / "requirements.txt").exists() and str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import streamlit as st
import sqlite3
import pandas as pd
from database.db_manager import DB_PATH
import os

if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
    st.warning("Please log in to access this feature.")
    st.stop()

if st.session_state['user'].get('is_admin') != 1:
    st.error("🚫 Unauthorized Access. You do not have administrator privileges.")
    st.stop()

st.set_page_config(page_title="Admin Dashboard", page_icon="⚙️", layout="wide")
from utils.theme import load_custom_css
load_custom_css()

st.title("⚙️ Administration Dashboard")
st.markdown("Manage users and view system database metrics.")

tab1, tab2 = st.tabs(["👥 User Management", "🗄️ System Database"])

with tab1:
    st.subheader("Registered Users")
    
    try:
        conn = sqlite3.connect(DB_PATH)
        # We don't select the password column for security
        users_df = pd.read_sql_query('SELECT id, name, email, is_admin FROM users', conn)
        conn.close()
        
        # Format the dataframe for display
        users_df['is_admin'] = users_df['is_admin'].apply(lambda x: "Yes" if x == 1 else "No")
        users_df.columns = ['User ID', 'Full Name', 'Email Address', 'Is Admin']
        
        st.dataframe(users_df, use_container_width=True, hide_index=True)
        
        st.metric("Total Registered Users", len(users_df))
    except Exception as e:
        st.error(f"Failed to load users: {e}")

with tab2:
    st.subheader("Database Information")
    
    if os.path.exists(DB_PATH):
        db_size = os.path.getsize(DB_PATH) / 1024 # KB
        st.write(f"**Database Path:** `{DB_PATH}`")
        st.write(f"**Database Size:** `{db_size:.2f} KB`")
    else:
        st.warning("Database file not found.")
        
    st.markdown("---")
    st.markdown("""
    ### Schema Reference
    The `users` table currently holds the following schema:
    - `id`: INTEGER PRIMARY KEY
    - `name`: TEXT
    - `email`: TEXT UNIQUE
    - `password`: TEXT (Bcrypt Hashed)
    - `is_admin`: INTEGER (0 for normal, 1 for admin)
    """)
