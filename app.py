# app.py

import streamlit as st
import psycopg2
import pandas as pd
import bcrypt
from dotenv import load_dotenv
import os
from datetime import datetime

# Load env variables
load_dotenv()

DB_PARAMS = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

# ------------------------- DATABASE FUNCTIONS -------------------------

def get_connection():
    return psycopg2.connect(**DB_PARAMS)

def get_user(email):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, name, username, email, password, role FROM users WHERE email=%s", (email,))
    result = cur.fetchone()
    conn.close()
    return result

def log_action(user_id, action):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO audit_log (user_id, action) VALUES (%s, %s)", (user_id, action))
    conn.commit()
    conn.close()

# ------------------------- AUTHENTICATION -------------------------

def login(email, password):
    user = get_user(email)
    if user and bcrypt.checkpw(password.encode(), user[4].encode() if isinstance(user[4], str) else user[4]):
        return {
            "id": user[0],
            "name": user[1],
            "username": user[2],
            "email": user[3],
            "role": user[5]
        }
    return None

# ------------------------- SESSION MANAGEMENT -------------------------

def init_session():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.user = {}

# ------------------------- FORECAST FUNCTION (EXAMPLE) -------------------------

@st.cache_data
def load_sample_data():
    data = pd.read_csv("internet_usage.csv")
    data["Year"] = pd.to_datetime(data["Year"], format="%Y")
    return data

def filter_data(df, start_year, end_year):
    return df[(df["Year"].dt.year >= start_year) & (df["Year"].dt.year <= end_year)]

def show_charts(df):
    st.line_chart(df.set_index("Year"))

# ------------------------- MAIN APP -------------------------

def main_app(user):
    st.sidebar.title(f"Welcome, {user['name']}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user = {}
        st.rerun()

    st.title("📊 Internet Usage Forecasting App")

    data = load_sample_data()

    # Sidebar Filters
    min_year, max_year = data["Year"].dt.year.min(), data["Year"].dt.year.max()
    year_range = st.sidebar.slider("Select Year Range", min_year, max_year, (min_year, max_year))

    filtered = filter_data(data, *year_range)

    st.subheader("📈 Filtered Internet Penetration Data")
    st.dataframe(filtered)

    st.subheader("📉 Forecast Chart")
    show_charts(filtered)

    if user["role"] == "superadmin":
        st.markdown("### 🛠 Admin Tools")
        st.write("You can add user management, audit trails, or report exports here.")

# ------------------------- LOGIN UI -------------------------

def login_ui():
    st.title("🔐 Login to Forecast App")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        user = login(email, password)
        if user:
            st.success(f"Welcome back, {user['name']}!")
            st.session_state.logged_in = True
            st.session_state.user = user
            log_action(user["id"], "Logged In")
            st.rerun()
        else:
            st.error("Invalid credentials")

# ------------------------- RUN APP -------------------------

init_session()

if st.session_state.logged_in:
    main_app(st.session_state.user)
else:
    login_ui()
