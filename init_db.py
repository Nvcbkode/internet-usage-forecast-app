# init_db.py
import psycopg2
import bcrypt
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env

# Connect to PostgreSQL using .env vars
def connect():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )
