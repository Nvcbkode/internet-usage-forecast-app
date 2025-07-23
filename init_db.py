import os
import psycopg2
import bcrypt

def connect():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME", "forecast_app"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "yourpassword"),
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432")
    )
