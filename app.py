# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from prophet import Prophet
from statsmodels.tsa.seasonal import seasonal_decompose
import io
import datetime
import numpy as np

st.set_page_config(page_title="Forecasting App", layout="wide")

st.title("\U0001F4C8 Generic Forecasting App")
st.markdown("Upload a dataset to predict future trends using various forecasting models.")

uploaded_file = st.file_uploader("\U0001F4E4 Upload CSV or Excel file", type=["xlsx", "csv"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            raw_df = pd.read_csv(uploaded_file)
        else:
            raw_df = pd.read_excel(uploaded_file)

        st.success("✅ File uploaded successfully!")
        st.dataframe(raw_df.head())

        columns = raw_df.columns.tolist()
        time_col = st.selectbox("Select Time Column (e.g. Year or Date)", options=columns)
        target_col = st.selectbox("Select Target Column (Numeric)", options=[col for col in columns if raw_df[col].dtype != 'object'])

        if time_col and target_col:
            df = raw_df[[time_col, target_col]].dropna()
            df = df.rename(columns={time_col: "Time", target_col: "Target"})

            # Convert time to datetime or int
            try:
                df['Time'] = pd.to_datetime(df['Time'], format='%Y')
                df['Year'] = df['Time'].dt.year
            except:
                df['Year'] = df['Time'].astype(int)

            df = df[['Year', 'Target']]
            df = df[df['Target'] >= 0]
            st.dataframe(df.head())

            X = df[['Year']]
            y = df['Target']

            linear_model = LinearRegression().fit(X, y)
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)
            poly_model = LinearRegression().fit(X_poly, y)

            prophet_df = df.rename(columns={"Year": "ds", "Target": "y"})
            prophet_df["ds"] = pd.to_datetime(prophet_df["ds"], format="%Y")
            model = Prophet()
            model.fit(prophet_df)

            future = model.make_future_dataframe(periods=10, freq='Y')
            forecast = model.predict(future)

            forecast_df = forecast[['ds', 'yhat']]
            forecast_df['Year'] = forecast_df['ds'].dt.year
            forecast_df['Forecast'] = forecast_df['yhat'].round(2)

            last_year = df['Year'].max()
            future_forecast = forecast_df[forecast_df['Year'] > last_year]

            st.subheader("Forecasted Values")
            st.dataframe(future_forecast[['Year', 'Forecast']])

            st.subheader("Forecast Line Chart")
            combined = pd.concat([
                df[['Year', 'Target']].rename(columns={'Target': 'Value'}),
                future_forecast[['Year', 'Forecast']].rename(columns={'Forecast': 'Value'})
            ], ignore_index=True)

            combined['Type'] = ['Actual'] * len(df) + ['Forecast'] * len(future_forecast)
            fig = px.line(combined, x='Year', y='Value', color='Type', markers=True)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Model Accuracy (Historical)")
            df['Linear'] = linear_model.predict(X)
            df['Poly'] = poly_model.predict(X_poly)
            df['Prophet'] = forecast.set_index('ds').loc[prophet_df['ds']]['yhat'].values

            for model in ['Linear', 'Poly', 'Prophet']:
                df[model] = np.clip(df[model], 0, None)

            metrics = {
                'Model': [], 'R2 Score': [], 'MAE': [], 'RMSE': []
            }
            for model in ['Linear', 'Poly', 'Prophet']:
                metrics['Model'].append(model)
                metrics['R2 Score'].append(round(r2_score(df['Target'], df[model]), 3))
                metrics['MAE'].append(round(mean_absolute_error(df['Target'], df[model]), 3))
                metrics['RMSE'].append(round(np.sqrt(mean_squared_error(df['Target'], df[model])), 3))

            st.dataframe(pd.DataFrame(metrics))

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
else:
    st.info("👆 Upload a CSV or Excel file with at least one time column and one numeric column to begin.")
