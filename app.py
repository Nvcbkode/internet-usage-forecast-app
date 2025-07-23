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

            try:
                df['Time'] = pd.to_datetime(df['Time'], format='%Y')
                df['Year'] = df['Time'].dt.year
            except:
                df['Year'] = pd.to_numeric(df['Time'], errors='coerce')

            df = df[['Year', 'Target']].dropna()
            df = df[df['Target'] >= 0]
            st.dataframe(df.head())

            # Filters
            year_range = st.slider("Select year range to view", int(df['Year'].min()), int(df['Year'].max()), (int(df['Year'].min()), int(df['Year'].max())))
            filtered_df = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]

            scenario_col1, scenario_col2 = st.columns(2)
            with scenario_col1:
                slow_growth = st.slider("Slow Growth Multiplier", 0.5, 1.0, 1.0, 0.05)
            with scenario_col2:
                rapid_growth = st.slider("Rapid Growth Multiplier", 1.0, 2.0, 1.0, 0.05)

            chart_type = st.radio("Select chart type", ["Line Chart", "Bar Chart"])

            X = filtered_df[['Year']]
            y = filtered_df['Target']

            linear_model = LinearRegression().fit(X, y)
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)
            poly_model = LinearRegression().fit(X_poly, y)

            prophet_df = filtered_df.rename(columns={"Year": "ds", "Target": "y"})
            prophet_df["ds"] = pd.to_datetime(prophet_df["ds"], format="%Y")
            model = Prophet()
            model.fit(prophet_df)

            future_period = st.slider("Select number of years to forecast", 1, 20, 10)
            future = model.make_future_dataframe(periods=future_period, freq='Y')
            forecast = model.predict(future)

            forecast_df = forecast[['ds', 'yhat']]
            forecast_df['Year'] = forecast_df['ds'].dt.year
            forecast_df['Forecast'] = forecast_df['yhat'].round(2)

            last_year = filtered_df['Year'].max()
            future_forecast = forecast_df[forecast_df['Year'] > last_year]

            # Apply scenario multipliers
            future_forecast['Slow Growth'] = future_forecast['Forecast'] * slow_growth
            future_forecast['Rapid Growth'] = future_forecast['Forecast'] * rapid_growth

            st.subheader("Forecasted Values")
            st.dataframe(future_forecast[['Year', 'Forecast', 'Slow Growth', 'Rapid Growth']])

            st.subheader("Forecast Visualization")
            combined = pd.concat([
                filtered_df[['Year', 'Target']].rename(columns={'Target': 'Value'}).assign(Type='Actual'),
                future_forecast[['Year', 'Forecast']].rename(columns={'Forecast': 'Value'}).assign(Type='Forecast')
            ], ignore_index=True)

            if chart_type == "Line Chart":
                fig = px.line(combined, x='Year', y='Value', color='Type', markers=True)
            else:
                fig = px.bar(combined, x='Year', y='Value', color='Type')
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Model Accuracy (Historical)")
            filtered_df['Linear'] = linear_model.predict(X)
            filtered_df['Poly'] = poly_model.predict(X_poly)
            filtered_df['Prophet'] = forecast.set_index('ds').loc[prophet_df['ds']]['yhat'].values

            for model in ['Linear', 'Poly', 'Prophet']:
                filtered_df[model] = np.clip(filtered_df[model], 0, None)

            metrics = {
                'Model': [], 'R2 Score': [], 'MAE': [], 'RMSE': []
            }
            for model in ['Linear', 'Poly', 'Prophet']:
                metrics['Model'].append(model)
                metrics['R2 Score'].append(round(r2_score(filtered_df['Target'], filtered_df[model]), 3))
                metrics['MAE'].append(round(mean_absolute_error(filtered_df['Target'], filtered_df[model]), 3))
                metrics['RMSE'].append(round(np.sqrt(mean_squared_error(filtered_df['Target'], filtered_df[model])), 3))

            st.dataframe(pd.DataFrame(metrics))

            st.subheader("Seasonal Decomposition")
            try:
                ts = pd.Series(filtered_df['Target'].values, index=pd.date_range(start=f"{filtered_df['Year'].min()}", periods=len(filtered_df), freq='Y'))
                decomposition = seasonal_decompose(ts, model='additive', period=1)
                fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
                decomposition.observed.plot(ax=axes[0], title='Observed')
                decomposition.trend.plot(ax=axes[1], title='Trend')
                decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
                decomposition.resid.plot(ax=axes[3], title='Residual')
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Decomposition skipped: {e}")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
else:
    st.info("👆 Upload a CSV or Excel file with at least one time column and one numeric column to begin.")
