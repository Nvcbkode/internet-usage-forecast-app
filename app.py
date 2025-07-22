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

st.set_page_config(page_title="Internet Forecast App", layout="wide")

st.title("\U0001F4E1 Internet Penetration in Nigeria Forecast")
st.markdown("Predict future internet usage in Nigeria using the ITNETUSERP2NGA dataset.")

uploaded_file = st.file_uploader("\U0001F4E4 Upload Excel file", type=["xlsx", "csv"])

show_accuracy = st.sidebar.checkbox("Show Predictive Accuracy Metrics", value=True)
select_models = st.sidebar.multiselect("Select Models to Display", ["Linear Forecast", "Polynomial Forecast", "Prophet Forecast", "Prophet Fitted", "Scenario: Slow Growth", "Scenario: Rapid Growth"], default=["Linear Forecast", "Polynomial Forecast", "Prophet Forecast"])
year_range = st.sidebar.slider("Select Year Range for Visualization", 1990, 2030, (2000, 2030))

decade_toggle = st.sidebar.selectbox("Decade Filter (Interactive Toggle)", options=["All", "1990s", "2000s", "2010s", "2020s", "2030s"])

chart_type = st.sidebar.radio("Chart Type", ["Line Chart", "Column Chart", "Both"])
show_data_labels = st.sidebar.checkbox("Show Data Labels on Charts", value=True)
show_decomposition = st.sidebar.checkbox("Show Time Series Decomposition", value=False)
smoothing = st.sidebar.checkbox("Apply Moving Average Smoothing", value=False)
color_toggle = st.sidebar.checkbox("Color Code Accuracy Table", value=True)
yoy_toggle = st.sidebar.checkbox("Show YoY % Change in Forecast Table", value=True)
include_historical_toggle = st.sidebar.checkbox("Include Historical Data in Charts", value=False)

slow_growth_factor = st.sidebar.slider("Slow Growth Multiplier", 0.7, 1.0, 0.9, 0.01)
rapid_growth_factor = st.sidebar.slider("Rapid Growth Multiplier", 1.0, 1.5, 1.1, 0.01)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        if 'YEAR' not in df.columns or 'VALUE' not in df.columns:
            st.error("File must contain 'YEAR' and 'VALUE' columns.")
        else:
            df = df[['YEAR', 'VALUE']].rename(columns={'YEAR': 'Year', 'VALUE': 'Penetration'})
            st.success("✅ File uploaded and read successfully!")
            st.dataframe(df.head())

            df = df.dropna()
            df = df[df['Penetration'] >= 0]
            df['Year'] = df['Year'].astype(int)
            X = df[['Year']]
            y = df['Penetration']

            if smoothing:
                df['Penetration'] = df['Penetration'].rolling(window=3, center=True).mean().fillna(method='bfill').fillna(method='ffill')

            linear_model = LinearRegression().fit(X, y)
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)
            poly_model = LinearRegression().fit(X_poly, y)

            prophet_df = df.rename(columns={"Year": "ds", "Penetration": "y"})
            prophet_df["ds"] = pd.to_datetime(prophet_df["ds"], format="%Y")
            model = Prophet()
            model.fit(prophet_df)
            future = model.make_future_dataframe(periods=2030 - df['Year'].max(), freq='Y')
            forecast = model.predict(future)

            prophet_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(2030 - df['Year'].max())
            prophet_forecast['Year'] = prophet_forecast['ds'].dt.year.astype(int)
            prophet_forecast['Prophet Forecast'] = np.clip(prophet_forecast['yhat'], 0, None).round(2)
            prophet_forecast['Lower Bound'] = np.clip(prophet_forecast['yhat_lower'], 0, None).round(2)
            prophet_forecast['Upper Bound'] = np.clip(prophet_forecast['yhat_upper'], 0, None).round(2)

            prophet_forecast['Scenario: Slow Growth'] = (prophet_forecast['Prophet Forecast'] * slow_growth_factor).round(2)
            prophet_forecast['Scenario: Rapid Growth'] = (prophet_forecast['Prophet Forecast'] * rapid_growth_factor).round(2)

            prophet_hist = forecast[['ds', 'yhat']].head(len(df))
            prophet_hist['Year'] = prophet_hist['ds'].dt.year.astype(int)
            prophet_hist['Prophet Fitted'] = np.clip(prophet_hist['yhat'], 0, None).round(2)
            df = df.merge(prophet_hist[['Year', 'Prophet Fitted']], on='Year')

            df['Linear Forecast'] = np.clip(linear_model.predict(X), 0, None).round(2)
            df['Polynomial Forecast'] = np.clip(poly_model.predict(X_poly), 0, None).round(2)

            future_years = list(range(df['Year'].max() + 1, 2031))
            future_df = pd.DataFrame(future_years, columns=['Year'])
            linear_preds = np.clip(linear_model.predict(future_df), 0, None)
            poly_preds = np.clip(poly_model.predict(poly.transform(future_df)), 0, None)

            forecast_df = pd.DataFrame({
                "Year": future_years,
                "Linear Forecast": linear_preds.round(2),
                "Polynomial Forecast": poly_preds.round(2)
            })

            export_df = forecast_df.merge(prophet_forecast[['Year', 'Prophet Forecast', 'Lower Bound', 'Upper Bound', 'Scenario: Slow Growth', 'Scenario: Rapid Growth']], on='Year')
            combined_df = export_df.copy()

            if yoy_toggle:
                for model in ['Linear Forecast', 'Polynomial Forecast', 'Prophet Forecast', 'Scenario: Slow Growth', 'Scenario: Rapid Growth']:
                    if model in combined_df.columns:
                        combined_df[f"{model} YoY%"] = combined_df[model].pct_change().fillna(0).round(4) * 100

            st.subheader("\U0001F4C8 Forecast for Future Years Only")
            st.dataframe(combined_df)

            if show_accuracy:
                st.subheader("\U0001F4CA Predictive Accuracy Metrics (on historical data)")
                metrics = {"Model": [], "R2 Score": [], "MAE": [], "RMSE": []}
                for model_name in select_models:
                    if model_name in df.columns:
                        metrics["Model"].append(model_name)
                        metrics["R2 Score"].append(round(r2_score(df['Penetration'], df[model_name]), 3))
                        metrics["MAE"].append(round(mean_absolute_error(df['Penetration'], df[model_name]), 3))
                        metrics["RMSE"].append(round(np.sqrt(mean_squared_error(df['Penetration'], df[model_name])), 3))
                metrics_df = pd.DataFrame(metrics)
                if color_toggle:
                    styled_metrics = metrics_df.style.background_gradient(axis=0, cmap='coolwarm')
                    st.dataframe(styled_metrics)
                else:
                    st.dataframe(metrics_df)

            if include_historical_toggle:
                chart_data = pd.concat([df[['Year', 'Penetration'] + [col for col in df.columns if col in select_models]], combined_df], ignore_index=True)
            else:
                chart_data = combined_df

            chart_data = chart_data[chart_data['Year'].between(*year_range)]

            if decade_toggle != "All":
                decade_map = {
                    "1990s": range(1990, 2000),
                    "2000s": range(2000, 2010),
                    "2010s": range(2010, 2020),
                    "2020s": range(2020, 2030),
                    "2030s": range(2030, 2040)
                }
                if decade_toggle in decade_map:
                    chart_data = chart_data[chart_data['Year'].isin(decade_map[decade_toggle])]

            if chart_type in ["Line Chart", "Both"]:
                st.subheader("\U0001F4C8 Forecast Line Chart")
                line_fig = px.line(chart_data, x="Year", y=select_models,
                                   markers=True, title="Forecast Line Chart (Filtered Models)",
                                   hover_data={col: ':.2f' for col in select_models})
                if 'Lower Bound' in chart_data.columns and 'Upper Bound' in chart_data.columns:
                    line_fig.add_scatter(x=chart_data["Year"], y=chart_data["Lower Bound"],
                                         mode='lines', name='Lower Bound',
                                         line=dict(color='rgba(255,0,0,0.4)', dash='dot'))
                    line_fig.add_scatter(x=chart_data["Year"], y=chart_data["Upper Bound"],
                                         mode='lines', name='Upper Bound',
                                         line=dict(color='rgba(0,0,255,0.4)', dash='dot'))
                if show_data_labels:
                    for trace in line_fig.data:
                        trace.update(mode="lines+markers+text", text=[f"{y:.2f}" for y in trace.y], textposition="top center")
                st.plotly_chart(line_fig, use_container_width=True)

            if chart_type in ["Column Chart", "Both"]:
                st.subheader("\U0001F4CA Forecast Comparison Column Chart")
                melted_df = chart_data.melt(id_vars='Year', value_vars=select_models, var_name="Model", value_name="Forecast")
                melted_df = melted_df.sort_values(by=['Year', 'Model'])
                col_fig = px.bar(melted_df, x="Year", y="Forecast", color="Model", barmode="group",
                                 text=melted_df["Forecast"].round(2),
                                 hover_name="Model",
                                 hover_data={"Forecast":":.2f", "Year":True},
                                 title="Forecast Comparison Column Chart (Filtered Models)")
                if show_data_labels:
                    col_fig.update_traces(textposition='outside')
                st.plotly_chart(col_fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
else:
    st.info("👆 Upload a file with 'YEAR' and 'VALUE' columns to continue.")
