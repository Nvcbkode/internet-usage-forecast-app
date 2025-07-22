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

st.title("\U0001F4E1 Internet Usage Forecasting App")
st.markdown("Predict future internet usage in Nigeria using the ITNETUSERP2NGA dataset.")

# ✅ Upload Excel file
uploaded_file = st.file_uploader("\U0001F4E4 Upload Excel file", type=["xlsx", "csv"])

# ✅ Optional filters
show_accuracy = st.sidebar.checkbox("Show Predictive Accuracy Metrics", value=True)
select_models = st.sidebar.multiselect("Select Models to Display", ["Linear Forecast", "Polynomial Forecast", "Prophet Forecast"], default=["Linear Forecast", "Polynomial Forecast", "Prophet Forecast"])
year_range = st.sidebar.slider("Select Year Range for Visualization", 1990, 2030, (2000, 2030))

# Interactive chart filters
chart_type = st.sidebar.radio("Chart Type", ["Line Chart", "Column Chart", "Both"])
show_data_labels = st.sidebar.checkbox("Show Data Labels on Charts", value=True)
show_decomposition = st.sidebar.checkbox("Show Time Series Decomposition", value=False)
smoothing = st.sidebar.checkbox("Apply Moving Average Smoothing", value=False)

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
            prophet_forecast = forecast[['ds', 'yhat']].tail(2030 - df['Year'].max())
            prophet_forecast['Year'] = prophet_forecast['ds'].dt.year.astype(int)
            prophet_forecast['Prophet Forecast'] = np.clip(prophet_forecast['yhat'], 0, None).round(2)

            df['Year'] = df['Year'].astype(int)
            future_years = list(range(df['Year'].max() + 1, 2031))
            future_df = pd.DataFrame(future_years, columns=['Year'])
            linear_preds = np.clip(linear_model.predict(future_df), 0, None)
            poly_preds = np.clip(poly_model.predict(poly.transform(future_df)), 0, None)

            forecast_df = pd.DataFrame({
                "Year": future_years,
                "Linear Forecast": linear_preds.round(2),
                "Polynomial Forecast": poly_preds.round(2)
            })

            export_df = forecast_df.merge(prophet_forecast[['Year', 'Prophet Forecast']], on='Year')

            df['Linear Forecast'] = np.clip(linear_model.predict(df[['Year']]), 0, None).round(2)
            df['Polynomial Forecast'] = np.clip(poly_model.predict(poly.transform(df[['Year']])), 0, None).round(2)
            prophet_hist = forecast[['ds', 'yhat']].head(len(df))
            prophet_hist['Year'] = prophet_hist['ds'].dt.year.astype(int)
            df = df.merge(prophet_hist[['Year', 'yhat']], on='Year')
            df = df.rename(columns={'yhat': 'Prophet Forecast'})
            df['Prophet Forecast'] = np.clip(df['Prophet Forecast'], 0, None).round(2)

            combined_df = pd.concat([df[['Year', 'Penetration', 'Linear Forecast', 'Polynomial Forecast', 'Prophet Forecast']], export_df])
            combined_df = combined_df.sort_values('Year')
            combined_df = combined_df[(combined_df['Year'] >= year_range[0]) & (combined_df['Year'] <= year_range[1])]

            st.subheader("\U0001F4C8 Full Forecast (Filtered by Selected Years)")
            st.dataframe(combined_df)

            if show_accuracy:
                st.subheader("\U0001F4CA Predictive Accuracy Metrics (on historical data)")
                metrics = {
                    "Model": [], "R2 Score": [], "MAE": [], "RMSE": []
                }
                for model_name in select_models:
                    if model_name in df.columns:
                        metrics["Model"].append(model_name)
                        metrics["R2 Score"].append(round(r2_score(df['Penetration'], df[model_name]), 3))
                        metrics["MAE"].append(round(mean_absolute_error(df['Penetration'], df[model_name]), 3))
                        metrics["RMSE"].append(round(np.sqrt(mean_squared_error(df['Penetration'], df[model_name])), 3))
                st.dataframe(pd.DataFrame(metrics))

            if chart_type in ["Line Chart", "Both"]:
                st.subheader("\U0001F4C8 Forecast Line Chart")
                line_fig = px.line(combined_df, x="Year", y=select_models,
                                   markers=True, title="Forecast Line Chart (Filtered Models)")
                if show_data_labels:
                    for trace in line_fig.data:
                        trace.update(mode="lines+markers+text", text=[f"{y:.2f}" for y in trace.y], textposition="top center")
                st.plotly_chart(line_fig, use_container_width=True)

            if chart_type in ["Column Chart", "Both"]:
                st.subheader("\U0001F4CA Forecast Comparison Column Chart")
                melted_df = combined_df.melt(id_vars='Year',
                                             value_vars=select_models,
                                             var_name="Model", value_name="Forecast")
                melted_df = melted_df.sort_values(by=['Year', 'Model'])
                col_fig = px.bar(melted_df, x="Year", y="Forecast", color="Model", barmode="group",
                                 text=melted_df["Forecast"].round(2),
                                 title="Forecast Comparison Column Chart (Filtered Models)")
                if show_data_labels:
                    col_fig.update_traces(textposition='outside')
                st.plotly_chart(col_fig, use_container_width=True)

            if show_decomposition:
                st.subheader("\U0001F4CB Time Series Decomposition")
                ts_df = df.set_index("Year")["Penetration"]
                ts_df.index = pd.to_datetime(ts_df.index, format='%Y')
                result = seasonal_decompose(ts_df, model='additive', period=1)
                st.line_chart(result.trend.dropna(), use_container_width=True)
                st.line_chart(result.seasonal.dropna(), use_container_width=True)
                st.line_chart(result.resid.dropna(), use_container_width=True)

            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                combined_df.to_excel(writer, index=False, sheet_name='Forecast')
            st.download_button("\U0001F4C5 Download Forecast Report (Excel)",
                               data=buffer,
                               file_name="forecast_report_full.xlsx",
                               mime="application/vnd.ms-excel")

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
else:
    st.info("👆 Upload a file with 'YEAR' and 'VALUE' columns to continue.")
