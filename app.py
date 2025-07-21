# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from prophet import Prophet
import io
import datetime

st.set_page_config(page_title="Internet Forecast App", layout="wide")

st.title("📡 Internet Usage Forecasting App")
st.markdown("Predict future internet usage in Nigeria using the ITNETUSERP2NGA dataset.")

# ✅ Upload Excel file
uploaded_file = st.file_uploader("📤 Upload Excel file", type=["xlsx", "csv"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # ✅ Check for required columns
        if 'YEAR' not in df.columns or 'VALUE' not in df.columns:
            st.error("File must contain 'YEAR' and 'VALUE' columns.")
        else:
            # ✅ Rename and preview
            df = df[['YEAR', 'VALUE']].rename(columns={'YEAR': 'Year', 'VALUE': 'Penetration'})
            st.success("✅ File uploaded and read successfully!")
            st.dataframe(df.head())

            # ✅ Clean data
            df = df.dropna()
            df = df[df['Penetration'] > 0]
            X = df[['Year']]
            y = df['Penetration']

            # ✅ Models
            linear_model = LinearRegression().fit(X, y)
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)
            poly_model = LinearRegression().fit(X_poly, y)

            # ✅ Forecast for next 5 years
            future_years = list(range(int(df['Year'].max()) + 1, int(df['Year'].max()) + 6))
            future_df = pd.DataFrame(future_years, columns=['Year'])
            linear_preds = linear_model.predict(future_df)
            poly_preds = poly_model.predict(poly.transform(future_df))

            forecast_df = pd.DataFrame({
                "Year": future_years,
                "Linear Forecast": [round(x, 2) for x in linear_preds],
                "Polynomial Forecast": [round(x, 2) for x in poly_preds]
            })

            # ✅ Prophet Model
            prophet_df = df.rename(columns={"Year": "ds", "Penetration": "y"})
            prophet_df["ds"] = pd.to_datetime(prophet_df["ds"], format="%Y")
            model = Prophet()
            model.fit(prophet_df)
            future = model.make_future_dataframe(periods=5, freq='Y')
            forecast = model.predict(future)
            prophet_forecast = forecast[['ds', 'yhat']].tail(5)
            prophet_forecast['Year'] = prophet_forecast['ds'].dt.year
            prophet_forecast['Prophet Forecast'] = prophet_forecast['yhat'].round(2)

            # ✅ Merge for column chart
            export_df = forecast_df.merge(prophet_forecast[['Year', 'Prophet Forecast']], on='Year')

            # ✅ Display Forecast Table
            st.subheader("📈 Multi-Year Forecast (Next 5 Years)")
            st.dataframe(export_df)

            # ✅ Line Chart
            line_fig = px.line(export_df, x="Year", y=["Linear Forecast", "Polynomial Forecast", "Prophet Forecast"],
                               markers=True, title="📈 Forecast Line Chart")
            for trace in line_fig.data:
                trace.update(text=trace.y, textposition="top center")
            st.plotly_chart(line_fig, use_container_width=True)

            # ✅ Column Chart
            col_fig = px.bar(export_df.melt(id_vars='Year',
                                            value_vars=["Linear Forecast", "Polynomial Forecast", "Prophet Forecast"],
                                            var_name="Model", value_name="Forecast"),
                             x="Year", y="Forecast", color="Model", barmode="group",
                             text_auto='.2s', title="📊 Forecast Comparison Column Chart")
            col_fig.update_traces(textposition='outside')
            st.plotly_chart(col_fig, use_container_width=True)

            # ✅ Download report
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                export_df.to_excel(writer, index=False, sheet_name='Forecast')
            st.download_button("📥 Download Forecast Report (Excel)",
                               data=buffer,
                               file_name="forecast_report_5yrs.xlsx",
                               mime="application/vnd.ms-excel")

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
else:
    st.info("👆 Upload a file with 'YEAR' and 'VALUE' columns to continue.")
