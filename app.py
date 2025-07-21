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
            linear_preds = linear_model.predict(pd.DataFrame(future_years))
            poly_preds = poly_model.predict(poly.transform(pd.DataFrame(future_years)))

            forecast_df = pd.DataFrame({
                "Year": future_years,
                "Linear Forecast": linear_preds,
                "Polynomial Forecast": poly_preds
            })

            # ✅ Prophet Model (Time Series Forecasting)
            prophet_df = df.rename(columns={"Year": "ds", "Penetration": "y"})
            prophet_df["ds"] = pd.to_datetime(prophet_df["ds"], format="%Y")
            model = Prophet()
            model.fit(prophet_df)
            future = model.make_future_dataframe(periods=5, freq='Y')
            forecast = model.predict(future)
            prophet_forecast = forecast[['ds', 'yhat']].tail(5)

            # ✅ Display all forecasts
            st.subheader("📈 Multi-Year Forecast (Next 5 Years)")
            st.write(forecast_df)
            st.write("🔮 Prophet Forecast:")
            st.dataframe(prophet_forecast.rename(columns={"ds": "Year", "yhat": "Prophet Forecast"}))

            # ✅ Plot with Plotly (Line Chart)
            fig_plotly = px.line(df, x='Year', y='Penetration', title='Actual Internet Usage')
            for model_name, preds in zip(['Linear', 'Polynomial'], [linear_preds, poly_preds]):
                fig_plotly.add_scatter(x=future_years, y=preds, mode='lines+markers+text', name=model_name,
                                       text=[f"{p:.1f}" for p in preds], textposition="top center")
            fig_plotly.add_scatter(x=prophet_forecast['ds'].dt.year, y=prophet_forecast['yhat'],
                                   mode='lines+markers+text', name='Prophet',
                                   text=[f"{p:.1f}" for p in prophet_forecast['yhat']], textposition="top center")
            st.plotly_chart(fig_plotly, use_container_width=True)

            # ✅ Plot Column Chart with Data Labels
            bar_fig = px.bar(
                forecast_df,
                x='Year',
                y=['Linear Forecast', 'Polynomial Forecast'],
                barmode='group',
                title='Forecast Comparison (Bar Chart)',
                text_auto=True
            )
            bar_fig.add_bar(
                x=prophet_forecast['ds'].dt.year,
                y=prophet_forecast['yhat'],
                name='Prophet Forecast',
                text=[f"{p:.1f}" for p in prophet_forecast['yhat']],
                textposition="outside"
            )
            st.plotly_chart(bar_fig, use_container_width=True)

            # ✅ Download report
            export_df = pd.DataFrame({
                "Year": future_years,
                "Linear Forecast": linear_preds,
                "Polynomial Forecast": poly_preds,
                "Prophet Forecast": list(prophet_forecast['yhat'])
            })

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
