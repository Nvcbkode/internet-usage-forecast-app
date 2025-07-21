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
uploaded_file = st.file_uploader("📄 Upload Excel file", type=["xlsx", "csv"])

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
            df['Year'] = df['Year'].astype(int)
            X = df[['Year']]
            y = df['Penetration']

            # ✅ Models
            linear_model = LinearRegression().fit(X, y)
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)
            poly_model = LinearRegression().fit(X_poly, y)

            # ✅ Forecast for next 5 years
            future_years = list(range(df['Year'].max() + 1, df['Year'].max() + 6))
            future_df = pd.DataFrame(future_years, columns=['Year'])
            linear_preds = linear_model.predict(future_df)
            poly_preds = poly_model.predict(poly.transform(future_df))

            forecast_df = pd.DataFrame({
                "Year": future_years,
                "Linear Forecast": linear_preds,
                "Polynomial Forecast": poly_preds
            })

            # ✅ Clip negative values
            forecast_df = forecast_df.clip(lower=0)

            # ✅ Prophet Model
            prophet_df = df.rename(columns={"Year": "ds", "Penetration": "y"})
            prophet_df["ds"] = pd.to_datetime(prophet_df["ds"], format="%Y")
            model = Prophet()
            model.fit(prophet_df)
            future = model.make_future_dataframe(periods=5, freq='Y')
            forecast = model.predict(future)
            prophet_forecast = forecast[['ds', 'yhat']].tail(5)
            prophet_forecast['Year'] = prophet_forecast['ds'].dt.year.astype(int)
            prophet_forecast['Prophet Forecast'] = prophet_forecast['yhat']
            prophet_forecast = prophet_forecast.clip(lower=0)

            # ✅ Merge for column chart
            export_df = forecast_df.merge(prophet_forecast[['Year', 'Prophet Forecast']], on='Year')

            # ✅ Add past years comparison
            df['Linear Forecast'] = linear_model.predict(df[['Year']])
            df['Polynomial Forecast'] = poly_model.predict(poly.transform(df[['Year']]))
            prophet_hist = forecast[['ds', 'yhat']].head(len(df))
            prophet_hist['Year'] = prophet_hist['ds'].dt.year.astype(int)
            df = df.merge(prophet_hist[['Year', 'yhat']], on='Year')
            df = df.rename(columns={'yhat': 'Prophet Forecast'})

            # ✅ Clip all negative values to 0
            df[['Linear Forecast', 'Polynomial Forecast', 'Prophet Forecast']] = df[['Linear Forecast', 'Polynomial Forecast', 'Prophet Forecast']].clip(lower=0)

            combined_df = pd.concat([df[['Year', 'Linear Forecast', 'Polynomial Forecast', 'Prophet Forecast']], export_df])
            combined_df = combined_df[combined_df['Year'] >= df['Year'].min()]  # Exclude years earlier than dataset
            combined_df = combined_df.sort_values('Year')

            # Round to 2 decimal places
            combined_df = combined_df.round(2)

            # ✅ Display Full Forecast Table
            st.subheader("📊 Full Forecast (Including Past & Future Years)")
            st.dataframe(combined_df)

            # ✅ Line Chart
            line_fig = px.line(combined_df, x="Year", y=["Linear Forecast", "Polynomial Forecast", "Prophet Forecast"],
                               markers=True, title="📈 Forecast Line Chart (All Years)")
            for trace in line_fig.data:
                trace.update(mode="lines+markers+text", text=[f"{y:.2f}" for y in trace.y], textposition="top center")
            st.plotly_chart(line_fig, use_container_width=True)

            # ✅ Column Chart
            melted_df = combined_df.melt(id_vars='Year',
                                         value_vars=["Linear Forecast", "Polynomial Forecast", "Prophet Forecast"],
                                         var_name="Model", value_name="Forecast")
            col_fig = px.bar(melted_df,
                             x="Year", y="Forecast", color="Model", barmode="group",
                             text=melted_df["Forecast"].round(2), title="📊 Forecast Comparison Column Chart (All Years)")
            col_fig.update_traces(textposition='outside')
            st.plotly_chart(col_fig, use_container_width=True)

            # ✅ Download report
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                combined_df.to_excel(writer, index=False, sheet_name='Forecast')
            st.download_button("📅 Download Forecast Report (Excel)",
                               data=buffer,
                               file_name="forecast_report_full.xlsx",
                               mime="application/vnd.ms-excel")

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
else:
    st.info("👆 Upload a file with 'YEAR' and 'VALUE' columns to continue.")
