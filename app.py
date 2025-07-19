# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import io

st.set_page_config(page_title="Internet Forecast App", layout="centered")
st.title("📡 Internet Usage Forecasting App")
st.markdown("Upload the Excel file with `YEAR` and `VALUE` columns to forecast future internet penetration.")

# ✅ Upload XLSX file
file = st.file_uploader("Upload Excel (.xlsx) file", type=["xlsx"])

if file:
    try:
        df = pd.read_excel(file)
        df.columns = df.columns.str.strip().str.upper()

        st.write("🧾 Columns detected:", df.columns.tolist())  # Debug output

        if 'YEAR' not in df.columns or 'VALUE' not in df.columns:
            st.error("❌ Excel must contain columns named 'YEAR' and 'VALUE'.")
        else:
            df = df[['YEAR', 'VALUE']].rename(columns={'YEAR': 'Year', 'VALUE': 'Penetration'})
            st.success("✅ Data loaded successfully!")
            st.dataframe(df.head())

            # ✅ Rename and clean
            df = df[['YEAR', 'VALUE']].rename(columns={'YEAR': 'Year', 'VALUE': 'Penetration'})
            df = df.dropna()
            df = df[df['Penetration'] > 0]
            st.dataframe(df.head())

            X = df[['Year']]
            y = df['Penetration']

            # ✅ Linear Model
            linear_model = LinearRegression().fit(X, y)
            linear_r2 = linear_model.score(X, y)

            # ✅ Polynomial Model
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)
            poly_model = LinearRegression().fit(X_poly, y)
            poly_r2 = r2_score(y, poly_model.predict(X_poly))

            # ✅ Year selector
            pred_year = st.slider("📅 Select a year to predict", int(X.min()), 2030, 2026)
            linear_pred = linear_model.predict([[pred_year]])[0]
            poly_pred = poly_model.predict(poly.transform([[pred_year]]))[0]

            st.subheader(f"🔮 Forecast for {pred_year}")
            st.write(f"📈 Linear Prediction: **{int(linear_pred):,}** users")
            st.write(f"🧮 Polynomial Prediction: **{int(poly_pred):,}** users")
            st.caption(f"Linear R²: {linear_r2:.3f}, Polynomial R²: {poly_r2:.3f}")

            # ✅ Plot
            fig, ax = plt.subplots()
            ax.scatter(X, y, color='black', label='Actual Data')
            ax.plot(X, linear_model.predict(X), color='blue', label='Linear Model')
            ax.plot(X, poly_model.predict(X_poly), color='green', linestyle='--', label='Polynomial Model')
            ax.set_xlabel("Year")
            ax.set_ylabel("Internet Penetration")
            ax.set_title("Internet Usage Forecast")
            ax.legend()
            st.pyplot(fig)

            # ✅ Report
            report_df = pd.DataFrame({
                "Model": ["Linear", "Polynomial"],
                "Year": [pred_year, pred_year],
                "Predicted Penetration": [int(linear_pred), int(poly_pred)]
            })

            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                report_df.to_excel(writer, index=False, sheet_name='Forecast')
            st.download_button("📥 Download Forecast Report (Excel)",
                               data=buffer,
                               file_name="forecast_report.xlsx",
                               mime="application/vnd.ms-excel")

    except Exception as e:
        st.error(f"⚠️ Error reading Excel file: {e}")
else:
    st.info("📂 Please upload your Excel file (.xlsx) to continue.")
