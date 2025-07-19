# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import io
import os

st.set_page_config(page_title="Internet Forecast App", layout="centered")

st.title("📡 Internet Usage Forecasting App")
st.markdown("Predict future internet usage in Nigeria using the ITNETUSERP2NGA dataset.")

# ✅ Load Excel file with proper error handling
file_path = "C:/Users/PC-2/Documents/School/Project 3/ITNETUSERP2NGA.xlsx"

if not os.path.exists(file_path):
    st.error(f"❌ File not found at:\n`{file_path}`\nPlease verify the path.")
else:
    try:
        df = pd.read_excel(file_path)
        df.columns = df.columns.str.strip().str.upper()

        if 'YEAR' in df.columns and 'VALUE' in df.columns:
            df = df[['YEAR', 'VALUE']].rename(columns={'YEAR': 'Year', 'VALUE': 'Penetration'})
        else:
            st.error("Excel file must contain 'YEAR' and 'VALUE' columns.")

        st.success("✅ ITNETUSERP2NGA dataset loaded successfully!")
        st.dataframe(df.head())

        # ✅ Clean data
        df = df.dropna()
        df = df[df['Penetration'] > 0]

        X = df[['Year']]
        y = df['Penetration']

        # ✅ Models
        linear_model = LinearRegression().fit(X, y)
        linear_r2 = linear_model.score(X, y)

        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        poly_model = LinearRegression().fit(X_poly, y)
        poly_r2 = r2_score(y, poly_model.predict(X_poly))

        # ✅ Year selection slider
        pred_year = st.slider("📅 Select a year to predict", int(X.min()), 2030, 2026)
        linear_pred = linear_model.predict([[pred_year]])[0]
        poly_pred = poly_model.predict(poly.transform([[pred_year]]))[0]

        # ✅ Display predictions
        st.subheader(f"🔮 Forecast for {pred_year}")
        st.write(f"📈 Linear Model Prediction: **{int(linear_pred):,}** users")
        st.write(f"🧮 Polynomial Model Prediction: **{int(poly_pred):,}** users")
        st.caption(f"Linear R²: {linear_r2:.3f}, Polynomial R²: {poly_r2:.3f}")

        # ✅ Plot results
        fig, ax = plt.subplots()
        ax.scatter(X, y, color='black', label='Actual Data')
        ax.plot(X, linear_model.predict(X), color='blue', label='Linear Model')
        ax.plot(X, poly_model.predict(X_poly), color='green', linestyle='--', label='Polynomial Model')
        ax.set_xlabel("Year")
        ax.set_ylabel("Internet Penetration")
        ax.set_title("Internet Usage Forecast")
        ax.legend()
        st.pyplot(fig)

        # ✅ Download report
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
        st.error(f"⚠️ Failed to read Excel file: {e}")
