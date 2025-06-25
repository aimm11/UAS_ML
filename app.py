import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.title("Aplikasi Prediksi Regresi Linier")

uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("📊 Data Awal:", data.head())

    # Ganti sesuai nama kolom di dataset kamu
    features = st.multiselect("Pilih fitur (X):", options=data.columns)
    target = st.selectbox("Pilih target (y):", options=data.columns)

    if features and target:
        X = data[features]
        y = data[target]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluasi
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        st.subheader("📈 Hasil Evaluasi Model:")
        st.write(f"**MAE:** {mae:.2f}")
        st.write(f"**MSE:** {mse:.2f}")
        st.write(f"**RMSE:** {rmse:.2f}")
        st.write(f"**MAPE:** {mape:.2f}%")

        st.subheader("🔍 Prediksi vs Real:")
        compare_df = pd.DataFrame({"Aktual": y_test.values, "Prediksi": y_pred})
        st.write(compare_df.head())
