import streamlit as st
import pandas as pd
import numpy as np
import joblib
import statsmodels.api as sm

st.write(
    """
    # Predict the Number of Poverty in Jawa Barat

    The model used is [multiple linear regression with dummy variables](https://github.com/ayusufalba25/Poverty-in-JawaBarat).
    """
)

# Import model, mean_std, and dummy_columns
model = joblib.load("poverty_model.pkl")
mean_std = joblib.load("mean_std.pkl")
dummy_columns = joblib.load("dummy_columns.pkl")

# Create a function for input parameter
def input_parameter():
    # Input parameter
    st.sidebar.header("Input Parameter")
    select_region = st.sidebar.selectbox(
        "District/city",
        dummy_columns
    )
    pengeluaran_perkapita = st.sidebar.number_input(
        '"Pengeluaran Perkapita" in Thousands of Rupiah',
        min_value = 0.0,
        value = 10203.0,
        step = 0.01
    )
    tpt = st.sidebar.number_input(
        '"Tingkat Pengangguran Terbuka"',
        min_value = 0.0,
        value = 5.07,
        step = 0.01
    )
    apk_pt = st.sidebar.number_input(
        '"Angka Partisipasi Kasar - Perguruan Tinggi"',
        min_value = 0.0,
        value = 19.56,
        step = 0.01
    )
    submit = st.sidebar.button("submit")
    
    input_data = dict(zip(tuple(dummy_columns), (0 for i in range(len(dummy_columns)))))
    input_data[select_region] = 1

    input_data["pengeluaran_perkapita"] = (pengeluaran_perkapita - mean_std["pengeluaran_perkapita"][0]) / mean_std["pengeluaran_perkapita"][1]
    input_data["tingkat_pengangguran_terbuka"] = (tpt - mean_std["tingkat_pengangguran_terbuka"][0]) / mean_std["tingkat_pengangguran_terbuka"][1]
    input_data["apk_perguruan_tinggi"] = (apk_pt - mean_std["apk_perguruan_tinggi"][0]) / mean_std["apk_perguruan_tinggi"][1]

    del input_data["KOTA SUKABUMI"]
    input_data = pd.Series(input_data)

    return select_region, submit, input_data

region, submit, povertyData = input_parameter()

if submit:
    # Prediction
    pred = model.get_prediction(povertyData.to_frame().T).summary_frame()
    
    # Input Data
    st.subheader("Input Data")
    st.write(
        """
        Below is the predictors that required for predicting the number of poverty in Jawa Barat
        based on input parameter.
        """
    )
    povertyData = povertyData.to_frame().rename(columns = {0: "Value"})
    povertyData

    # Output
    st.subheader("Prediction Result")
    st.write("Predicted Number of Poverty (thousands) in " + region +":")
    st.write(pred[["mean", "mean_ci_lower", "mean_ci_upper"]].rename(columns = {
        "mean": "Number of Poverty",
        "mean_ci_lower": "Lower Bound",
        "mean_ci_upper": "Upper Bound"
    }).T.rename(columns = {0: "Predicted Value"}))
else:
    st.write("Please click the **submit** button!")