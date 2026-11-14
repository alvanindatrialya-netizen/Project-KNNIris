import streamlit as st
import joblib

scaler = joblib.load("scaler.pkl")
model = joblib.load("model_knniris.pkl")

st.title("Klasifikasi Bunga Iris dengan KNN")
st.caption("Model: KNN (K=5), Euclidean Distance, Train:Test = 80:20")

sl = st.number_input("Sepal Length (cm)", 4.0, 8.0, 5.1, step=0.1)
sw = st.number_input("Sepal Width (cm)", 2.0, 4.5, 3.5, step=0.1)
pl = st.number_input("Petal Length (cm)", 1.0, 7.0, 1.4, step=0.1)
pw = st.number_input("Petal Width (cm)", 0.1, 2.5, 0.2, step=0.1)

if st.button("Prediksi"):
    data = scaler.transform([[sl, sw, pl, pw]])
    hasil = model.predict(data)[0]
    st.success(f"Hasil prediksi: {hasil}")

    proba = model.predict_proba(data)[0]
    for nama, p in zip(model.classes_, proba):
        st.write(f"- {nama}: {p:.0%}")