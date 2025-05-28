import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Laptop Price Predictor",
    layout="centered"
)

model = 'laptop-price-prediction/models/best_laptop_price_model.pkl'

@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.error(f"Model file not found: {path}")
        st.stop()
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model(model)

companies = ['Dell', 'Lenovo', 'HP', 'Asus', 'Acer', 'MSI', 'Toshiba', 'Apple', 'Samsung', 'Razer']
typenames = ['Notebook', 'Gaming', 'Ultrabook', '2 in 1 Convertible', 'Workstation', 'Netbook']
rams = sorted([2, 4, 6, 8, 12, 16, 24, 32, 64])
cpu_brands = ['Intel Core i7', 'Intel Core i5', 'Intel Core i3', 'AMD']
ssds = sorted([0, 8, 16, 32, 64, 128, 180, 240, 256, 512, 768, 1024])
hdds = [0, 32, 128, 500, 1024, 2048]
gpu_brands = ['Intel', 'Nvidia', 'AMD']
oss = ['Windows', 'No OS', 'Linux', 'Chrome OS', 'macOS', 'Android']

st.markdown(
    """
    <style>
    .futuristic-title {
        color: #58a6ff;
        text-align: center;
        font-size: 3rem;
        font-weight: 800;
        text-shadow:
            0 0 5px #58a6ff,
            0 0 10px #58a6ff,
            0 0 20px #58a6ff,
            0 0 40px #0ff;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="futuristic-title">Futuristic Laptop Price Predictor</h1>', unsafe_allow_html=True)
st.write("Fill in the laptop specs below to get an estimated price.")

with st.form("laptop_form"):
    col1, col2 = st.columns(2)
    with col1:
        company = st.selectbox("Company", companies)
    with col2:
        typename = st.selectbox("Type", typenames)

    col3, col4 = st.columns(2)
    with col3:
        ram = st.selectbox("RAM (GB)", rams)
    with col4:
        weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=1.8, step=0.1)

    col5, col6 = st.columns(2)
    with col5:
        touchscreen = st.selectbox("Touchscreen", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    with col6:
        ips = st.selectbox("IPS Display", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    col7, col8 = st.columns(2)
    with col7:
        ppi = st.number_input("PPI (Pixels Per Inch)", min_value=50.0, max_value=400.0, value=141.21, step=0.1)
    with col8:
        cpu_brand = st.selectbox("CPU Brand", cpu_brands)

    col9, col10 = st.columns(2)
    with col9:
        ssd = st.selectbox("SSD (GB)", ssds)
    with col10:
        hdd = st.selectbox("HDD (GB)", hdds)

    col11, col12 = st.columns(2)
    with col11:
        gpu_brand = st.selectbox("GPU Brand", gpu_brands)
    with col12:
        os_type = st.selectbox("Operating System", oss)

    submit = st.form_submit_button("Predict Price")

if submit:
    input_data = pd.DataFrame(
        [{
            'Company': company,
            'TypeName': typename,
            'Ram': ram,
            'Weight': weight,
            'TouchScreen': touchscreen,
            'IPS': ips,
            'ppi': ppi,
            'CpuBrand': cpu_brand,
            'SSD': ssd,
            'HDD': hdd,
            'GpuBrand': gpu_brand,
            'os': os_type
        }]
    )

    try:
        log_price = model.predict(input_data)[0]
        predicted_price = np.exp(log_price)
        st.success(f"Estimated Price: ${predicted_price:,.2f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")
