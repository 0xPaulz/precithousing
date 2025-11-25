import streamlit as st
import pandas as pd
import joblib

# page setup
st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("House Price Prediction (Machine Learning Group 6)")
st.markdown("Enter the house details below")

# load the model only once
@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

model = load_model()

# two equal columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Size & Rooms")
    area = st.number_input("area (sq ft)", min_value=1650, max_value=20000, value=7500, step=100)
    bedrooms = st.slider("bedrooms", 1, 6, 3)
    bathrooms = st.slider("bathrooms", 1, 5, 2)
    stories = st.slider("stories", 1, 4, 2)
    parking = st.slider("parking spaces", 0, 3, 1)

with col2:
    st.subheader("Location & Extras")
    mainroad = st.selectbox("near main road?", ("yes", "no"))
    guestroom = st.selectbox("guest room?", ("yes", "no"))
    basement = st.selectbox("basement?", ("yes", "no"))
    hotwaterheating = st.selectbox("hot water heating?", ("yes", "no"))
    airconditioning = st.selectbox("air conditioning?", ("yes", "no"))
    prefarea = st.selectbox("preferred area?", ("yes", "no"))

# full-width row at the bottom
st.markdown("---")
furnishingstatus = st.radio(
    "furnishing status",
    ["furnished", "semi-furnished", "unfurnished"],
    horizontal=True
)

# prediction button
if st.button("Predict House Price", type="primary", use_container_width=True):
    # put everything in the same order as during training
    data = pd.DataFrame({
        'area': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'stories': [stories],
        'mainroad': [mainroad],
        'guestroom': [guestroom],
        'basement': [basement],
        'hotwaterheating': [hotwaterheating],
        'airconditioning': [airconditioning],
        'parking': [parking],
        'prefarea': [prefarea],
        'furnishingstatus': [furnishingstatus]
    })

    price = model.predict(data)[0]

    st.markdown("## ")
    st.success(f"Predicted price: **₦{price:,.0f}**")
    st.balloons()
