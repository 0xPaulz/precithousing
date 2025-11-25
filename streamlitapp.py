import streamlit as st
import pandas as pd
import joblib

#loading model
model = joblib.load('best_model.pkl')

st.title("House Price Prediction (Machine Learning Group 6)")
st.markdown("### Enter details below")

#collecting user input
col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (sq ft)", 1000, 20000, 5500)
    bedrooms = st.selectbox("Bedrooms", [1,2,3,4,5,6])
    bathrooms = st.selectbox("Bathrooms", [1,2,3,4])
    stories = st.selectbox("Stories", [1,2,3,4])
    parking = st.selectbox("Parking Spaces", [0,1,2,3])
    mainroad = st.selectbox("Near Main Road", ["yes", "no"])

with col2:
    guestroom = st.selectbox("Guest Room", ["yes", "no"])
    basement = st.selectbox("Basement", ["yes", "no"])
    hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])
    airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])
    prefarea = st.selectbox("Preferred Area", ["yes", "no"])
    furnishingstatus = st.selectbox("Furnishing", ["furnished", "semi-furnished", "unfurnished"])

#prediction button
if st.button("Predict House Price", type="primary"):
    data = {
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
    }
    
    input_data = pd.DataFrame(data)
    input_encoded = pd.get_dummies(input_data, drop_first=True)
    
    training_columns = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking',
                            'mainroad_yes', 'guestroom_yes', 'basement_yes',
                            'hotwaterheating_yes', 'airconditioning_yes',
                            'prefarea_yes', 'furnishingstatus_semi-furnished',
                            'furnishingstatus_unfurnished']
    input_ready = input_encoded.reindex(columns=training_columns, fill_value=0)

    #predicting
    price = model.predict(input_ready)[0]
    
    #printing the price
    st.markdown(f"""
    <h2 style='text-align: center; color: green;'>
        Predicted Price: <b>₦{price:,.0f}</b>
    </h2>
    """, unsafe_allow_html=True)
    st.balloons()