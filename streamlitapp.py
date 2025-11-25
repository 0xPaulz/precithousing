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
    st.markdown("#### Size & Rooms")
    area        = st.number_input("Area (sq ft)", 1650, 20000, 6000, step=100)
    bedrooms    = st.selectbox("Bedrooms", [1,2,3,4,5,6], index=2)
    bathrooms   = st.selectbox("Bathrooms", [1,2,3,4], index=1)
    stories     = st.selectbox("Stories", [1,2,3,4], index=1)
    parking     = st.selectbox("Parking Spaces", [0,1,2,3], index=1)
    mainroad    = st.selectbox("Near Main Road", ["yes", "no"])

with col2:
    st.markdown("#### Luxury & Location")
    airconditioning = st.selectbox("Air Conditioning", ["yes", "no"], index=1)
    prefarea       = st.selectbox("Preferred Neighborhood", ["yes", "no"])
    guestroom      = st.selectbox("Guest Room", ["yes", "no"])
    basement       = st.selectbox("Basement", ["yes", "no"])
    hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])
    furnishingstatus = st.selectbox("Furnishing", ["unfurnished", "semi-furnished", "furnished"], index=2)
    
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
