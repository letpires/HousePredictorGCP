import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from google.cloud import storage
from io import BytesIO
from google.oauth2 import service_account

# Path to the authentication JSON
credentials = service_account.Credentials.from_service_account_file(
    "/Users/leticiapires/Desktop/HousePredictorGCP/talk-gdg-61fa935c78f1.json"
)

# Create storage client
client = storage.Client(credentials=credentials)

bucket = client.bucket("talk-gdg")
blob = bucket.blob("models/model_pipeline.joblib")

blob.download_to_filename("model_pipeline.joblib")
print("File downloaded successfully!")

# Load the pipeline (model and scaler)
pipeline = load('model_pipeline.joblib')
model = pipeline['model']
scaler = pipeline['scaler']
y_transform = pipeline['y_transform']  # Information about the target variable transformation

# Stylized app title
st.markdown(
    """
    <h1 style='color: #4A90E2; font-size: 42px;'>🏠 House Price Prediction</h1>
    """,
    unsafe_allow_html=True
)

# Centered form title
st.markdown("### Enter the property data")

# Hidden description using st.expander
with st.expander("🔍 **Variable Description**"):
    st.markdown("""
    - **Bedrooms**: Number of bedrooms available in each house.  
    - **Bathrooms**: Number of bathrooms available in each house.  
    - **Living Area**: Size of the living area in square feet.  
    - **Lot Area**: Total lot area in square feet, including the area occupied by the house and external spaces like garden or yard.  
    - **Floors**: Number of floors available in the house.  
    - **Waterfront**: Indicator if the house is located by a lake or beach (0: no waterfront, 1: waterfront).  
    - **View**: Visual evaluation of the property based on the available view (scale from 0 to 4).  
    - **Grade**: Quality rating of the design and construction of the property, on a scale from 1 to 12.  
    - **Renovated**: 1 if renovated, 0 if not.  
    - **Basement**: 1 if there is a basement, 0 if not.  
    - **Condition**: General condition of the house: 'Poor', 'Average', 'Fair', 'Good', 'Excellent' (scale from 1 to 5).  
    """)

# Creating the form with inputs
with st.form(key='form_data'):
    bedrooms = st.number_input('Number of bedrooms', min_value=1, max_value=10, value=3)
    bathrooms = st.number_input('Number of bathrooms', min_value=1, max_value=5, value=2)
    sqft_living = st.number_input('Living area (in sq ft)', min_value=500, max_value=10000, value=1500)
    sqft_lot = st.number_input('Lot size (in sq ft)', min_value=500, max_value=50000, value=5000)
    floors = st.number_input('Number of floors', min_value=1, max_value=3, value=1)
    waterfront = st.selectbox('Has waterfront view?', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    view = st.slider('View (scale from 0 to 4)', min_value=0, max_value=4, value=0)
    grade = st.slider('Property grade (1 to 12)', min_value=1, max_value=12, value=7)
    renovated = st.selectbox('Renovated?', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    basement = st.number_input('Basement size (in sq ft)', min_value=0, max_value=5000, value=0)
    condition = st.slider('Property condition (1 to 5)', min_value=1, max_value=5, value=3)

    # Submit button
    submit_button = st.form_submit_button(label='Predict Price')

# Processing the prediction
if submit_button:
    # Organize input data
    new_house = pd.DataFrame({
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'sqft_living': [sqft_living],
        'sqft_lot': [sqft_lot],
        'floors': [floors],
        'waterfront': [waterfront],
        'view': [view],
        'grade': [grade],
        'renovated': [renovated],
        'basement': [basement],
        'condition': [condition]
    })

    # Normalize input data
    new_house_scaled = scaler.transform(new_house)

    # Make the prediction
    y_pred_log = model.predict(new_house_scaled)

    # Reverse the log to the original scale, if necessary
    if y_transform == 'log':
        y_pred = np.exp(y_pred_log)
    else:
        y_pred = y_pred_log

    # Using markdown to display the result with style
    st.markdown(
        f"""
        <div style='text-align: center; margin-top: 20px;'>
            <h2 style='color: #2E8B57;'>🏡 The estimated price of the property is:</h2>
            <h1 style='color: #FF5733;'>${y_pred[0]:,.2f}</h1>
        </div>
        """,
        unsafe_allow_html=True
    ) 