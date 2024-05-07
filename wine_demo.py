import streamlit as st
import numpy as np
import pickle

# Load the model and DataFrame
model = pickle.load(open('wine1.pkl', 'rb'))
df = pickle.load(open('wine_train1.pkl', 'rb'))

st.sidebar.title('Wine Quality Predictor')

# Add some space between the title and the image in the sidebar
st.sidebar.write('')

image = "wines.jpg"
st.sidebar.image(image, use_column_width=True)

alcohol = st.sidebar.selectbox('Alcohol', df['alcohol'].unique())

sulphate = st.sidebar.selectbox('Sulphate', df['sulphates'].unique())

volatile = st.sidebar.selectbox('Volatile Acidity', df['volatile acidity'].unique())

tsd = st.sidebar.selectbox('Sulphur dioxide', df['total sulfur dioxide'].unique())

logo_image = "wine.png"
st.image(logo_image, use_column_width=True)

# About Our Model
st.markdown("---")
st.markdown("## About Our Model")
st.markdown("Our model is based on a machine learning algorithm (Random Forest) trained on a dataset containing various wine characteristics and corresponding quality ratings. It takes into account factors such as alcohol content, sulphate level, volatile acidity, total sulfur dioxide, and other indicators to predict the quality of wine. The model has been trained and fine-tuned to provide accurate predictions based on the input provided by the user.")
st.markdown("---")

if st.sidebar.button('Predict Wine Quality'):
    query = np.array([alcohol, sulphate, volatile, tsd])
    prediction = model.predict([query])[0]

    st.markdown("---")

    # Map prediction to quality categories
    quality_categories = {3: "Worst", 4: "Average", 5: "Above Average", 6: "Good", 7: "Excellent", 8: "Best"}
    quality = quality_categories.get(prediction, "Unknown")
    st.info(f"### Predicted Wine Quality: {quality}")
