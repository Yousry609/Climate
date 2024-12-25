import streamlit as st
import pandas as pd
import joblib
import numpy as np


# Load the saved model
model = joblib.load('agglomerative_model.pkl')

# Streamlit App Title
st.title("Clustering Prediction App")

st.write("### Input Features")

# Input features based on provided image
actual_brightness = st.number_input("Actual Brightness", value=0.0)
normal_temperature = st.number_input("Normal Temperature", value=0.0)
relative_humidity = st.number_input("Relative Humidity", value=0.0)
rain = st.number_input("Rain", value=0.0)
wind = st.number_input("Wind", value=0.0)

# Prediction button
if st.button('Predict Cluster'):
    # Create a dataframe for input data
    input_data = pd.DataFrame([[actual_brightness, normal_temperature,
                                relative_humidity, rain, wind]],
                              columns=['Actual Brightness', 'Normal Temperature',
                                       'Relative Humidity', 'Rain', 'Wind'])


    # Since AgglomerativeClustering does not support prediction directly, we append the input to itself
    # Create a duplicate to simulate batch processing (minimum 2 samples required)
    input_batch = np.vstack([input_data, input_data])

    # Fit the model again for clustering
    agg_cluster = model.fit(input_batch)
    labels = agg_cluster.labels_

    # Display the prediction for the first sample
    cluster_labels = ['A', 'B']  # Adjust based on your cluster labeling
    st.write(f"Predicted Cluster: {cluster_labels[labels[0]]}")
