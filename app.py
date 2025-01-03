import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import pairwise_distances

# Load the saved data and cluster centers
cluster_centers = joblib.load('cluster_centers.pkl')  # Replace with your precomputed cluster centers


# Ensure cluster_centers is converted to NumPy array with float type
try:
    cluster_centers = np.array(cluster_centers, dtype=float)
except Exception as e:
    st.write(f"Error converting cluster_centers to NumPy array: {e}")

# Streamlit App Title
st.title("Clustering Prediction App")

st.write("### Input Features")

# Input features
actual_brightness = st.number_input("Actual Brightness", value=0.0)
normal_temperature = st.number_input("Normal Temperature", value=0.0)
relative_humidity = st.number_input("Relative Humidity", value=0.0)
rain = st.number_input("Rain", value=0.0)
wind = st.number_input("Wind", value=0.0)

# Prediction button
if st.button('Predict Cluster'):
    # Create input data
    input_data = pd.DataFrame([[actual_brightness, normal_temperature,
                                relative_humidity, rain, wind]],
                              columns=['Actual Brightness', 'Normal Temperature',
                                       'Relative Humidity', 'Rain', 'Wind'])

    # Ensure input_data is converted to NumPy array and matches dtype
    input_data = np.array(input_data, dtype=float)


    # Calculate distances to cluster centers
    distances = pairwise_distances(input_data, cluster_centers)

    # Predict the closest cluster
    predicted_cluster = np.argmin(distances, axis=1)[0]

    # Map numeric cluster to labels A or B
    cluster_labels = ['A', 'B']
    predicted_label = cluster_labels[predicted_cluster]

    # Display the prediction
    st.write(f"Predicted Cluster: {predicted_label}")
