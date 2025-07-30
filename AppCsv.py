import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

model = tf.keras.models.load_model("final_model.keras")

with open("scaler8.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder8.pkl", "rb") as f:
    label_encoder = pickle.load(f)

st.title("Human Activity Recognition")

st.write("Please upload a CSV file containing exactly 20 rows of sensor readings:")

feature_names = [
    "attitude.roll", "attitude.pitch", "attitude.yaw",
    "gravity.x", "gravity.y", "gravity.z",
    "rotationRate.x", "rotationRate.y", "rotationRate.z",
    "userAcceleration.x", "userAcceleration.y", "userAcceleration.z"
]

uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        if df.shape != (20, 12):
            st.error("The CSV file must contain exactly 20 rows and 12 columns.")
        else:
            X = df[feature_names].values
            X_scaled = scaler.transform(X)
            X_seq = X_scaled.reshape(1, 20, 12)

            prediction = model.predict(X_seq)
            predicted_class = np.argmax(prediction, axis=1)[0]
            predicted_label = label_encoder.inverse_transform([predicted_class])[0]

            st.success(f"Predicted Activity: {predicted_label}")
    except Exception as e:
        st.error(f"Error: {e}")