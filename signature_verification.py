import streamlit as st
from tensorflow.keras.models import load_model
from signature_utils import extract_features
import numpy as np

# Function for signature verification page
def signature_verification_page():
    # Upload signature to verify
    uploaded_file_to_verify = st.file_uploader("Upload a signature to verify", type=['png', 'jpg', 'jpeg'])

    if uploaded_file_to_verify is not None:
        # Load the trained model
        model = load_model('signature_model.keras')

        # Save uploaded file temporarily
        img_path_to_verify = "signature_to_verify.png"
        with open(img_path_to_verify, 'wb') as f:
            f.write(uploaded_file_to_verify.getbuffer())

        # Extract features
        features_to_verify = np.array(extract_features(img_path_to_verify)).reshape(1, -1)

        # ✅ Normalize features same as training
        features_scaled = features_to_verify / np.max(features_to_verify)

        # Predict
        prediction = model.predict(features_scaled)

        if prediction > 0.5:
            st.success("✅ Signature is Genuine!")
        else:
            st.error("❌ Signature is Forged!")
