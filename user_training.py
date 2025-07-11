import streamlit as st
import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
from signature_utils import extract_features
import tensorflow as tf

# Folder for saving user signature images
user_signatures_dir = 'user_signatures/'

# Make sure directory exists
if not os.path.exists(user_signatures_dir):
    os.makedirs(user_signatures_dir)

# Build model
def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train user model
def train_model(user_signatures):
    data = []
    labels = []

    for i, img_path in enumerate(user_signatures):
        features = extract_features(img_path)
        data.append(features)
        labels.append(1 if i < 3 else 0)  # First 3 = genuine, last 2 = forged

    X_train = np.array(data)
    y_train = np.array(labels)

    # Use StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = build_model(X_train_scaled.shape[1])
    model.fit(X_train_scaled, y_train, epochs=10, verbose=0)

    model.save('user_signature_model.keras')  # ✅ using .keras
    joblib.dump(scaler, 'user_signature_scaler.pkl')

    st.success("✅ Model trained and saved successfully!")

# 🔁 This wraps the whole app logic
def user_training_page():
    st.title("🧪 User Signature Training")
    st.info("Upload 5 signatures: First 3 are genuine, last 2 are forged.")

    uploaded_signatures = []
    for i in range(5):
        uploaded_file = st.file_uploader(f"Upload signature {i+1}", type=['png', 'jpg', 'jpeg'], key=f"user_sig_{i}")
        if uploaded_file is not None:
            img_path = os.path.join(user_signatures_dir, f"user_signature_{i+1}.png")
            with open(img_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            uploaded_signatures.append(img_path)

    if len(uploaded_signatures) == 5:
        train_model(uploaded_signatures)

    # Verification section
    st.title("🔍 Verify Signature Against User Model")
    uploaded_verification = st.file_uploader("Upload a signature to verify", type=['png', 'jpg', 'jpeg'], key="verify_sig")

    if uploaded_verification is not None:
        model = load_model('user_signature_model.keras')
        scaler = joblib.load('user_signature_scaler.pkl')

        img_path = os.path.join(user_signatures_dir, 'verify_input.png')
        with open(img_path, 'wb') as f:
            f.write(uploaded_verification.getbuffer())

        features = np.array(extract_features(img_path)).reshape(1, -1)
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0][0]

        if prediction > 0.5:
            st.success("✅ Signature is Genuine!")
        else:
            st.error("❌ Signature is Forged!")
