import streamlit as st
import os

# Title of the App
st.title("Signature Verification System")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Signature Verification", "User Training"))

# If the user selects 'Signature Verification'
if page == "Signature Verification":
    st.write("## Signature Verification")
    # Call your signature verification page
    import signature_verification
    signature_verification.signature_verification_page()

# If the user selects 'User Training'
elif page == "User Training":
    st.write("## User Training")
    # Call your user training page
    import user_training
    user_training.user_training_page()
