# ✍️ Signature Verification System 



---

## 📖 Overview

This project implements a **user-trainable signature verification system** using a neural network. Users can upload their own real and forged signatures via a Streamlit interface, train a personalized model, and then verify additional signatures in real time. The system extracts geometric and statistical features from signature images using `scikit-image`, and uses a simple neural network built with TensorFlow/Keras.

---

## 🔥 Live Demo

🌐 [Try the App Live](https://signature-verification-system-e38n.onrender.com)


---

## 🧩 Code Structure

### 🔧 Libraries Used

- `numpy` — numerical arrays
- `scikit-image` — image preprocessing & thresholding
- `scikit-learn` — feature scaling and logistic regression (optional)
- `tensorflow` — deep learning model
- `joblib` — model/scaler saving/loading
- `streamlit` — UI for uploading, training, predicting

---

### 🔍 Feature Extraction Functions

All are defined in `signature_utils.py`:

- **`preproc(img)`**  
  Crops and binarizes the image using Otsu's thresholding.

- **`Ratio(img)`**  
  Computes ratio of white pixels to total pixels.

- **`Centroid(img)`**  
  Computes normalized center of mass of the signature.

- **`EccentricitySolidity(img)`**  
  Measures shape eccentricity and solidity.

- **`SkewKurtosis(img)`**  
  Computes statistical skewness and kurtosis on x and y axes.

- **`extract_features(image_path)`**  
  Aggregates all features above into a single list.

---

## 🧪 Workflow

1. **User Upload**  
   Upload 5 samples (3 genuine, 2 forged) through Streamlit.

2. **Feature Extraction**  
   `extract_features()` computes 9 features from each image.

3. **Model Training**  
   A simple neural network is trained and saved as `.keras`.

4. **Verification**  
   Upload a new image → extract features → scale → predict.

---

## 🖥️ Streamlit UI Structure

- `main.py` — Navigation
- `user_training.py` — Model training and prediction
- `signature_verification.py` — Global verification page (optional)
- `signature_utils.py` — Feature extraction

---

## ⚙️ Usage

### 📦 Requirements

Install dependencies:
```bash
pip install -r requirements.txt
