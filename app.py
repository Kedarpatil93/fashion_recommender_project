import streamlit as st
import numpy as np
import pickle
from PIL import Image
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input

# Streamlit config
st.set_page_config(page_title="Fashion Recommender", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🛍️ Fashion Recommender System</h1>", unsafe_allow_html=True)
st.markdown("Upload an image of a clothing item to get similar recommendations:")

# Load filenames and feature vectors from disk (S3 URLs)
@st.cache_data
def load_data():
    feature_list = np.array(pickle.load(open("all_embeddings_efficient_max.pkl", "rb")))
    filenames = pickle.load(open("filenames.pkl", "rb"))  # list of S3 URLs
    return feature_list, filenames

# Load pretrained EfficientNetB3 once
@st.cache_resource
def load_model():
    return EfficientNetB3(weights="imagenet", include_top=False, pooling="max")

# Cache Nearest Neighbors model
@st.cache_resource
def load_neighbors(features):
    neighbors = NearestNeighbors(n_neighbors=5, algorithm="brute", metric="euclidean")
    neighbors.fit(features)
    return neighbors

# Extract image features from uploaded file (in-memory)
def extract_features(uploaded_file, model):
    img = Image.open(uploaded_file).resize((224, 224)).convert("RGB")
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    return result / norm(result)

# Main logic
feature_list, filenames = load_data()
model = load_model()
neighbors = load_neighbors(feature_list)

uploaded_file = st.file_uploader("")

if uploaded_file is not None:
    st.markdown("##### Uploaded Image:")
    st.image(uploaded_file, width=150)

    features = extract_features(uploaded_file, model)
    distances, indices = neighbors.kneighbors([features])

    st.markdown("##### 🔍 Recommended Similar Items:")
    cols = st.columns(5)
    for i, col in enumerate(cols):
        image_url = filenames[indices[0][i]]  # S3 public URL
        col.image(image_url, width=125)
