import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import xgboost as xgb
from PIL import Image

# Load models
@st.cache_resource
def load_models():
    cnn = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet', pooling='avg')  # type: ignore
    cnn.trainable = False
    
    xgb_mdl = xgb.XGBClassifier()
    xgb_mdl.load_model("models/xgb_currency_model.json")
    
    return cnn, xgb_mdl

cnn_model, xgb_model = load_models()

# UI
st.title("💵 Currency Detector")
uploaded_file = st.file_uploader("Upload image", type=["jpg", "png"])

if uploaded_file:
    col1, col2 = st.columns(2)
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
    
    img = np.array(image)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    img_resized = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
    features = cnn_model(np.expand_dims(img_resized, 0), training=False).numpy()
    prediction = xgb_model.predict(features)[0]
    
    with col2:
        if prediction == 1:
            st.success("✅ REAL")
            color = (0, 255, 0)
        else:
            st.error("❌ FAKE")
            color = (255, 0, 0)
        
        result = img.copy()
        cv2.rectangle(result, (10, 10), (result.shape[1]-10, result.shape[0]-10), color, 5)
        st.image(result, use_column_width=True)