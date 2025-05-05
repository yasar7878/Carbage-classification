import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import tempfile
from groq import Groq
import os

# Set your API key (ensure security in real apps)
os.environ['GRDQ_API_KEY'] = "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"
client = Groq(api_key=os.environ['GRDQ_API_KEY'])

# Streamlit Page Setup
st.set_page_config(page_title="♻️ Waste Classifier AI", layout="centered")

# --- ✨ Custom Styling with Gradient Background ---
st.markdown("""
    <style>
    body, .stApp {
        background: linear-gradient(135deg, #c2e9fb 0%, #a1c4fd 100%) !important;
        color: #333333;
        font-family: 'Segoe UI', sans-serif;
    }
    .box {
        background-color: white;
        padding: 20px;
        border-radius: 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
        margin-top: 20px;
    }
    .title {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        color: #0b3954;
    }
    .subtitle {
        font-size: 18px;
        text-align: center;
        color: #333;
        margin-bottom: 40px;
    }
    </style>
""", unsafe_allow_html=True)

# Load Model
model = load_model("inceptionv3_waste.h5")

# Waste classes
categories = ['battery', 'organic', 'brown-glass', 'cardboard', 'clothes',
              'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']

# --- Title Section ---
st.markdown('<div class="title">♻️ Waste Classifier AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Classify your waste and learn how to deal with it smartly.</div>', unsafe_allow_html=True)

# --- Image Uploader ---
uploaded_file = st.file_uploader("Drag and drop an image here", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        image_path = tmp.name

    # Preprocess
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (150, 150))
    image_input = np.expand_dims(image_resized / 255.0, axis=0)

    # Predict
    prediction = model.predict(image_input)
    pred_index = np.argmax(prediction)
    predicted_class = categories[pred_index]

    # Display uploaded image
    # Display uploaded image
    st.image(image_rgb, caption="Uploaded Image", use_container_width=True)


    # Ask Groq for explanation
    with st.spinner("🧠 Analyzing waste and generating tips..."):
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": f"Give this {predicted_class} waste information in bullet points with emojis"}],
        )
        waste_info = response.choices[0].message.content

    # --- Layout: Result Cards ---
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
            <div class="box">
                <h4>🧠 Predicted Waste Category</h4>
                <div style="font-size: 24px; background-color: #FFA94D; color: white; padding: 10px 20px; border-radius: 10px; display: inline-block;">
                    ♻️ {predicted_class.capitalize()}
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="box">
                <h4>📖 What Should You Know?</h4>
                <div style="font-size: 16px;">{waste_info}</div>
            </div>
        """, unsafe_allow_html=True)
