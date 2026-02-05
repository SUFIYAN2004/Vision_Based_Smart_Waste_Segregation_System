import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import datetime

# 1. PAGE SETUP
st.set_page_config(page_title="AI05: Smart Waste Pro", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0d1117; color: #e6edf3; }
    .stMetric { border: 2px solid #30363d; padding: 20px; border-radius: 15px; background-color: #161b22; }
    </style>
    """, unsafe_allow_html=True)

# 2. PRO-LEVEL ARCHITECTURE (Matching ResNet50V2)
def build_resnet_model():
    # We use ResNet50V2 because it's better at identifying organic textures
    base_model = tf.keras.applications.ResNet50V2(
        input_shape=(224, 224, 3), include_top=False, weights=None
    )
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Rescaling(1./255), # Auto-normalization
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    return model

@st.cache_resource
def load_pro_model():
    model = build_resnet_model()
    # Ensure this matches the filename from your Colab download
    model_path = 'waste_pro_final.keras' 
    try:
        model.load_weights(model_path)
    except:
        model = tf.keras.models.load_model(model_path, compile=False)
    return model

model = load_pro_model()
class_names = ['Hazardous', 'Organic', 'Recyclable']

# 3. SIDEBAR ANALYTICS
st.sidebar.title("📊 City Analytics")
st.sidebar.write("Project: Smart Waste Segregator")
st.sidebar.write(f"Node: Tirupathur_Alpha_01")

# Municipal Tracking
mock_data = pd.DataFrame({
    'Type': ['Organic', 'Recyclable', 'Hazardous'],
    'Daily Total': [45, 32, 8]
})
st.sidebar.table(mock_data)

if 'score' not in st.session_state:
    st.session_state.score = 250

st.sidebar.metric("User Green Points 🏆", f"{st.session_state.score}")

# 4. MAIN INTERFACE
st.title("♻️ Vision-Based Waste Segregator (Pro)")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("📷 Image Classification")
    uploaded_file = st.file_uploader("Drop an image of waste here...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        raw_img = Image.open(uploaded_file).convert("RGB")
        st.image(raw_img, caption="Analyzing Material Texture...", use_container_width=True)
        
        # PRE-PROCESSING
        img_prep = raw_img.resize((224, 224))
        img_array = tf.keras.utils.img_to_array(img_prep)
        img_array = np.expand_dims(img_array, axis=0)

        # PREDICTION
        preds = model.predict(img_array)
        conf_scores = preds[0]
        class_idx = np.argmax(conf_scores)
        label = class_names[class_idx]
        acc = conf_scores[class_idx] * 100

with col2:
    st.subheader("🏁 Segregation Protocol")
    if uploaded_file:
        # Visual Alerts based on Category
        if label == 'Organic':
            st.success(f"### DETECTED: {label}")
            st.write("✅ **Instruction:** Dispose in the **GREEN BIN**.")
            st.info("💡 Pro-tip: This can be converted into nutrient-rich compost.")
        elif label == 'Recyclable':
            st.info(f"### DETECTED: {label}")
            st.write("✅ **Instruction:** Dispose in the **BLUE BIN**.")
            st.info("💡 Pro-tip: Ensure the item is dry and free of food residue.")
        else:
            st.error(f"### DETECTED: {label}")
            st.write("⚠️ **Instruction:** Dispose in the **RED BIN**.")
            st.warning("🚨 Warning: Contains chemicals. Do not mix with other waste.")

        st.metric("AI Confidence", f"{acc:.2f}%")
        
        # Gamification
        if acc > 85:
            st.session_state.score += 15
            st.toast("Points Awarded!", icon='🔥')

        # Probability Distribution Chart
        st.write("---")
        st.write("#### Neural Network Distribution")
        chart_data = pd.DataFrame({
            'Category': class_names,
            'Confidence': conf_scores
        }).set_index('Category')
        st.bar_chart(chart_data)

# 5. FOOTER
st.markdown("---")
st.caption("Architecture: ResNet50V2 Transfer Learning | Dataset: Augmented Waste-Classification-V5")