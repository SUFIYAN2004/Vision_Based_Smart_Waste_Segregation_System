import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import csv
import time
from datetime import datetime

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(
    page_title="AI05 · Smart Waste Segregation",
    layout="wide"
)

# ============================
# CSS (CLEAN UI – NO BIN)
# ============================
st.markdown("""
<style>
body { background:#0d1117; color:#e6edf3; }

.header {
  padding:20px;
  border-radius:20px;
  background:linear-gradient(90deg,#0f2027,#203a43,#2c5364);
}

.card {
  padding:30px;
  border-radius:20px;
  margin-top:20px;
  text-align:center;
  font-size:24px;
  font-weight:bold;
}

.organic {
  background:#002d1f;
  border:4px solid #00ff99;
  color:#00ff99;
}

.recycle {
  background:#001c3d;
  border:4px solid #3b82f6;
  color:#3b82f6;
}

.hazard {
  background:#3d0000;
  border:4px solid red;
  color:#ff4d4d;
  animation: blink 1s infinite;
}

@keyframes blink {
  0% { opacity:1; }
  50% { opacity:0.4; }
  100% { opacity:1; }
}
</style>
""", unsafe_allow_html=True)

# ============================
# HEADER
# ============================
st.markdown("""
<div class="header">
  <h1>♻️ Vision-Based Smart Waste Segregation</h1>
  <p>AI05 · Real-Time Camera System</p>
</div>
""", unsafe_allow_html=True)

# ============================
# MODEL
# ============================
@st.cache_resource
def load_model():
    base = tf.keras.applications.ResNet50V2(
        input_shape=(224,224,3),
        include_top=False,
        weights=None
    )
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224,224,3)),
        tf.keras.layers.Rescaling(1./255),
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.load_weights("waste_pro_final.keras")
    return model

model = load_model()
CLASSES = ['Hazardous', 'Organic', 'Recyclable']

# ============================
# LOGGING
# ============================
def log_disposal(label, conf):
    with open("waste_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            label,
            f"{conf:.2f}"
        ])

# ============================
# SIDEBAR
# ============================
st.sidebar.title("📊 Analytics")
if "points" not in st.session_state:
    st.session_state.points = 100
st.sidebar.metric("Green Points 🌱", st.session_state.points)

# ============================
# CAMERA STATE
# ============================
if "camera_on" not in st.session_state:
    st.session_state.camera_on = False

# ============================
# LAYOUT
# ============================
col1, col2 = st.columns([1.4, 1])

with col1:
    st.subheader("📷 Live Camera")
    toggle = st.toggle("Start / Stop Camera")
    frame_box = st.image([])

with col2:
    st.subheader("🚦 Waste Instruction")
    alert_box = st.empty()
    conf_box = st.empty()

st.session_state.camera_on = toggle

# ============================
# CAMERA LOOP
# ============================
if st.session_state.camera_on:
    cap = cv2.VideoCapture(0)

    while st.session_state.camera_on:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera error")
            break

        h, w, _ = frame.shape
        s = 300
        x1, y1 = (w-s)//2, (h-s)//2
        x2, y2 = x1+s, y1+s
        roi = frame[y1:y2, x1:x2]

        img = cv2.resize(roi, (224,224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)

        preds = model.predict(img, verbose=0)[0]
        idx = np.argmax(preds)
        label = CLASSES[idx]
        conf = preds[idx] * 100

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_box.image(frame)

        if conf > 65:
            conf_box.metric("AI Confidence", f"{conf:.2f}%")

            if label == "Hazardous":
                alert_box.markdown("""
                <div class="card hazard">
                ⚠️ HAZARDOUS WASTE<br>
                <small>Dispose immediately in RED BIN</small>
                </div>
                """, unsafe_allow_html=True)

            elif label == "Organic":
                alert_box.markdown("""
                <div class="card organic">
                🌱 ORGANIC WASTE<br>
                <small>Dispose in GREEN BIN</small>
                </div>
                """, unsafe_allow_html=True)

            else:
                alert_box.markdown("""
                <div class="card recycle">
                ♻️ RECYCLABLE WASTE<br>
                <small>Dispose in BLUE BIN</small>
                </div>
                """, unsafe_allow_html=True)

            if conf > 85:
                st.session_state.points += 2
                log_disposal(label, conf)

        else:
            alert_box.warning("Analyzing material texture…")
            conf_box.empty()

        time.sleep(0.03)

    cap.release()
    cv2.destroyAllWindows()
