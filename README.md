# ♻️ Vision-Based Smart Waste Segregation System

This project implements an AI-powered waste segregation system using **Computer Vision and Deep Learning**.  
It classifies waste into **Organic, Recyclable, and Hazardous** categories using a CNN-based model and provides real-time guidance for proper disposal.

---

## 🚀 Features

- CNN-based waste classification using **ResNet50V2**
- Image upload-based prediction (Colab / Streamlit)
- Real-time camera-based detection (OpenCV)
- Confidence score for predictions
- Clean and interactive UI using Streamlit
- Logging support for analytics (optional)

---

## 🧠 Model Architecture

- Backbone: **ResNet50V2**
- Input shape: `224 × 224 × 3`
- Output classes:
  - Organic
  - Recyclable
  - Hazardous
- Activation: Softmax

---

