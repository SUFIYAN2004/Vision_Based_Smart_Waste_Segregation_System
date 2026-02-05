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

## ⚠️ Model Availability

The trained model file is not included in this repository due to GitHub file size limitations.

To reproduce or test the model:
- Open the provided Jupyter Notebook (`.ipynb`) in Google Colab
- Run all cells to train or load the model
- Use the prediction cells for image upload or camera testing
