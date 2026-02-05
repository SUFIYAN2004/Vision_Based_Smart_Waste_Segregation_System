import cv2
import numpy as np
import tensorflow as tf

# 1. ARCHITECTURE REBUILD (Must match your ResNet50V2 Training)
def build_pro_model():
    base_model = tf.keras.applications.ResNet50V2(
        input_shape=(224, 224, 3), include_top=False, weights=None
    )
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Rescaling(1./255),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    return model

print("--- Loading ResNet Intelligence Hub ---")
model = build_pro_model()
# Load your latest weights from Colab
model.load_weights('waste_pro_final.keras') 

class_names = ['Hazardous', 'Organic', 'Recyclable']

# 2. VISION CONFIGURATION
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # Define Scanning ROI (Centralized)
    h, w, _ = frame.shape
    box_sz = 320
    x1, y1 = (w - box_sz) // 2, (h - box_sz) // 2
    x2, y2 = x1 + box_sz, y1 + box_sz

    # Extract ROI
    roi = frame[y1:y2, x1:x2]
    
    # Pre-processing for ResNet (Same as training)
    img = cv2.resize(roi, (224, 224))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = tf.keras.utils.img_to_array(img_rgb)
    img_array = np.expand_dims(img_array, axis=0)

    # 3. REAL-TIME INFERENCE
    preds = model.predict(img_array, verbose=0)
    scores = preds[0]
    idx = np.argmax(scores)
    label = class_names[idx]
    conf = scores[idx] * 100

    # 4. TECH-DISPLAY LOGIC
    # Color logic: Organic=Green, Recyclable=Blue, Hazardous=Red
    ui_colors = {'Organic': (0, 255, 0), 'Recyclable': (255, 100, 0), 'Hazardous': (0, 0, 255)}
    
    # Confidence Filter: Don't show labels if the AI is confused (< 65%)
    if conf > 65:
        color = ui_colors.get(label, (255, 255, 255))
        display_msg = f"{label} [{conf:.1f}%]"
    else:
        color = (0, 255, 255) # Yellow
        display_msg = "Analyzing Texture..."

    # Draw UI Elements
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.rectangle(frame, (x1, y1-35), (x2, y1), color, -1)
    cv2.putText(frame, display_msg, (x1+5, y1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Project ID Tag
    cv2.putText(frame, "AI05: SMART WASTE PRO", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow('ResNet-50V2 Live Vision', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()