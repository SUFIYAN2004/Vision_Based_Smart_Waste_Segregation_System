import cv2
import numpy as np
import tensorflow as tf
import serial
import time

# =========================
# CONFIG
# =========================
CAMERA_INDEX = 0
WINDOW_W, WINDOW_H = 1000, 600
CONF_THRESHOLD = 70          # %
SERVO_COOLDOWN = 3           # seconds
COM_PORT = "COM5"
BAUD_RATE = 9600

CLASSES = ["Hazardous", "Organic", "Recyclable"]
SERVO_CMD = {
    "Organic": "L",
    "Recyclable": "R",
    "Hazardous": "M"
}

# =========================
# LOAD MODEL (SAME AS TRAINING)
# =========================
base = tf.keras.applications.ResNet50V2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights=None
)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    tf.keras.layers.Rescaling(1./255),
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax")
])

model.load_weights("waste_pro_final.keras")
print("✅ AI model loaded")

# =========================
# CONNECT ARDUINO
# =========================
arduino = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
time.sleep(2)
print("✅ Arduino connected")

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_H)

last_servo_time = 0
last_label = None

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (WINDOW_W, WINDOW_H))

    # ROI box
    box_size = 300
    cx, cy = WINDOW_W // 2, WINDOW_H // 2
    x1, y1 = cx - box_size // 2, cy - box_size // 2
    x2, y2 = cx + box_size // 2, cy + box_size // 2

    roi = frame[y1:y2, x1:x2]

    # AI preprocess
    img = cv2.resize(roi, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img, verbose=0)[0]
    idx = np.argmax(preds)
    label = CLASSES[idx]
    conf = preds[idx] * 100

    # =========================
    # UI DRAW
    # =========================
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    text = f"{label} : {conf:.1f}%"
    color = (0, 255, 0) if conf > CONF_THRESHOLD else (0, 0, 255)

    cv2.putText(
        frame,
        text,
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        color,
        3
    )

    cv2.putText(
        frame,
        "Press Q to Quit",
        (30, WINDOW_H - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (200, 200, 200),
        2
    )

    # =========================
    # SERVO CONTROL (SAFE)
    # =========================
    current_time = time.time()

    if conf > CONF_THRESHOLD:
        if (label != last_label) or (current_time - last_servo_time > SERVO_COOLDOWN):
            cmd = SERVO_CMD[label]
            arduino.write(cmd.encode())
            print(f"Detected: {label} ({conf:.1f}%) → Servo {cmd}")

            last_servo_time = current_time
            last_label = label

    # =========================
    # SHOW WINDOW
    # =========================
    cv2.imshow("AI Smart Waste Segregation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()
arduino.close()
print("🛑 System stopped")
