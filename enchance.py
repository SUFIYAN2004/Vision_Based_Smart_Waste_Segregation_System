import sys, time, csv
import cv2, serial
import numpy as np
import tensorflow as tf
from collections import deque, Counter
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QHBoxLayout, QVBoxLayout,
    QTableWidget, QTableWidgetItem, QMessageBox, QProgressBar,
    QPushButton, QToolBar, QFrame, QTabWidget
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSize
from PySide6.QtGui import QImage, QPixmap, QFont, QColor, QIcon

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# ===============================
# ARDUINO
# ===============================
arduino = serial.Serial("COM5", 9600, timeout=1)
time.sleep(2)

def send_to_arduino(label):
    mapping = {
        "Organic": "L",
        "Hazardous": "M",
        "Recyclable": "R"
    }
    arduino.write(mapping[label].encode())

# ===============================
# MODEL
# ===============================
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
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax")
    ])
    model.load_weights("waste_pro_final.keras")
    return model

MODEL = load_model()

CLASSES = ["Hazardous", "Organic", "Recyclable"]
COLORS = {
    "Organic": "#4CAF50",
    "Recyclable": "#FF5722",
    "Hazardous": "#2196F3"
}
RGB_COLORS = {
    "Organic": (0, 255, 0),
    "Recyclable": (255, 0, 0),
    "Hazardous": (0, 0, 255)
}
REWARDS = {"Organic": 10, "Recyclable": 8, "Hazardous": 5}

# ===============================
# ADVANCED PIE CHART
# ===============================
class AdvancedPieChart(FigureCanvasQTAgg):
    def __init__(self):
        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.fig.patch.set_facecolor('#f7f7f7')
        super().__init__(self.fig)
        self.counts = Counter()
        self.draw_empty()

    def draw_empty(self):
        self.ax.clear()
        self.ax.text(0.5, 0.5, "No data yet",
                     ha="center", va="center", fontsize=14,
                     color="#999", weight="bold")
        self.ax.set_aspect('equal')
        self.draw()

    def update_chart(self, label):
        self.counts[label] += 1
        self.ax.clear()

        labels = list(self.counts.keys())
        values = list(self.counts.values())
        colors = [COLORS[label].lstrip('#') for label in labels]
        colors = ['#' + c for c in colors]

        wedges, texts, autotexts = self.ax.pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            textprops={'fontsize': 10, 'weight': 'bold'}
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(9)
            autotext.set_weight('bold')

        self.ax.set_title("Waste Distribution", fontsize=12, weight='bold', pad=20)
        self.draw()

# ===============================
# LIVE STATS WIDGET
# ===============================
class LiveStatsWidget(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        layout = QVBoxLayout(self)

        # Header
        header = QLabel("Live Prediction Confidence")
        header_font = QFont()
        header_font.setPointSize(11)
        header_font.setBold(True)
        header.setFont(header_font)
        layout.addWidget(header)

        # Progress bars for each class
        self.progress_bars = {}
        for cls in CLASSES:
            bar = QProgressBar()
            bar.setMaximum(100)
            bar.setMinimum(0)
            bar.setValue(0)
            bar.setFormat(f"{cls}: %p%")
            color = COLORS[cls].lstrip('#')
            bar.setStyleSheet(f"""
                QProgressBar {{
                    border: 2px solid #ddd;
                    border-radius: 5px;
                    text-align: center;
                    background: #fff;
                }}
                QProgressBar::chunk {{
                    background: #{color};
                    border-radius: 3px;
                }}
            """)
            self.progress_bars[cls] = bar
            layout.addWidget(bar)

        layout.addStretch()
        self.setStyleSheet("""
            QFrame {
                background: #ffffff;
                border-radius: 8px;
                padding: 10px;
                border: 1px solid #e0e0e0;
            }
        """)

    def update_predictions(self, predictions):
        for i, cls in enumerate(CLASSES):
            confidence = int(predictions[i] * 100)
            self.progress_bars[cls].setValue(confidence)

# ===============================
# LEADERBOARD WIDGET
# ===============================
class LeaderboardWidget(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        layout = QVBoxLayout(self)

        # Header
        header = QLabel("Session Leaderboard")
        header_font = QFont()
        header_font.setPointSize(11)
        header_font.setBold(True)
        header.setFont(header_font)
        layout.addWidget(header)

        # Leaderboard table
        self.table = QTableWidget(3, 2)
        self.table.setHorizontalHeaderLabels(["Type", "Count"])
        self.table.setMaximumHeight(150)
        
        for i, cls in enumerate(CLASSES):
            self.table.setItem(i, 0, QTableWidgetItem(cls))
            self.table.setItem(i, 1, QTableWidgetItem("0"))
        
        self.table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #e0e0e0;
                border-radius: 5px;
                background: #fff;
            }
            QHeaderView::section {
                background: #1976d2;
                color: white;
                padding: 5px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.table)

        # Total points
        self.total_label = QLabel("Total Points: 0")
        self.total_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(self.total_label)

        layout.addStretch()
        self.setStyleSheet("""
            QFrame {
                background: #ffffff;
                border-radius: 8px;
                padding: 10px;
                border: 1px solid #e0e0e0;
            }
        """)
        self.counts = Counter()
        self.total_points = 0

    def update_leaderboard(self, label):
        self.counts[label] += 1
        self.total_points += REWARDS[label]

        for i, cls in enumerate(CLASSES):
            self.table.item(i, 1).setText(str(self.counts[cls]))

        self.total_label.setText(f"Total Points: {self.total_points}")

# ===============================
# THREADS
# ===============================
class CameraThread(QThread):
    frame_ready = Signal(object)

    def run(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.frame_ready.emit(frame)

    def stop(self):
        self.cap.release()
        self.quit()

class AIThread(QThread):
    result_ready = Signal(str, float, object)

    def __init__(self):
        super().__init__()
        self.frame = None
        self.running = True

    def update_frame(self, frame):
        self.frame = frame

    def run(self):
        while self.running:
            if self.frame is None:
                continue

            # ============ ONLY PROCESS INSIDE DETECTION BOX ============
            h, w, _ = self.frame.shape
            box = 300
            x1, y1 = (w - box) // 2, (h - box) // 2
            x2, y2 = x1 + box, y1 + box
            
            # Crop to detection box only
            crop = self.frame[y1:y2, x1:x2]
            
            img = cv2.resize(crop, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.expand_dims(img, axis=0)

            preds = MODEL.predict(img, verbose=0)[0]
            idx = np.argmax(preds)

            self.result_ready.emit(CLASSES[idx], preds[idx] * 100, preds)
            self.msleep(400)

    def stop(self):
        self.running = False
        self.quit()

# ===============================
# MAIN UI - NEXT LEVEL
# ===============================
class SmartWasteUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🌱 AI Smart Waste Segregation System")
        self.showMaximized()

        # Live state
        self.current_label = ""
        self.current_conf = 0.0

        # Apply modern stylesheet
        self.apply_stylesheet()

        # Create header toolbar
        self.header = self.create_header()

        # Camera view
        self.video = QLabel(alignment=Qt.AlignCenter)
        self.video.setMinimumSize(400, 400)
        self.video.setStyleSheet("""
            QLabel {
                border: 3px solid #1976d2;
                border-radius: 10px;
                background: #000;
            }
        """)

        # Main analytics section
        self.analytics_tabs = QTabWidget()

        # Tab 1: Real-time stats + Leaderboard
        stats_widget = QWidget()
        stats_layout = QVBoxLayout(stats_widget)
        self.stats_panel = LiveStatsWidget()
        self.leaderboard = LeaderboardWidget()
        stats_layout.addWidget(self.stats_panel)
        stats_layout.addWidget(self.leaderboard)
        self.analytics_tabs.addTab(stats_widget, "📊 Live Stats")

        # Tab 2: Pie chart
        self.pie = AdvancedPieChart()
        self.analytics_tabs.addTab(self.pie, "📈 Distribution")

        # Tab 3: History table
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Time", "Date", "Type", "Confidence", "Reward"]
        )
        self.table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #e0e0e0;
                border-radius: 5px;
                background: #fff;
            }
            QHeaderView::section {
                background: #1976d2;
                color: white;
                padding: 6px;
                font-weight: bold;
            }
        """)
        self.analytics_tabs.addTab(self.table, "📋 History")

        # Layout
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.header)

        content_layout = QHBoxLayout()
        content_layout.addWidget(self.video, 7)
        content_layout.addWidget(self.analytics_tabs, 3)

        main_layout.addLayout(content_layout)

        # Threads
        self.camera = CameraThread()
        self.ai = AIThread()

        self.camera.frame_ready.connect(self.update_frame)
        self.camera.frame_ready.connect(self.ai.update_frame)
        self.ai.result_ready.connect(self.handle_result)

        self.pred_buffer = deque(maxlen=6)
        self.last_action = 0

        self.camera.start()
        self.ai.start()

    def apply_stylesheet(self):
        self.setStyleSheet("""
            QWidget {
                background: #f7f7f7;
                font-family: 'Segoe UI', Arial;
            }
            QLabel {
                color: #333;
            }
            QPushButton {
                background: #1976d2;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background: #1565c0;
            }
            QPushButton:pressed {
                background: #0d47a1;
            }
            QTableWidget {
                background: #fff;
                border-radius: 8px;
                gridline-color: #e0e0e0;
            }
            QTabWidget::pane {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
            }
            QTabBar::tab {
                background: #e0e0e0;
                color: #333;
                padding: 6px 16px;
                margin: 2px;
                border-radius: 4px;
            }
            QTabBar::tab:selected {
                background: #1976d2;
                color: white;
                font-weight: bold;
            }
        """)

    def create_header(self):
        header_widget = QWidget()
        header_widget.setStyleSheet("""
            QWidget {
                background: #1976d2;
                color: white;
                padding: 10px;
            }
            QLabel {
                color: white;
                font-weight: bold;
            }
        """)
        header_layout = QHBoxLayout(header_widget)

        title = QLabel("🌱 AI Smart Waste Segregation System")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        header_layout.addWidget(title)

        header_layout.addStretch()

        self.clock_label = QLabel()
        self.clock_label.setFont(QFont("Arial", 11, QFont.Bold))
        header_layout.addWidget(self.clock_label)

        # Update clock every second
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_clock)
        self.timer.start(1000)

        return header_widget

    def update_clock(self):
        now = datetime.now()
        self.clock_label.setText(now.strftime("📅 %d-%m-%Y | 🕐 %H:%M:%S"))

    def handle_result(self, label, conf, predictions):
        self.current_label = label
        self.current_conf = conf

        # Update live prediction bars
        self.stats_panel.update_predictions(predictions)

        self.pred_buffer.append(label)
        common, count = Counter(self.pred_buffer).most_common(1)[0]

        if count < 3 or conf < 70:
            return

        if time.time() - self.last_action < 5:
            return

        self.last_action = time.time()
        self.pred_buffer.clear()

        send_to_arduino(common)

        reward = REWARDS[common]
        now = datetime.now()

        # ============ ADD TO HISTORY TABLE WITH COLOR ============
        row = self.table.rowCount()
        self.table.insertRow(row)

        data = [
            now.strftime("%H:%M:%S"),
            now.strftime("%d-%m-%Y"),
            common,
            f"{conf:.1f}%",
            f"+{reward}"
        ]

        # Set background colors based on waste type
        if common == "Organic":
            bg_color = QColor("#c8e6c9")  # Light green
            fg_color = QColor("#2e7d32")  # Dark green
        elif common == "Hazardous":
            bg_color = QColor("#bbdefb")  # Light blue
            fg_color = QColor("#1565c0")  # Dark blue
        elif common == "Recyclable":
            bg_color = QColor("#ffccbc")  # Light orange
            fg_color = QColor("#e64a19")  # Dark orange

        for c, v in enumerate(data):
            item = QTableWidgetItem(v)
            item.setFont(QFont("Arial", 10, QFont.Bold))
            item.setBackground(bg_color)
            item.setForeground(fg_color)
            self.table.setItem(row, c, item)

        # Update leaderboard
        self.leaderboard.update_leaderboard(common)

        # Update pie chart
        self.pie.update_chart(common)

        # Save to CSV
        with open("analytics.csv", "a", newline="") as f:
            csv.writer(f).writerow(data)

        # Show enhanced popup
        self.popup(common, conf, reward)

    def popup(self, label, conf, reward):
        msg = QMessageBox(self)
        msg.setWindowTitle("✅ Waste Classified")
        msg.setIcon(QMessageBox.Information)
        msg.setMinimumSize(450, 280)

        html_text = f"""
        <div style="text-align: center; padding: 20px;">
            <h1 style="color: {COLORS[label]}; margin: 0;">🗑️ {label}</h1>
            <h3 style="color: #666; margin: 10px 0;">Confidence: <b>{conf:.1f}%</b></h3>
            <hr style="border: none; border-top: 2px solid #e0e0e0; margin: 15px 0;">
            <p style="font-size: 16px; color: #333;">
                <b>Reward Points:</b> <span style="color: #4CAF50; font-size: 20px;">+{reward} 🎯</span>
            </p>
            <p style="color: #999; font-size: 12px; margin-top: 15px;">
                Item successfully classified and sorted!
            </p>
        </div>
        """

        msg.setText(html_text)
        msg.exec()

    def update_frame(self, frame):
        h, w, _ = frame.shape
        box = 300
        x1, y1 = (w - box) // 2, (h - box) // 2
        x2, y2 = x1 + box, y1 + box

        # Draw capture box with glow effect
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), (0, 200, 0), 1)

        # Draw background for text
        if self.current_label:
            color = RGB_COLORS[self.current_label]
            cv2.rectangle(frame, (20, 40), (600, 80), (0, 0, 0), -1)
            cv2.rectangle(frame, (20, 40), (600, 80), color, 2)
            cv2.putText(
                frame,
                f"{self.current_label.upper()} | {self.current_conf:.1f}% confidence",
                (35, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                color,
                2
            )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
        self.video.setPixmap(QPixmap.fromImage(img))

    def closeEvent(self, event):
        print("🛑 Closing application...")

        if self.camera.isRunning():
            self.camera.stop()
            self.camera.wait()

        if self.ai.isRunning():
            self.ai.stop()
            self.ai.wait()

        global arduino
        if arduino and arduino.is_open:
            arduino.close()
            print("🔌 Arduino disconnected")

        event.accept()


# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = SmartWasteUI()
    win.show()
    sys.exit(app.exec())