import sys
import os

if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
    sys.path.insert(0, base_path)
    os.chdir(base_path)
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

import cv2
import numpy as np
from collections import deque
import time
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QWidget, QFormLayout, 
                             QLineEdit, QGroupBox, QMessageBox, QStatusBar)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap

from varsity_logic import AppConfig, VarsityAIEngine, normalize_lighting


class ClickableVideoLabel(QLabel):
    clicked = pyqtSignal(int, int)
    
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(500)
        self.setMinimumWidth(800)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #1a1a2e; border-radius: 10px; border: 2px solid #2d3561;")
        # ADD THIS LINE - makes the video label focusable
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(int(event.position().x()), int(event.position().y()))
        # ADD THIS LINE - grab focus when clicked
        self.setFocus()
        super().mousePressEvent(event)


class VideoProcessor(QThread):
    frame_ready = pyqtSignal(object)
    status_update = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    learning_update = pyqtSignal(str, int)
    learning_complete = pyqtSignal(str)
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.running = True
        self.cap = None
        self.engine = VarsityAIEngine(config)
        self.learning_active = False
        self.learning_team = None
        
    def setup(self):
        try:
            if not self.engine._init_model():
                self.error_occurred.emit("Failed to load YOLO model")
                return False
            
            for cam_id in [0, 1]:
                self.cap = cv2.VideoCapture(cam_id)
                if self.cap.isOpened():
                    self.config.camera_id = cam_id
                    break
            
            if not self.cap or not self.cap.isOpened():
                self.error_occurred.emit("Cannot open camera")
                return False
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
            
            # Set default court polygon
            if self.engine.court_calibrator.court_polygon is None:
                self.engine.court_calibrator.court_polygon = np.array([
                    [0, 0],
                    [self.config.frame_width, 0],
                    [self.config.frame_width, self.config.frame_height],
                    [0, self.config.frame_height]
                ], dtype=np.int32)
            
            return True
        except Exception as e:
            self.error_occurred.emit(str(e))
            return False
    
    def start_learning(self, team):
        self.learning_active = True
        self.learning_team = team
        self.engine.player_tracker.color_detector.start_learning(team)
        self.status_update.emit(f"Click on {team} team jerseys (5 clicks)")
    
    def process_click(self, x, y, label_w, label_h):
        if not self.learning_active or not hasattr(self.engine, 'current_frame'):
            return
        
        frame = self.engine.current_frame
        if frame is None:
            return
        
        scale_x = self.config.frame_width / label_w
        scale_y = self.config.frame_height / label_h
        fx = int(x * scale_x)
        fy = int(y * scale_y)
        fx = max(0, min(fx, self.config.frame_width - 1))
        fy = max(0, min(fy, self.config.frame_height - 1))
        
        results = self.engine.model(frame, classes=[0], conf=0.4, verbose=False)
        
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            for box in boxes:
                x1, y1, x2, y2 = box
                if x1 <= fx <= x2 and y1 <= fy <= y2:
                    crop = self.engine._crop_jersey_region(frame, [x1, y1, x2, y2])
                    if crop is not None and crop.size > 0:
                        detector = self.engine.player_tracker.color_detector
                        success, count = detector.add_sample_from_click(crop)
                        if success:
                            self.learning_update.emit(self.learning_team, count)
                            if count >= 5:
                                self.learning_active = False
                                self.learning_complete.emit(self.learning_team)
                    return
    
    def run(self):
        if not self.setup():
            return
        
        self.status_update.emit("Ready - Click LEARN buttons to start")
        
        for _ in range(10):
            self.cap.read()
        
        frame_count = 0
        SKIP_DETECTION = 2  # Run detection every 2 frames (keeps smoothness)
        
        t1_range = ([0, 100, 100], [10, 255, 255])
        t2_range = ([100, 100, 100], [130, 255, 255])
        
        cached_westy = []
        cached_opp = []
        cached_cam_dx = 0
        cached_cam_dy = 0
        last_gray = None
        
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            frame = cv2.resize(frame, (self.config.frame_width, self.config.frame_height))
            frame = normalize_lighting(frame)
            self.engine.current_frame = frame.copy()
            height, width = frame.shape[:2]
            
            # Run heavy detection every SKIP_DETECTION frames
            if frame_count % SKIP_DETECTION == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                y_cutoff = int(height * 0.22)
                
                cam_dx, cam_dy = self.engine.flow_tracker.update(gray, frame_count, y_cutoff)
                cached_cam_dx, cached_cam_dy = cam_dx, cam_dy
                last_gray = gray
                
                westy_players, opp_players, pack_momentum = self.engine._process_detections(
                    frame, frame_count, t1_range, t2_range
                )
                
                adjusted_momentum = [m - cam_dx for m in pack_momentum]
                self.engine.possession.update(adjusted_momentum, self.engine.possession.westy_attacks_right)
                
                if self.engine.possession.current_possession == "Westy" and westy_players and frame_count % 15 == 0:
                    avg_qsq = np.mean([p.get('qsq', 50) for p in westy_players])
                    self.engine.scout.log_shot_quality(avg_qsq)
                
                cached_westy = westy_players
                cached_opp = opp_players
            else:
                # Use cached results
                westy_players = cached_westy
                opp_players = cached_opp
                cam_dx, cam_dy = cached_cam_dx, cached_cam_dy
            
            # Draw everything (every frame)
            self.engine._draw_player_boxes(frame, westy_players, opp_players)
            
            offensive_players = westy_players if self.engine.possession.current_possession == "Westy" else opp_players
            self.engine._detect_pick_and_roll(frame, offensive_players)
            self.engine._detect_mismatches(frame, westy_players, opp_players, frame_count)
            self.engine._detect_cramping(frame, westy_players, width, height, frame_count)
            
            defending_team = opp_players if self.engine.possession.current_possession == "Westy" else westy_players
            self.engine._detect_defensive_shell(frame, defending_team, self.engine.possession.current_possession)
            
            self.engine._draw_hud(frame)
            
            # Learning overlay and status text (your existing code here - keep as is)
            if self.learning_active:
                detector = self.engine.player_tracker.color_detector
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (width, 60), (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
                count = len(detector.home_samples) if self.learning_team == 'home' else len(detector.away_samples)
                cv2.putText(frame, f"LEARNING {self.learning_team.upper()} - Click jerseys ({count}/5)", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            detector = self.engine.player_tracker.color_detector
            home_status = "✅ Home: Learned" if detector.home_learned else f"Home: {len(detector.home_samples)}/5"
            away_status = "✅ Away: Learned" if detector.away_learned else f"Away: {len(detector.away_samples)}/5"
            cv2.putText(frame, home_status, (width - 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, away_status, (width - 180, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Emit frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            self.frame_ready.emit(qt_img)
            
            frame_count += 1
            time.sleep(0.01)
        
        if self.cap:
            self.cap.release()
    
    def stop(self):
        self.running = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voop Basketball Analytics")
        self.setMinimumSize(1400, 850)
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)
        
        left = QWidget()
        left.setFixedWidth(320)
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(12)
        
        title = QLabel("🏀 VOOP")
        title.setStyleSheet("font-size: 28px; font-weight: bold; color: #e94560;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(title)
        
        self.start_btn = QPushButton("START CAMERA")
        self.start_btn.setStyleSheet("background-color: #2ecc71; color: white; font-size: 14px; padding: 12px; border-radius: 6px;")
        self.start_btn.clicked.connect(self.start_camera)
        left_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("STOP CAMERA")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("background-color: #e74c3c; color: white; font-size: 14px; padding: 12px; border-radius: 6px;")
        self.stop_btn.clicked.connect(self.stop_camera)
        left_layout.addWidget(self.stop_btn)
        
        left_layout.addSpacing(20)
        
        learn_group = QGroupBox("TEAM COLOR LEARNING")
        learn_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        learn_layout = QVBoxLayout(learn_group)
        
        self.home_btn = QPushButton("LEARN HOME TEAM")
        self.home_btn.setEnabled(False)
        self.home_btn.setStyleSheet("background-color: #f39c12; padding: 10px; font-size: 13px;")
        self.home_btn.clicked.connect(lambda: self.start_learning('home'))
        learn_layout.addWidget(self.home_btn)
        
        self.home_status = QLabel("⚡ Not learned (0/5)")
        self.home_status.setStyleSheet("color: #f39c12; padding: 5px;")
        learn_layout.addWidget(self.home_status)
        
        learn_layout.addSpacing(10)
        
        self.away_btn = QPushButton("LEARN AWAY TEAM")
        self.away_btn.setEnabled(False)
        self.away_btn.setStyleSheet("background-color: #e74c3c; padding: 10px; font-size: 13px;")
        self.away_btn.clicked.connect(lambda: self.start_learning('away'))
        learn_layout.addWidget(self.away_btn)
        
        self.away_status = QLabel("⚡ Not learned (0/5)")
        self.away_status.setStyleSheet("color: #e74c3c; padding: 5px;")
        learn_layout.addWidget(self.away_status)
        
        left_layout.addWidget(learn_group)
        
        name_group = QGroupBox("DISPLAY NAMES")
        name_layout = QFormLayout(name_group)
        
        self.home_name = QLineEdit("Home")
        self.home_name.setStyleSheet("padding: 6px;")
        name_layout.addRow("Home Team:", self.home_name)
        
        self.away_name = QLineEdit("Away")
        self.away_name.setStyleSheet("padding: 6px;")
        name_layout.addRow("Away Team:", self.away_name)
        
        left_layout.addWidget(name_group)
        
        guide = QLabel(
            "HOW TO USE:\n\n"
            "1. Click START CAMERA\n"
            "2. Click LEARN HOME TEAM\n"
            "3. Click on 5 Home players\n"
            "4. Click LEARN AWAY TEAM\n"
            "5. Click on 5 Away players\n\n"
            "✅ System automatically tracks!\n\n"
            "🟡 Yellow = Home\n"
            "🔴 Red = Away"
        )
        guide.setWordWrap(True)
        guide.setStyleSheet("background-color: #2c3e50; padding: 12px; border-radius: 8px;")
        left_layout.addWidget(guide)
        
        left_layout.addStretch()
        
        right = QWidget()
        right_layout = QVBoxLayout(right)
        
        self.video_label = ClickableVideoLabel()
        self.video_label.clicked.connect(self.on_click)
        right_layout.addWidget(self.video_label)
        
        layout.addWidget(left)
        layout.addWidget(right, stretch=2)
        
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Click START CAMERA")
        
        self.setStyleSheet("""
            QMainWindow { background-color: #1a1a2e; }
            QGroupBox { color: white; border: 2px solid #2d3561; border-radius: 8px; margin-top: 12px; }
            QGroupBox::title { color: #e94560; }
            QLabel { color: white; }
            QLineEdit { background-color: #2d3561; color: white; border: 1px solid #e94560; border-radius: 4px; }
        """)
        
        self.processor = None
    
    def start_camera(self):
        if self.processor:
            self.processor.stop()
        
        self.processor = VideoProcessor(AppConfig())
        self.processor.frame_ready.connect(self.update_frame)
        self.processor.status_update.connect(self.status_bar.showMessage)
        self.processor.error_occurred.connect(self.handle_error)
        self.processor.learning_update.connect(self.on_learning_update)
        self.processor.learning_complete.connect(self.on_learning_complete)
        self.processor.start()
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.home_btn.setEnabled(True)
        self.away_btn.setEnabled(True)
    
    def stop_camera(self):
        if self.processor:
            self.processor.stop()
            self.processor = None
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.home_btn.setEnabled(False)
        self.away_btn.setEnabled(False)
        self.home_status.setText("⚡ Not learned (0/5)")
        self.away_status.setText("⚡ Not learned (0/5)")
        self.video_label.clear()
        self.status_bar.showMessage("Camera stopped")
    
    def start_learning(self, team):
        if self.processor:
            self.processor.start_learning(team)
    
    def on_learning_update(self, team, count):
        if team == 'home':
            self.home_status.setText(f"📸 Learning: {count}/5")
        else:
            self.away_status.setText(f"📸 Learning: {count}/5")
    
    def on_learning_complete(self, team):
        if team == 'home':
            self.home_status.setText("✅ HOME TEAM LEARNED!")
            self.status_bar.showMessage("Home learned! Now learn Away team")
        else:
            self.away_status.setText("✅ AWAY TEAM LEARNED!")
            self.status_bar.showMessage("Both teams learned! System is tracking")
    
    def on_click(self, x, y):
        # Focus the video label when clicking on it
        self.video_label.setFocus()
        if self.processor and self.processor.learning_active:
            self.processor.process_click(x, y, self.video_label.width(), self.video_label.height())
    
    def update_frame(self, qt_img):
        pixmap = QPixmap.fromImage(qt_img)
        scaled = pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.video_label.setPixmap(scaled)
    
    def handle_error(self, msg):
        QMessageBox.critical(self, "Error", msg)
        self.stop_camera()
    
    def closeEvent(self, event):
        if self.processor:
            self.processor.stop()
        event.accept()
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_H:
            if self.processor and self.processor.engine:
                self.processor.engine.possession.flip_attack_direction()
                self.status_bar.showMessage("Attack direction flipped", 2000)
        elif event.key() == Qt.Key.Key_R:
            if self.processor and self.processor.engine:
                filename = self.processor.engine.scout.generate_timeout_report()
                self.status_bar.showMessage(f"Report saved: {filename}", 3000)
                QMessageBox.information(self, "Report Generated", f"Timeout report saved as:\n{filename}")
        elif event.key() == Qt.Key.Key_Q:
            self.stop_camera()
            self.close()
        super().keyPressEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())