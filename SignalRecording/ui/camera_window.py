from PyQt5 import QtCore, QtWidgets, QtGui
from datetime import datetime

import csv
import cv2
import numpy as np
import logging
import win32file
import sifi_bridge_py as sbp
import pyqtgraph as pg
import time
import json


class CameraStreamWindow(QtWidgets.QWidget):
    """Separate window for displaying Kinect camera stream."""
    
    closed = QtCore.pyqtSignal()
    
    def __init__(self, pipe_handle):
        super().__init__()
        self.pipe_handle = pipe_handle
        self.running = False
        self.frame_count = 0
        
        self.setWindowTitle("Kinect Camera Stream")
        self.resize(1280, 720)
        
        layout = QtWidgets.QVBoxLayout()
        
        # Image label
        self.image_label = QtWidgets.QLabel()
        self.image_label.setScaledContents(True)
        layout.addWidget(self.image_label)
        
        # Control buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.save_screenshot_button = QtWidgets.QPushButton("Save Screenshot")
        self.close_button = QtWidgets.QPushButton("Close Stream")
        
        self.save_screenshot_button.clicked.connect(self.save_screenshot)
        self.close_button.clicked.connect(self.close)
        
        button_layout.addWidget(self.save_screenshot_button)
        button_layout.addWidget(self.close_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
        # Timer for updating frames
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        self.current_frame = None
    
    def start_stream(self):
        """Start streaming frames."""
        self.running = True
        self.timer.start(33)  # ~30 FPS
        logging.info("Camera stream started")
    
    def update_frame(self):
        """Fetch and display next frame."""
        if not self.running or self.pipe_handle is None:
            return
        
        try:
            # Send request
            win32file.WriteFile(self.pipe_handle, b"Request color image")
            
            # Read response
            COLOR_BUFSIZE = 1920 * 1080 * 3
            result, color_data = win32file.ReadFile(self.pipe_handle, COLOR_BUFSIZE)
            
            if len(color_data) == 0:
                return
            
            # Decode MJPEG
            color_array = np.frombuffer(color_data, dtype=np.uint8)
            color_img = cv2.imdecode(color_array, cv2.IMREAD_COLOR)
            
            if color_img is None:
                return
            
            self.current_frame = color_img
            self.frame_count += 1
            
            # Convert BGR to RGB for Qt display
            rgb_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_img.shape
            bytes_per_line = ch * w
            
            qt_image = QtGui.QImage(rgb_img.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qt_image)
            
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.size(), 
                QtCore.Qt.KeepAspectRatio, 
                QtCore.Qt.SmoothTransformation
            ))
            
            if self.frame_count % 30 == 0:
                logging.info(f"Camera frames received: {self.frame_count}")
                
        except Exception as e:
            logging.error(f"Error updating frame: {e}")
    
    def save_screenshot(self):
        """Save current frame as screenshot."""
        if self.current_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"kinect_screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, self.current_frame)
            logging.info(f"Screenshot saved: {filename}")
            QtWidgets.QMessageBox.information(self, "Saved", f"Screenshot saved: {filename}")
    
    def closeEvent(self, event):
        """Handle window close."""
        self.running = False
        self.timer.stop()
        logging.info(f"Camera stream closed. Total frames: {self.frame_count}")
        self.closed.emit()
        event.accept()
