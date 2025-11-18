from ui.camera_window import CameraStreamWindow
from ui.marker_editor import MarkerEditorDialog
from utils.markers import MarkerItem
from utils.saving import save_recorded_data
from collections import deque
from PyQt5 import QtCore, QtWidgets, QtGui
from datetime import datetime
from pathlib import Path

import csv
import cv2
import numpy as np
import logging
import win32file
import sifi_bridge_py as sbp
import pyqtgraph as pg
import time
import json


class PlotWindow(QtWidgets.QMainWindow):
    """Main window for displaying and analyzing real-time data from the BioPoint device."""

    def __init__(self, worker, kinect_worker):
        super().__init__()

        self.worker = worker
        self.kinect_worker = kinect_worker
        self.camera_window = None

        # Define channel names and sampling frequencies
        self.channel_names = ["EMG", "ECG", "Temperature", "EDA", "PPG", "IMU"]
        self.FS = {"emg": 2000, "ecg": 500, "temperature": 1, "eda": 100, "ppg": 50, "imu": 50}

        # Initialize data buffers
        self.data_buffers = {
            "emg": deque(maxlen=self.FS["emg"] * 60),
            "ecg": deque(maxlen=self.FS["ecg"] * 60),
            "temperature": deque(maxlen=self.FS["temperature"] * 60),
            "eda": deque(maxlen=self.FS["eda"] * 60),
            "ppg_blue": deque(maxlen=int(self.FS["ppg"] * 60)),
            "ppg_green": deque(maxlen=int(self.FS["ppg"] * 60)),
            "ppg_red": deque(maxlen=int(self.FS["ppg"] * 60)),
            "ppg_ir": deque(maxlen=int(self.FS["ppg"] * 60)),
            "ax": deque(maxlen=self.FS["imu"] * 60),
            "ay": deque(maxlen=self.FS["imu"] * 60),
            "az": deque(maxlen=self.FS["imu"] * 60),
        }
        
        # Storage for all recorded data
        self.recorded_data = {
            "emg": [], "ecg": [], "temperature": [], "eda": [],
            "ppg_blue": [], "ppg_green": [], "ppg_red": [], "ppg_ir": [],
            "ax": [], "ay": [], "az": []
        }
        
        self.data_start_time = None
        self.sample_counts = {
            "emg": 0, "ecg": 0, "temperature": 0, "eda": 0, "ppg": 0, "imu": 0
        }

        # Markers
        self.markers = []
        self.marker_items = {}
        self.next_marker_number = 1
        
        self.is_connected = False
        self.acquisition_running = False
        self.is_recording = False
        self.kinect_connected = False
        self.kinect_recording = False

        self.init_ui()
        self.init_plots()
        self.connect_signals()
        self.init_plot_timer()
        self.update_button_states()

    def init_ui(self):
        """Initializes the main window's user interface."""
        self.setWindowTitle("BioPoint Real-time Monitoring & Kinect Control")
        self.resize(1400, 1000)

        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)

        self.main_layout = QtWidgets.QVBoxLayout()

        # Create Menu Bar
        menu_bar = self.menuBar()
        
        # Connection menu
        connection_menu = menu_bar.addMenu("Connection")
        connect_default_action = QtWidgets.QAction("Connect BioPoint Default", self)
        connect_default_action.triggered.connect(self.connect_default_biopoint)
        connection_menu.addAction(connect_default_action)

        connect_mac_action = QtWidgets.QAction("Connect BioPoint by MAC...", self)
        connect_mac_action.triggered.connect(self.connect_biopoint_mac)
        connection_menu.addAction(connect_mac_action)
        
        connection_menu.addSeparator()
        
        connect_kinect_action = QtWidgets.QAction("Connect to Kinect", self)
        connect_kinect_action.triggered.connect(self.connect_to_kinect)
        connection_menu.addAction(connect_kinect_action)
        
        disconnect_kinect_action = QtWidgets.QAction("Disconnect from Kinect", self)
        disconnect_kinect_action.triggered.connect(self.disconnect_from_kinect)
        connection_menu.addAction(disconnect_kinect_action)
        
        # File menu
        file_menu = menu_bar.addMenu("File")
        
        save_data_action = QtWidgets.QAction("Save Data...", self)
        save_data_action.triggered.connect(self.save_data)
        file_menu.addAction(save_data_action)
        
        # Markers menu
        markers_menu = menu_bar.addMenu("Markers")
        
        add_marker_action = QtWidgets.QAction("Add Marker (Space)", self)
        add_marker_action.setShortcut("Space")
        add_marker_action.triggered.connect(self.add_marker)
        markers_menu.addAction(add_marker_action)
        
        edit_markers_action = QtWidgets.QAction("Edit Markers...", self)
        edit_markers_action.triggered.connect(self.show_marker_editor)
        markers_menu.addAction(edit_markers_action)
        
        clear_markers_action = QtWidgets.QAction("Clear All Markers", self)
        clear_markers_action.triggered.connect(self.clear_all_markers)
        markers_menu.addAction(clear_markers_action)

        # Create Plot Widgets
        self.emg_plot_widget = pg.PlotWidget(title="EMG")
        self.ecg_plot_widget = pg.PlotWidget(title="ECG")
        self.temperature_plot_widget = pg.PlotWidget(title="Temperature")
        self.eda_plot_widget = pg.PlotWidget(title="EDA")
        self.ppg_plot_widget = pg.PlotWidget(title="PPG")
        self.imu_norm_plot_widget = pg.PlotWidget(title="IMU Norm")
        
        self.plot_widgets = [
            self.emg_plot_widget,
            self.ecg_plot_widget,
            self.temperature_plot_widget,
            self.eda_plot_widget,
            self.ppg_plot_widget,
            self.imu_norm_plot_widget
        ]

        # Create Plot Curves
        self.emg_curve = self.emg_plot_widget.plot(pen='y')
        self.ecg_curve = self.ecg_plot_widget.plot(pen='c')
        self.temperature_curve = self.temperature_plot_widget.plot(pen='m')
        self.eda_curve = self.eda_plot_widget.plot(pen='g')

        self.ppg_blue_curve = self.ppg_plot_widget.plot(pen='b', name="Blue")
        self.ppg_green_curve = self.ppg_plot_widget.plot(pen='g', name="Green")
        self.ppg_red_curve = self.ppg_plot_widget.plot(pen='r', name="Red")
        self.ppg_ir_curve = self.ppg_plot_widget.plot(pen=(255, 255, 255), name="Infrared")

        self.imu_norm_curve = self.imu_norm_plot_widget.plot(pen='m')

        for plot_widget in self.plot_widgets:
            self.main_layout.addWidget(plot_widget)

        # Status bar
        self.status_label = QtWidgets.QLabel("BioPoint: Disconnected")
        self.recording_label = QtWidgets.QLabel("BioPoint Recording: OFF")
        self.kinect_status_label = QtWidgets.QLabel("Kinect: Disconnected")
        self.kinect_recording_label = QtWidgets.QLabel("Kinect Recording: OFF")
        self.marker_label = QtWidgets.QLabel("Markers: 0")
        
        status_layout = QtWidgets.QHBoxLayout()
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.recording_label)
        status_layout.addWidget(QtWidgets.QLabel("|"))
        status_layout.addWidget(self.kinect_status_label)
        status_layout.addWidget(self.kinect_recording_label)
        status_layout.addWidget(QtWidgets.QLabel("|"))
        status_layout.addWidget(self.marker_label)
        status_layout.addStretch()
        
        self.main_layout.addLayout(status_layout)

        # Create Control Buttons Layout
        self.button_layout = QtWidgets.QHBoxLayout()

        # BioPoint controls
        self.start_button = QtWidgets.QPushButton("Start BioPoint")
        self.stop_button = QtWidgets.QPushButton("Stop BioPoint")
        self.record_button = QtWidgets.QPushButton("Start BioPoint Recording")
        
        # Kinect controls
        self.view_camera_button = QtWidgets.QPushButton("View Camera Stream")
        self.kinect_record_button = QtWidgets.QPushButton("Start Kinect Recording")
        self.record_all_button = QtWidgets.QPushButton("Start All Recording")

        # Other controls
        self.add_marker_button = QtWidgets.QPushButton("Add Marker (Space)")
        self.disconnect_button = QtWidgets.QPushButton("Disconnect All")
        self.power_off_button = QtWidgets.QPushButton("Power Off BioPoint")

        self.start_button.clicked.connect(self.start_acquisition)
        self.stop_button.clicked.connect(self.stop_acquisition)
        self.record_button.clicked.connect(self.toggle_recording)
        self.view_camera_button.clicked.connect(self.view_camera_stream)
        self.kinect_record_button.clicked.connect(self.toggle_kinect_recording)
        self.record_all_button.clicked.connect(self.start_recording_all)
        self.add_marker_button.clicked.connect(self.add_marker)
        self.disconnect_button.clicked.connect(self.disconnect_all)
        self.power_off_button.clicked.connect(self.power_off_biopoint)

        self.button_layout.addWidget(self.start_button)
        self.button_layout.addWidget(self.stop_button)
        self.button_layout.addWidget(self.record_button)
        self.button_layout.addWidget(QtWidgets.QLabel("|"))
        self.button_layout.addWidget(self.view_camera_button)
        self.button_layout.addWidget(self.kinect_record_button)
        self.button_layout.addWidget(self.record_all_button)
        self.button_layout.addWidget(QtWidgets.QLabel("|"))
        self.button_layout.addWidget(self.add_marker_button)
        self.button_layout.addWidget(self.disconnect_button)
        self.button_layout.addWidget(self.power_off_button)

        self.main_layout.addLayout(self.button_layout)
        self.central_widget.setLayout(self.main_layout)

    def init_plots(self):
        """Initialize plot settings."""
        self.emg_plot_widget.setLabel('left', "Amplitude", units='mV')
        self.emg_plot_widget.setLabel('bottom', "Time", units='s')
        self.ecg_plot_widget.setLabel('left', "Amplitude", units='mV')
        self.ecg_plot_widget.setLabel('bottom', "Time", units='s')
        self.temperature_plot_widget.setLabel('left', "Temperature", units='°C')
        self.temperature_plot_widget.setLabel('bottom', "Time", units='s')
        self.eda_plot_widget.setLabel('left', "Conductance", units='µS')
        self.eda_plot_widget.setLabel('bottom', "Time", units='s')
        self.ppg_plot_widget.setLabel('left', "Amplitude", units='Arb. Units')
        self.ppg_plot_widget.setLabel('bottom', "Time", units='s')
        self.imu_norm_plot_widget.setLabel('left', "Acceleration", units='g')
        self.imu_norm_plot_widget.setLabel('bottom', "Time", units='s')

    def connect_signals(self):
        """Connects signals from workers to slots."""
        self.worker.data_received.connect(self.handle_data)
        self.worker.finished.connect(self.handle_worker_finished)
        self.worker.stopped.connect(self.on_worker_stopped)
        self.worker.connected.connect(self.on_connected_status_changed)
        
        # Kinect worker signals
        self.kinect_worker.connected_changed.connect(self.on_kinect_connected_changed)
        self.kinect_worker.recording_status_changed.connect(self.on_kinect_recording_changed)

    def init_plot_timer(self):
        """Initializes the timer for updating the plots."""
        self.plot_timer = QtCore.QTimer()
        self.plot_timer.timeout.connect(self.update_plots)
        self.plot_timer.start(50)
        logging.info("Plot timer started.")

    def keyPressEvent(self, event):
        """Handle keyboard events."""
        if event.key() == QtCore.Qt.Key_Space:
            self.add_marker()
        else:
            super().keyPressEvent(event)

    def connect_to_kinect(self):
        """Connect to Kinect application."""
        logging.info("Connecting to Kinect...")
        QtCore.QMetaObject.invokeMethod(self.kinect_worker, "connect_to_kinect", 
                                      QtCore.Qt.QueuedConnection)

    def disconnect_from_kinect(self):
        """Disconnect from Kinect application."""
        logging.info("Disconnecting from Kinect...")
        
        # Close camera window if open
        if self.camera_window:
            self.camera_window.close()
            self.camera_window = None
        
        QtCore.QMetaObject.invokeMethod(self.kinect_worker, "disconnect_from_kinect", 
                                      QtCore.Qt.QueuedConnection)

    def view_camera_stream(self):
        """Open camera stream window."""
        if not self.kinect_connected:
            # Auto-connect if not connected
            logging.info("Kinect not connected. Connecting...")
            self.connect_to_kinect()
            
            # Wait a bit for connection
            QtCore.QTimer.singleShot(1000, self._open_camera_stream)
        else:
            self._open_camera_stream()
    
    def _open_camera_stream(self):
        """Helper to open camera stream."""
        if not self.kinect_connected:
            QtWidgets.QMessageBox.warning(self, "Not Connected", 
                                        "Failed to connect to Kinect. Please ensure the C++ application is running.")
            return
        
        if self.camera_window is None:
            self.camera_window = CameraStreamWindow(self.kinect_worker.pipe_handle)
            self.camera_window.closed.connect(self.on_camera_window_closed)
            self.camera_window.show()
            self.camera_window.start_stream()
            self.view_camera_button.setText("Close Camera Stream")
        else:
            self.camera_window.close()
            self.camera_window = None
            self.view_camera_button.setText("View Camera Stream")
    
    def on_camera_window_closed(self):
        """Handle camera window closed."""
        self.camera_window = None
        self.view_camera_button.setText("View Camera Stream")

    def toggle_kinect_recording(self):
        """Toggle Kinect recording on/off."""
        if not self.kinect_connected:
            # Auto-connect if needed
            self.connect_to_kinect()
            QtCore.QTimer.singleShot(1000, self._toggle_kinect_after_connect)
            return
        
        self._do_toggle_kinect()
    
    def _toggle_kinect_after_connect(self):
        """Helper to toggle after auto-connect."""
        if self.kinect_connected:
            self._do_toggle_kinect()
        else:
            QtWidgets.QMessageBox.warning(self, "Not Connected", 
                                        "Failed to connect to Kinect.")
    
    def _do_toggle_kinect(self):
        """Actually toggle Kinect recording."""
        if self.kinect_recording:
            logging.info("Stopping Kinect recording...")
            QtCore.QMetaObject.invokeMethod(self.kinect_worker, "stop_kinect_recording", 
                                          QtCore.Qt.QueuedConnection)
        else:
            logging.info("Starting Kinect recording...")
            QtCore.QMetaObject.invokeMethod(self.kinect_worker, "start_kinect_recording", 
                                          QtCore.Qt.QueuedConnection)

    def start_recording_all(self):
        """Toggle recording on both BioPoint and Kinect."""
        if not self.acquisition_running:
            QtWidgets.QMessageBox.warning(self, "Cannot Record", 
                                        "BioPoint acquisition must be running to record.")
            return
        self.toggle_recording()     
        self.toggle_kinect_recording()

    def on_kinect_connected_changed(self, connected):
        """Handle Kinect connection status change."""
        self.kinect_connected = connected
        if connected:
            self.kinect_status_label.setText("Kinect: Connected")
            self.kinect_status_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.kinect_status_label.setText("Kinect: Disconnected")
            self.kinect_status_label.setStyleSheet("")
            self.kinect_recording = False
            self.kinect_recording_label.setText("Kinect Recording: OFF")
            self.kinect_recording_label.setStyleSheet("")
            
            # Close camera window if open
            if self.camera_window:
                self.camera_window.close()
                self.camera_window = None
        
        self.update_button_states()

    def on_kinect_recording_changed(self, recording):
        """Handle Kinect recording status change."""
        self.kinect_recording = recording
        if recording:
            self.kinect_recording_label.setText("Kinect Recording: ON")
            self.kinect_recording_label.setStyleSheet("color: red; font-weight: bold;")
            self.kinect_record_button.setText("Stop Kinect Recording")
            self.record_all_button.setText("Stop All Recording")
        else:
            self.kinect_recording_label.setText("Kinect Recording: OFF")
            self.kinect_recording_label.setStyleSheet("")
            self.kinect_record_button.setText("Start Kinect Recording")
            if not self.is_recording:
                self.record_all_button.setText("Start All Recording")

    def disconnect_all(self):
        """Disconnect from all devices."""
        self.stop_acquisition()
        time.sleep(0.5)  # Give some time to stop acquisition
        self.disconnect_biopoint()
        self.disconnect_from_kinect()

    def add_marker(self):
        """Add a marker at the current time position."""
        if not self.acquisition_running or self.data_start_time is None:
            QtWidgets.QMessageBox.warning(self, "Cannot Add Marker", 
                                        "Data acquisition must be running to add markers.")
            return
        
        current_time = time.time() - self.data_start_time
        
        label, ok = QtWidgets.QInputDialog.getText(self, "Add Marker", 
                                                f"Enter label for marker at {current_time:.2f}s:",
                                                text=f"M{self.next_marker_number}")
        if ok:
            if not label:
                label = f"M{self.next_marker_number}"
            
            colors = ['r', 'g', 'b', 'y', 'm', 'c']
            color = colors[(self.next_marker_number - 1) % len(colors)]
            
            self.markers.append((current_time, label, color))
            self.next_marker_number += 1
            
            self.add_marker_to_plots(current_time, label, color)
            self.marker_label.setText(f"Markers: {len(self.markers)}")
            
            logging.info(f"Marker added: {label} at {current_time:.2f}s")

    def add_marker_to_plots(self, time_pos, label, color):
        """Add visual marker to all plot widgets."""
        if label not in self.marker_items:
            self.marker_items[label] = []
        
        for plot_widget in self.plot_widgets:
            marker = MarkerItem(time_pos, label, color, movable=True)
            plot_widget.addItem(marker)
            plot_widget.addItem(marker.text_label)
            self.marker_items[label].append(marker)
            marker.sigPositionChanged.connect(lambda m=marker, l=label: self.on_marker_moved(m, l))

    def on_marker_moved(self, marker, label):
        """Handle when a marker is moved."""
        new_time = marker.value()
        
        for i, (time, lbl, color) in enumerate(self.markers):
            if lbl == label:
                self.markers[i] = (new_time, lbl, color)
                break
        
        if label in self.marker_items:
            for m in self.marker_items[label]:
                if m != marker:
                    m.setValue(new_time)
                m.update_label_position()

    def show_marker_editor(self):
        """Show dialog to edit or delete markers."""
        if not self.markers:
            QtWidgets.QMessageBox.information(self, "No Markers", "No markers to edit.")
            return
        
        dialog = MarkerEditorDialog(self.markers, self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            old_markers = self.markers.copy()
            self.markers = dialog.get_markers()
            self.clear_marker_visuals()
            
            for time_pos, label, color in self.markers:
                self.add_marker_to_plots(time_pos, label, color)
            
            self.marker_label.setText(f"Markers: {len(self.markers)}")

    def clear_marker_visuals(self):
        """Remove all visual markers from plots."""
        for label, marker_list in self.marker_items.items():
            for marker in marker_list:
                for plot_widget in self.plot_widgets:
                    plot_widget.removeItem(marker)
                    plot_widget.removeItem(marker.text_label)
        self.marker_items.clear()

    def clear_all_markers(self):
        """Clear all markers."""
        reply = QtWidgets.QMessageBox.question(self, 'Clear Markers', 
                                               "Are you sure you want to clear all markers?",
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                               QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            self.markers.clear()
            self.clear_marker_visuals()
            self.marker_label.setText("Markers: 0")
            logging.info("All markers cleared.")

    def toggle_recording(self):
        """Toggle BioPoint recording on/off."""
        if not self.acquisition_running:
            QtWidgets.QMessageBox.warning(self, "Cannot Record", 
                                        "Data acquisition must be running to record.")
            return
        
        if self.is_recording:
            duration = time.time() - self.recording_start_time if hasattr(self, 'recording_start_time') else 0
            self.is_recording = False
            self.record_button.setText("Start BioPoint Recording")
            if not self.kinect_recording:
                self.record_all_button.setText("Start All Recording")
            self.recording_label.setText("BioPoint Recording: OFF")
            self.recording_label.setStyleSheet("")
            logging.info(f"BioPoint recording stopped. Duration: {duration:.1f} seconds")
        else:
            for key in self.recorded_data:
                self.recorded_data[key].clear()
            
            self.recording_start_time = time.time()
            self.recording_start_offset = time.time() - self.data_start_time
            self.recording_start_sample = self.sample_counts["ecg"]
            self.is_recording = True
            
            self.record_button.setText("Stop BioPoint Recording")
            self.record_all_button.setText("Stop All Recording")
            self.recording_label.setText("BioPoint Recording: ON")
            self.recording_label.setStyleSheet("color: red; font-weight: bold;")
            logging.info(f"BioPoint recording started at {datetime.now().strftime('%H:%M:%S')}")

    def handle_data(self, data):
        """Handles incoming data from the SifiBridgeWorker."""
        timestamp = time.time()
        if self.data_start_time is None:
            self.data_start_time = timestamp

        # EMG data
        if data.get('packet_type') == 'emg' and 'data' in data and isinstance(data['data'], dict) and 'emg' in data['data']:
            emg_data = data['data']['emg']
            self.data_buffers["emg"].extend(emg_data)
            self.sample_counts["emg"] += len(emg_data)
            if self.is_recording:
                self.recorded_data["emg"].extend(emg_data)

        # ECG data (inverted for conventional display)
        elif data.get('packet_type') == 'ecg' and 'data' in data and isinstance(data['data'], dict) and 'ecg' in data['data']:
            ecg_data = [-val for val in data['data']['ecg']]
            self.data_buffers["ecg"].extend(ecg_data)
            self.sample_counts["ecg"] += len(ecg_data)
            if self.is_recording:
                self.recorded_data["ecg"].extend(ecg_data)

        # Temperature data
        elif data.get('packet_type') == 'temperature' and 'data' in data and isinstance(data['data'], dict) and 'temperature' in data['data']:
            temp_data = data['data']['temperature']
            self.data_buffers["temperature"].extend(temp_data)
            self.sample_counts["temperature"] += len(temp_data)
            if self.is_recording:
                self.recorded_data["temperature"].extend(temp_data)

        # EDA data
        elif data.get('packet_type') == 'eda' and 'data' in data and isinstance(data['data'], dict) and 'eda' in data['data']:
            eda_data = data['data']['eda']
            self.data_buffers["eda"].extend(eda_data)
            self.sample_counts["eda"] += len(eda_data)
            if self.is_recording:
                self.recorded_data["eda"].extend(eda_data)

        # PPG data
        elif data.get('packet_type') == 'ppg' and 'data' in data and isinstance(data['data'], dict):
            ppg_data = data['data']
            num_samples_ppg = max(len(ppg_data.get(key, [])) for key in ['b', 'g', 'r', 'ir'])
            self.sample_counts["ppg"] += num_samples_ppg

            for key, buffer_key in zip(['b', 'g', 'r', 'ir'], ['ppg_blue', 'ppg_green', 'ppg_red', 'ppg_ir']):
                if key in ppg_data:
                    self.data_buffers[buffer_key].extend(ppg_data[key])
                    if self.is_recording:
                        self.recorded_data[buffer_key].extend(ppg_data[key])

        # IMU data
        elif data.get('packet_type') == 'imu' and 'data' in data and isinstance(data['data'], dict):
            imu_data = data['data']
            num_samples_imu = max(len(imu_data.get(key, [])) for key in ['ax', 'ay', 'az'])
            self.sample_counts["imu"] += num_samples_imu

            for key in ['ax', 'ay', 'az']:
                if key in imu_data:
                    self.data_buffers[key].extend(imu_data[key])
                    if self.is_recording:
                        self.recorded_data[key].extend(imu_data[key])

    def update_plots(self):
        """Updates the plots with the latest data."""
        if self.data_start_time is None:
            return
        
        total_elapsed = time.time() - self.data_start_time

        def update_plot(curve, data_buffer, plot_widget, fs, sample_count):
            if data_buffer:
                data_array = np.array(list(data_buffer))
                num_points = len(data_array)

                if num_points > 0:
                    max_time = total_elapsed
                    min_time = max(0, max_time - (num_points / fs))
                    time_vector = np.linspace(min_time, max_time, num_points)
                    
                    curve.setData(time_vector, data_array)
                    plot_widget.enableAutoRange(x=True, y=True)
            else:
                curve.clear()

        update_plot(self.emg_curve, self.data_buffers["emg"], self.emg_plot_widget, self.FS["emg"], self.sample_counts["emg"])
        update_plot(self.ecg_curve, self.data_buffers["ecg"], self.ecg_plot_widget, self.FS["ecg"], self.sample_counts["ecg"])
        update_plot(self.temperature_curve, self.data_buffers["temperature"], self.temperature_plot_widget, self.FS["temperature"], self.sample_counts["temperature"])
        update_plot(self.eda_curve, self.data_buffers["eda"], self.eda_plot_widget, self.FS["eda"], self.sample_counts["eda"])

        # PPG Plot
        ppg_lengths = {color: len(self.data_buffers[f"ppg_{color}"]) for color in ["blue", "green", "red", "ir"]}
        max_ppg_len = max(ppg_lengths.values())

        if max_ppg_len > 0:
            max_time_ppg = total_elapsed
            min_time_ppg = max(0, max_time_ppg - (max_ppg_len / self.FS["ppg"]))
            time_ppg = np.linspace(min_time_ppg, max_time_ppg, max_ppg_len)

            for color in ["blue", "green", "red", "ir"]:
                data_len = ppg_lengths[color]
                curve = getattr(self, f"ppg_{color}_curve")

                if data_len > 0:
                    buffer_array = np.array(list(self.data_buffers[f"ppg_{color}"]))
                    curve.setData(time_ppg[-data_len:], buffer_array)
            self.ppg_plot_widget.enableAutoRange(x=True, y=True)
        else:
            for color in ["blue", "green", "red", "ir"]:
                getattr(self, f"ppg_{color}_curve").clear()

        # IMU Plot
        imu_len = len(self.data_buffers["ax"])
        if imu_len > 0:
            max_time_imu = total_elapsed
            min_time_imu = max(0, max_time_imu - (imu_len / self.FS["imu"]))
            time_imu = np.linspace(min_time_imu, max_time_imu, imu_len)

            imu_ax_data = np.array(list(self.data_buffers["ax"]))
            imu_ay_data = np.array(list(self.data_buffers["ay"]))
            imu_az_data = np.array(list(self.data_buffers["az"]))
            imu_data = np.column_stack([imu_ax_data, imu_ay_data, imu_az_data])

            norm_data_ms2 = np.linalg.norm(imu_data, axis=1)
            norm_data_g = norm_data_ms2 / 9.80665 - 1.0

            self.imu_norm_curve.setData(time_imu, norm_data_g)
            self.imu_norm_plot_widget.enableAutoRange(x=True, y=True)
        else:
            self.imu_norm_curve.clear()

    def _get_next_recording_folder(self, base_dir):
        """Find next recording folder (recording_001, recording_002, ...)"""
        base = Path(base_dir)
        base.mkdir(exist_ok=True)

        i = 1
        while True:
            folder = base / f"recording_{i:03d}"
            if not folder.exists():
                folder.mkdir()
                return folder
            i += 1

    def save_data(self):
        """Save recorded data + metadata + markers into a new recording folder."""
        if not any(self.recorded_data.values()):
            QtWidgets.QMessageBox.warning(self, "No Data", "No data has been recorded yet.")
            return

        # User selects base directory
        base_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Base Directory for Recordings"
        )
        if not base_dir:
            return

        # Create new folder: recording_001, recording_002, ...
        recording_dir = self._get_next_recording_folder(base_dir)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            # Save channel data (CSV)
            for channel, data in self.recorded_data.items():
                if data:
                    filename = recording_dir / f"biopoint_{channel}.csv"

                    # Determine sampling rate
                    if channel in ["emg"]:
                        fs = self.FS["emg"]
                    elif channel in ["ecg"]:
                        fs = self.FS["ecg"]
                    elif channel in ["temperature"]:
                        fs = self.FS["temperature"]
                    elif channel in ["eda"]:
                        fs = self.FS["eda"]
                    elif channel.startswith("ppg"):
                        fs = self.FS["ppg"]
                    else:
                        fs = self.FS["imu"]

                    # Write CSV
                    with open(filename, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(["Time (s)", channel, f"Sampling Rate: {fs} Hz"])
                        for i, value in enumerate(data):
                            time_s = i / fs
                            writer.writerow([time_s, value])

                    logging.info(f"Saved {channel} data to {filename}")

            # Save metadata
            metadata_file = recording_dir / "metadata.json"
            metadata = {
                "timestamp": timestamp_str,
                "recording_start": datetime.fromtimestamp(
                    self.recording_start_time).strftime('%Y-%m-%d %H:%M:%S')
                    if hasattr(self, 'recording_start_time') else "Unknown",
                "sampling_rates": self.FS,
                "total_samples": {k: len(v) for k, v in self.recorded_data.items() if v},
                "duration_seconds": {
                    k: len(v) / self.FS.get(k.split('_')[0], 1)
                    for k, v in self.recorded_data.items() if v
                }
            }

            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            logging.info(f"Saved metadata to {metadata_file}")

            # Save markers
            markers_file = recording_dir / "markers.json"
            marker_list = [
                {
                    "time": time - self.recording_start_offset
                    if hasattr(self, "recording_start_offset") else time,
                    "label": label,
                    "color": color
                }
                for time, label, color in self.markers
            ] if self.markers else []

            with open(markers_file, 'w') as f:
                json.dump({"markers": marker_list, "total_markers": len(marker_list)}, f, indent=2)

            logging.info(f"Saved markers to {markers_file}")

            QtWidgets.QMessageBox.information(
                self,
                "Saved",
                f"All data saved to:\n{recording_dir}"
            )

        except Exception as e:
            logging.error(f"Error saving data: {e}", exc_info=True)
            QtWidgets.QMessageBox.critical(self, "Error", f"Error saving data: {e}")

    def handle_worker_finished(self):
        """Handles the 'finished' signal from the worker."""
        logging.info("Worker finished.")

    def on_worker_stopped(self):
        """Handles the 'stopped' signal from the worker."""
        logging.info("Acquisition stopped.")
        self.acquisition_running = False
        self.is_recording = False
        self.record_button.setText("Start BioPoint Recording")
        self.recording_label.setText("BioPoint Recording: OFF")
        self.recording_label.setStyleSheet("")
        self.update_button_states()

    def closeEvent(self, event):
        """Handles the window close event."""
        logging.info("Closing application...")

        # Close camera window
        if self.camera_window:
            self.camera_window.close()

        self.worker.stop_acquisition()
        self.worker.disconnect_bridge()
        self.disconnect_from_kinect()
        self.plot_timer.stop()

        if hasattr(self.worker, 'thread') and self.worker.thread() is not None:
            self.worker.thread().quit()
            self.worker.thread().wait()

        if hasattr(self.kinect_worker, 'thread') and self.kinect_worker.thread() is not None:
            self.kinect_worker.thread().quit()
            self.kinect_worker.thread().wait()

        self.worker.deleteLater()
        self.kinect_worker.deleteLater()
        event.accept()

    def connect_default_biopoint(self):
        """Initiates connection to BioPoint using default method."""
        logging.info("Connecting to BioPoint (default)...")
        QtCore.QMetaObject.invokeMethod(self.worker, "connect_default", QtCore.Qt.QueuedConnection)

    def connect_biopoint_mac(self):
        """Gets MAC address from user and initiates connection."""
        mac_address, ok = QtWidgets.QInputDialog.getText(self, "Connect by MAC Address",
                                                         "Enter BioPoint MAC Address:")
        if ok and mac_address:
            logging.info(f"Connecting to BioPoint (MAC: {mac_address})...")
            QtCore.QMetaObject.invokeMethod(self.worker, "connect_mac_address",
                                          QtCore.Qt.QueuedConnection,
                                          QtCore.Q_ARG(str, mac_address))

    def start_acquisition(self):
        """Starts data acquisition."""
        if self.is_connected and not self.acquisition_running:
            logging.info("Starting data acquisition...")

            self.data_start_time = None
            for buffer in self.data_buffers.values():
                buffer.clear()
            for key in self.sample_counts:
                self.sample_counts[key] = 0

            for key in self.recorded_data:
                self.recorded_data[key].clear()

            self.acquisition_running = True
            QtCore.QMetaObject.invokeMethod(self.worker, "start_acquisition", QtCore.Qt.QueuedConnection)
            self.update_button_states()
        elif not self.is_connected:
            QtWidgets.QMessageBox.warning(self, "Not Connected", "Please connect to BioPoint first.")

    def stop_acquisition(self):
        """Stops data acquisition."""
        if self.acquisition_running:
            logging.info("Stopping data acquisition...")
            QtCore.QMetaObject.invokeMethod(self.worker, "stop_acquisition", QtCore.Qt.QueuedConnection)
            self.acquisition_running = False
            self.update_button_states()

    def disconnect_biopoint(self):
        """Disconnects from BioPoint."""
        logging.info("Disconnecting from BioPoint...")
        QtCore.QMetaObject.invokeMethod(self.worker, "disconnect_bridge", QtCore.Qt.QueuedConnection)

    def power_off_biopoint(self):
        """Sends power off command to BioPoint."""
        reply = QtWidgets.QMessageBox.question(self, 'Power Off',
                                               "Are you sure you want to power off the BioPoint?",
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                               QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            logging.info("Powering off BioPoint...")
            QtCore.QMetaObject.invokeMethod(self.worker, "power_off_device", QtCore.Qt.QueuedConnection)

    def on_connected_status_changed(self, connected):
        """Handles changes in the BioPoint connection status."""
        self.is_connected = connected
        self.update_button_states()

        if connected:
            self.status_label.setText("BioPoint: Connected")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")

            self.data_start_time = None
            for buffer in self.data_buffers.values():
                buffer.clear()
            for key in self.sample_counts:
                self.sample_counts[key] = 0
            for key in self.recorded_data:
                self.recorded_data[key].clear()
        else:
            self.status_label.setText("BioPoint: Disconnected")
            self.status_label.setStyleSheet("")

    def update_button_states(self):
        """Updates the enabled/disabled state of buttons."""
        self.start_button.setEnabled(self.is_connected and not self.acquisition_running)
        self.stop_button.setEnabled(self.is_connected and self.acquisition_running)
        self.record_button.setEnabled(self.is_connected and self.acquisition_running)
        self.add_marker_button.setEnabled(self.acquisition_running)
        self.power_off_button.setEnabled(self.is_connected)

        # Kinect buttons - always enabled for auto-connect feature
        self.view_camera_button.setEnabled(True)
        self.kinect_record_button.setEnabled(True)
        self.record_all_button.setEnabled(self.is_connected and self.acquisition_running)


class MarkerEditorDialog(QtWidgets.QDialog):
    """Dialog for editing markers."""

    def __init__(self, markers, parent=None):
        super().__init__(parent)
        self.markers = markers.copy()
        self.init_ui()

    def init_ui(self):
        """Initialize the dialog UI."""
        self.setWindowTitle("Edit Markers")
        self.resize(600, 400)

        layout = QtWidgets.QVBoxLayout()

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Time (s)", "Label", "Color", "Actions"])
        self.table.horizontalHeader().setStretchLastSection(True)

        self.populate_table()

        layout.addWidget(self.table)

        button_layout = QtWidgets.QHBoxLayout()

        ok_button = QtWidgets.QPushButton("OK")
        ok_button.clicked.connect(self.accept)

        cancel_button = QtWidgets.QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)

        button_layout.addStretch()
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def populate_table(self):
        """Populate the table with markers."""
        self.table.setRowCount(len(self.markers))

        for row, (time, label, color) in enumerate(self.markers):
            time_item = QtWidgets.QTableWidgetItem(f"{time:.2f}")
            self.table.setItem(row, 0, time_item)

            label_item = QtWidgets.QTableWidgetItem(label)
            self.table.setItem(row, 1, label_item)

            color_combo = QtWidgets.QComboBox()
            colors = [('Red', 'r'), ('Green', 'g'), ('Blue', 'b'),
                     ('Yellow', 'y'), ('Magenta', 'm'), ('Cyan', 'c')]
            for color_name, color_code in colors:
                color_combo.addItem(color_name, color_code)

            for i, (_, c) in enumerate(colors):
                if c == color:
                    color_combo.setCurrentIndex(i)
                    break

            self.table.setCellWidget(row, 2, color_combo)

            delete_button = QtWidgets.QPushButton("Delete")
            delete_button.clicked.connect(lambda checked, r=row: self.delete_marker(r))
            self.table.setCellWidget(row, 3, delete_button)

    def delete_marker(self, row):
        """Delete a marker."""
        reply = QtWidgets.QMessageBox.question(self, 'Delete Marker',
                                               "Are you sure you want to delete this marker?",
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            self.markers.pop(row)
            self.populate_table()

    def get_markers(self):
        """Get the updated markers."""
        updated_markers = []

        for row in range(self.table.rowCount()):
            time_item = self.table.item(row, 0)
            label_item = self.table.item(row, 1)
            color_combo = self.table.cellWidget(row, 2)

            if time_item and label_item and color_combo:
                try:
                    time = float(time_item.text())
                    label = label_item.text()
                    color = color_combo.currentData()
                    updated_markers.append((time, label, color))
                except ValueError:
                    pass

        return updated_markers