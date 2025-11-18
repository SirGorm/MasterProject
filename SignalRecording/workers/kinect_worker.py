from PyQt5 import QtCore, QtWidgets, QtGui
from datetime import datetime

import csv
import cv2
import numpy as np
import logging
import win32file
import sifi_bridge_py as sbp
import pyqtgraph as pg
import json
class KinectControlWorker(QtCore.QObject):
    """Worker class for controlling Kinect recording via named pipe."""

    connected_changed = QtCore.pyqtSignal(bool)
    recording_status_changed = QtCore.pyqtSignal(bool)
    
    def __init__(self):
        super().__init__()
        self.pipe_handle = None
        self.is_connected = False
    
    @QtCore.pyqtSlot()
    def connect_to_kinect(self):
        """Connect to the Kinect C++ application via named pipe."""
        try:
            logging.info("Connecting to Kinect named pipe...")
            self.pipe_handle = win32file.CreateFile(
                "\\\\.\\pipe\\mynamedpipe",
                win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                0, None,
                win32file.OPEN_EXISTING,
                0, None)
            
            self.is_connected = True
            self.connected_changed.emit(True)
            logging.info("Connected to Kinect pipe successfully!")
            
        except Exception as e:
            logging.error(f"Failed to connect to Kinect pipe: {e}")
            logging.error("Make sure the C++ Kinect application is running first.")
            self.is_connected = False
            self.connected_changed.emit(False)
    
    @QtCore.pyqtSlot()
    def disconnect_from_kinect(self):
        """Disconnect from the Kinect pipe."""
        if self.pipe_handle and self.is_connected:
            try:
                win32file.CloseHandle(self.pipe_handle)
                self.pipe_handle = None
                self.is_connected = False
                self.connected_changed.emit(False)
                logging.info("Disconnected from Kinect pipe")
            except Exception as e:
                logging.error(f"Error disconnecting from Kinect: {e}")
    
    @QtCore.pyqtSlot()
    def start_kinect_recording(self):
        """Send START_RECORDING command to Kinect."""
        if not self.is_connected or not self.pipe_handle:
            logging.warning("Cannot start Kinect recording: Not connected to pipe")
            return
        
        try:
            command = "START_RECORDING"
            win32file.WriteFile(self.pipe_handle, command.encode())
            logging.info("Sent START_RECORDING command to Kinect")
            self.recording_status_changed.emit(True)
            
            # Read response to keep pipe alive
            try:
                result, data = win32file.ReadFile(self.pipe_handle, 1024)
            except:
                pass
            
        except Exception as e:
            logging.error(f"Error sending START_RECORDING command: {e}")
    
    @QtCore.pyqtSlot()
    def stop_kinect_recording(self):
        """Send STOP_RECORDING command to Kinect."""
        if not self.is_connected or not self.pipe_handle:
            logging.warning("Cannot stop Kinect recording: Not connected to pipe")
            return
        
        try:
            command = "STOP_RECORDING"
            win32file.WriteFile(self.pipe_handle, command.encode())
            logging.info("Sent STOP_RECORDING command to Kinect")
            self.recording_status_changed.emit(False)
            
            # Read response to keep pipe alive
            try:
                result, data = win32file.ReadFile(self.pipe_handle, 1024)
            except:
                pass
            
        except Exception as e:
            logging.error(f"Error sending STOP_RECORDING command: {e}")

