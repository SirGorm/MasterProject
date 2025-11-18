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

class SifiBridgeWorker(QtCore.QObject):
    """Worker class for handling data acquisition from the SifiBridge device in a separate thread."""

    data_received = QtCore.pyqtSignal(dict)
    finished = QtCore.pyqtSignal()
    stopped = QtCore.pyqtSignal()
    connected = QtCore.pyqtSignal(bool)

    def __init__(self, device_type=sbp.DeviceType.BIOPOINT_V1_3):
        super().__init__()
        self.sb = sbp.SifiBridge()
        self.device_type = device_type
        self.running = False
        self.is_connected = False

    @QtCore.pyqtSlot()
    def connect_default(self):
        """Attempts to connect to the BioPoint device using the default method."""
        self.connect_bridge()

    @QtCore.pyqtSlot(str)
    def connect_mac_address(self, mac_address):
        """Attempts to connect to the BioPoint device using a specified MAC address."""
        self.connect_bridge(mac_address=mac_address)

    def connect_bridge(self, mac_address=None):
        """Establishes a connection to the SifiBridge device."""
        while not self.is_connected:
            try:
                if mac_address:
                    success = self.sb.connect(self.device_type, address=mac_address)
                else:
                    success = self.sb.connect(self.device_type)

                if success:
                    logging.info(f"SifiBridge connected (MAC: {mac_address if mac_address else 'Default'})")
                    self.is_connected = True
                    self.connected.emit(True)
                    return True

                logging.warning(f"Failed to connect (MAC: {mac_address if mac_address else 'Default'}). Retrying...")

            except Exception as e:
                logging.error(f"Connection error: {e}", exc_info=True)

            time.sleep(1)

        return False

    def configure_bridge(self):
        """Configures the SifiBridge device to stream the desired data channels."""
        try:
            self.sb.set_channels(ecg=True, emg=True, eda=True, imu=True, ppg=True)
            self.sb.set_ble_power(sbp.BleTxPower.HIGH)
            self.sb.set_filters(True)
            self.sb.configure_ecg(bandpass_freqs=(0, 20))
            self.sb.configure_emg(bandpass_freqs=(20, 300), notch_freq='on50')
            logging.info("SifiBridge channels configured.")
            return True
        except Exception as e:
            logging.error(f"Configuration error: {e}", exc_info=True)
            return False

    @QtCore.pyqtSlot()
    def start_acquisition(self):
        """Starts the data acquisition process."""
        if not self.is_connected:
            logging.warning("Cannot start acquisition: Not connected.")
            return

        if not self.configure_bridge():
            return

        self.running = True
        try:
            self.sb.start()
            logging.info("Data acquisition started.")
            self.acquire_data()
        except Exception as e:
            logging.error(f"Acquisition error: {e}", exc_info=True)
        finally:
            self.stopped.emit()

    def acquire_data(self):
        """Continuously acquires data from the SifiBridge device until stopped."""
        while self.running:
            try:
                data = self.sb.get_data()
                if data:
                    self.data_received.emit(data)

                QtWidgets.QApplication.processEvents()
                time.sleep(0.005)

            except Exception as e:
                logging.error(f"Data acquisition error: {e}", exc_info=True)
                self.running = False
                break

        logging.info("Data acquisition loop finished.")

    @QtCore.pyqtSlot()
    def stop_acquisition(self):
        """Stops the data acquisition process."""
        logging.info("Stopping data acquisition...")
        self.running = False
        try:
            self.sb.stop()
            logging.info("Data acquisition stopped.")
        except Exception as e:
            logging.error(f"Error stopping acquisition: {e}", exc_info=True)

    @QtCore.pyqtSlot()
    def disconnect_bridge(self):
        """Disconnects from the SifiBridge device."""
        if self.is_connected:
            try:
                logging.info("Disconnecting SifiBridge...")
                self.sb.disconnect()
                self.is_connected = False
                self.connected.emit(False)
                logging.info("SifiBridge disconnected.")
            except Exception as e:
                logging.error(f"Disconnection error: {e}", exc_info=True)
        else:
            logging.info("SifiBridge already disconnected.")

    @QtCore.pyqtSlot()
    def power_off_device(self):
        """Sends a power-off command to the BioPoint device."""
        if self.is_connected:
            try:
                logging.info("Sending power off command...")
                self.sb.send_command(sbp.DeviceCommand.POWER_OFF)
                logging.info("Power off command sent.")
                self.disconnect_bridge()
            except AttributeError as e:
                logging.error(f"Power off not supported: {e}")
            except Exception as e:
                logging.error(f"Power off error: {e}", exc_info=True)
        else:
            logging.info("Cannot power off: Not connected.")