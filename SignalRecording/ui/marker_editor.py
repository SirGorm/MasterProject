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