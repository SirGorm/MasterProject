import sys
from PyQt5 import QtWidgets, QtCore
from workers.sifi_worker import SifiBridgeWorker
from workers.kinect_worker import KinectControlWorker
from ui.plot_window import PlotWindow
import sifi_bridge_py as sbp


def main():
    app = QtWidgets.QApplication(sys.argv)

    # Create workers
    worker = SifiBridgeWorker(device_type=sbp.DeviceType.BIOPOINT_V1_3)
    kinect_worker = KinectControlWorker()

    # Create threads
    thread = QtCore.QThread()
    kinect_thread = QtCore.QThread()

    worker.moveToThread(thread)
    kinect_worker.moveToThread(kinect_thread)

    window = PlotWindow(worker, kinect_worker)
    window.show()

    thread.start()
    kinect_thread.start()

    # Cleanup
    worker.finished.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)

    kinect_thread.finished.connect(kinect_thread.deleteLater)

    app.aboutToQuit.connect(window.close)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
