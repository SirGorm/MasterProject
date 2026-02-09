import sys
import os
import logging

# Setup logging to file only (avoid console which may cause Rust panics)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('signalrecording.log')]
)

# Setup safe stdio FIRST, before any other imports that might use Rust libraries
def setup_safe_stdio():
    """
    Ensure stdout/stderr file descriptors are valid.
    This prevents Rust libraries from panicking when trying to write to stdio.
    """
    try:
        # Check if stdout is valid by trying to get its fileno
        if sys.stdout is None:
            sys.stdout = open(os.devnull, 'w')
        else:
            try:
                sys.stdout.fileno()
            except (OSError, ValueError):
                sys.stdout = open(os.devnull, 'w')

        if sys.stderr is None:
            sys.stderr = open(os.devnull, 'w')
        else:
            try:
                sys.stderr.fileno()
            except (OSError, ValueError):
                sys.stderr = open(os.devnull, 'w')

        # On Windows, ensure the file descriptors 1 and 2 are valid
        if sys.platform == 'win32':
            import msvcrt
            try:
                msvcrt.get_osfhandle(1)
            except (OSError, ValueError):
                devnull_fd = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull_fd, 1)
                os.close(devnull_fd)

            try:
                msvcrt.get_osfhandle(2)
            except (OSError, ValueError):
                devnull_fd = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull_fd, 2)
                os.close(devnull_fd)

    except Exception:
        pass  # Ignore errors during stdio setup

# Call immediately before any Rust library imports
setup_safe_stdio()

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
