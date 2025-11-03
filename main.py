"""
main.py

The main entry point for the DeepLabCut Live GUI application.

This script handles:
- Critical import checks (PyQt5, DeepLabCut).
- Setting the multiprocessing start method (crucial for stability).
- Initializing the QApplication and the main window (App).
- Handling top-level exceptions to prevent silent crashes.
"""

import sys
import traceback
import multiprocessing as mp
from PyQt5.QtWidgets import QApplication, QMessageBox
from gui.main_window import App

# Attempt to import DeepLabCut to check for availability
try:
    from deeplabcut.core.config import read_config_as_dict
    DLC_AVAILABLE = True
except ImportError:
    DLC_AVAILABLE = False
except Exception as e:
    print(f"An unexpected error occurred while importing DeepLabCut: {e}")
    DLC_AVAILABLE = False


def main():
    """
    Main entry point for the application.
    Initializes and runs the PyQt5 application.
    """
    try:
        if not DLC_AVAILABLE:
            # Show an error message if DLC is not installed
            app = QApplication(sys.argv)
            QMessageBox.critical(
                None,
                "Dependency Error",
                "DeepLabCut could not be imported.\n"
                "Please ensure DeepLabCut is installed correctly in your environment."
            )
            sys.exit(1)

        # 'spawn' is required for CUDA in subprocesses and is safer
        # on macOS and Linux.
        if sys.platform != 'win32':
            try:
                mp.set_start_method("spawn", force=True)
            except RuntimeError as e:
                # This might happen if the method is already set
                print(f"[Main] Warning: Could not force 'spawn' method: {e}")

        app = QApplication(sys.argv)
        win = App()
        win.show()
        sys.exit(app.exec_())

    except Exception as e:
        # Last-resort crash handler if something fails during app execution
        print("\n" + "="*50)
        print("--- AN UNHANDLED ERROR OCCURRED IN main() ---")
        print(f"--- Error Type: {type(e)}")
        print(f"--- Error: {e}")
        print("\n--- Full Traceback: ---")
        print(traceback.format_exc())
        print("="*50 + "\n")
        
        # Try to show a final message box
        try:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Application Error")
            msg.setText("A critical error occurred. The application must close.")
            msg.setDetailedText(traceback.format_exc())
            msg.exec_()
        except Exception:
            pass # Failed to even show a message box
            
        sys.exit(1)

# This guard is crucial for multiprocessing to prevent
# child processes from re-executing the main script.
if __name__ == '__main__':
    main()
