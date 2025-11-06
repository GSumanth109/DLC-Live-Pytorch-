"""
Main entry point for the DLC Live application.

This script is responsible for:
1.  Verifying that critical dependencies (like DeepLabCut) are importable.
2.  Setting up the correct multiprocessing start method (using "spawn").
3.  Initializing the QApplication.
4.  Creating and showing the main application window (App from gui.main_window).
5.  Executing the application's main event loop.
6.  Providing robust error catching and display for critical failures during startup.
"""

import sys
import traceback
import time
import multiprocessing as mp

# --- PyQt5 Imports ---
# We wrap this in a try...except to catch early import errors
try:
    from PyQt5.QtWidgets import QApplication, QMessageBox
except ImportError as e:
    print("="*50)
    print("FATAL ERROR: PyQt5 could not be imported.")
    print(f"Error: {e}")
    print("Please ensure PyQt5 is installed correctly in your environment.")
    print("Try running: pip install PyQt5")
    print("="*50)
    time.sleep(10)
    sys.exit(1)

# --- Application-Specific Imports ---

# Check for DeepLabCut availability
try:
    # We import a specific function to test if DLC is installed
    from deeplabcut.core.config import read_config_as_dict
    DLC_AVAILABLE = True
    print("[Main] DeepLabCut found.")
except ImportError:
    print("[Main] WARNING: DeepLabCut not found.")
    DLC_AVAILABLE = False
except Exception as e:
    print(f"[Main] WARNING: DeepLabCut import failed with an unexpected error: {e}")
    DLC_AVAILABLE = False

# Import the main application window
try:
    from gui.main_window import App
except ImportError as e:
    # This catches errors in main_window.py OR its own imports
    print("="*50)
    print("FATAL ERROR: Could not import the main application window (App).")
    print(f"Error: {e}")
    print("This might be due to an error in 'gui/main_window.py' or one of its imports (e.g., utils, processing).")
    print("\n--- Full Traceback: ---")
    print(traceback.format_exc())
    print("="*50)
    time.sleep(10)
    sys.exit(1)


def main():
    """
    Main function to initialize and run the PyQt application.
    """
    print("[Main] Starting application...")
    
    # This `try...except` block catches any errors that happen *after*
    # the application object is created.
    try:
        # 1. Check for DLC dependency
        if not DLC_AVAILABLE:
            print("[Main] DLC_AVAILABLE is False. Showing error box.")
            # Need a temporary QApplication to show a message box
            app = QApplication(sys.argv) 
            QMessageBox.critical(
                None, 
                "Dependency Error", 
                "DeepLabCut could not be imported.\n"
                "Please ensure DeepLabCut is installed correctly in your environment."
            )
            print("[Main] Exiting due to missing DLC.")
            sys.exit(1)
        
        print("[Main] All dependencies available.")

        # 2. Set Multiprocessing Start Method
        # This is CRITICAL for stability on macOS and Linux.
        # "spawn" creates a fresh, new process, avoiding issues with
        # shared state or libraries (like CUDA) from the parent process.
        # Windows defaults to "spawn", so this is mainly for other OSes.
        if sys.platform != 'win32':
            print(f"[Main] Setting multiprocessing start method to 'spawn' (OS: {sys.platform})...")
            try:
                # 'force=True' is used in case it was set differently before
                mp.set_start_method("spawn", force=True)
                print("[Main] Multiprocessing start method set to 'spawn'.")
            except RuntimeError as e:
                print(f"[Main] Warning: Could not force 'spawn' method: {e}")
        else:
            print("[Main] On Windows, default 'spawn' method is used.")
        
        # 3. Initialize QApplication
        print("[Main] Initializing QApplication...")
        app = QApplication(sys.argv)
        print("[Main] QApplication initialized.")
        
        # 4. Initialize the Main Window
        print("[Main] Creating main window (App)...")
        win = App()
        print("[Main] Main window (App) created.")
        
        # 5. Show the window and run the app
        print("[Main] Showing window and starting event loop (app.exec_)...")
        win.show()
        sys.exit(app.exec_())

    except Exception as e:
        # Final catch-all for any unhandled exceptions during runtime
        print("\n" + "="*50)
        print("--- [Main] AN UNHANDLED ERROR OCCURRED ---")
        print(f"--- Error Type: {type(e)}")
        print(f"--- Error: {e}")
        print("\n--- Full Traceback: ---")
        print(traceback.format_exc())
        print("="*50 + "\n")
        
        # Try to show an error to the user
        try:
            QMessageBox.critical(
                None, 
                "Application Error", 
                "An unhandled error occurred and the application must close.\n\n"
                f"Error: {e}\n\n"
                "Please check the console output for details."
            )
        except Exception:
            pass # Failed to show GUI error, console print will have to do

        print("The application will close in 10 seconds...")
        time.sleep(10)
        sys.exit(1)

# This guard is essential for multiprocessing.
# When a new process is "spawned", it re-imports this script.
# This 'if' block ensures that main() is only called when the script
# is executed directly (e.g., `python main.py`), not when it's imported.
if __name__ == '__main__':
    print("[Main] Script executed directly (`__name__ == '__main__`). Calling main().")
    main()
