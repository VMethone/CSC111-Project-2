"""Launches the Dash web application with automatic browser support and dynamic port selection."""
import webbrowser
import threading
import time
import socket

# --- IMPORTANT ---
from app import app


# === Utility to Find Free Port ===
def get_free_port() -> int:
    """Find and return a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  # Bind to a free port provided by the host
        return s.getsockname()[1]


# --- Configuration ---
HOST = "127.0.0.1"
PORT = get_free_port()
URL = f"http://{HOST}:{PORT}/"
DEBUG_MODE = True


def run_dash_app() -> None:
    """Function to run the Dash app server."""
    print("----- Starting Dash Server -----")
    print(f"Server running on: {URL}")
    print(f"Debug mode: {'On' if DEBUG_MODE else 'Off'}")
    print("Press CTRL+C to stop the server.")
    print("--------------------------------")

    # Disable reloader to avoid signal issues in thread
    app.run(host=HOST, port=PORT, debug=DEBUG_MODE, use_reloader=False)


def open_browser() -> None:
    """Waits a moment then opens the default web browser."""
    try:
        print("Attempting to open browser automatically...")
        time.sleep(1.5)
        webbrowser.open_new_tab(URL)
        print(f"Successfully opened browser tab to {URL}")
    except webbrowser.Error as e:
        print(f"Warning: Could not open browser automatically ({e}).")
        print(f"Please manually navigate to: {URL}")


# --- Main Execution Block ---
if __name__ == '__main__':
    server_thread = threading.Thread(target=run_dash_app, daemon=True)
    server_thread.start()
    open_browser()
    server_thread.join()
    print("----- Script Finished -----")

    import python_ta
    python_ta.check_all(config={
        'extra-imports': ["webbrowser", "threading", "time", "socket", "app"],
        'allowed-io': ["run_dash_app", "open_browser"],
        'max-line-length': 120,
        'disable': ["E9997", "R0913", "R0914", "C9103", "W0621", "E9992", "E1120"]
    })
