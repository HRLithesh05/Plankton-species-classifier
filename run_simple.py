"""
Simple launcher for the functional Plankton Classifier
"""

import webbrowser
import time
import threading
from flask_app import app

def open_browser():
    """Open browser after a short delay"""
    time.sleep(2)  # Give Flask time to start
    webbrowser.open('http://localhost:5000')

if __name__ == '__main__':
    print("=" * 60)
    print("🔬 PLANKTON SPECIES CLASSIFIER - FUNCTIONAL VERSION")
    print("=" * 60)
    print("✅ Simplified interface focused on core ML functionality")
    print("🚀 Starting Flask application...")
    print("🌐 Server: http://localhost:5000")
    print("⚡ Features: File upload, ML prediction, results display")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)

    # Start browser in a separate thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()

    # Start Flask app
    try:
        app.run(debug=False, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        print("Thanks for using Oceanic Precision!")