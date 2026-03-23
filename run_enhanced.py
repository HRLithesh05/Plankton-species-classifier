"""
Enhanced Plankton Classifier Launcher
Professional UI with working ML functionality
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
    print("=" * 70)
    print("🔬 PLANKTON SPECIES CLASSIFIER - ENHANCED VERSION 🌊")
    print("=" * 70)
    print("✨ Professional UI with confidence visualization")
    print("📊 Interactive charts with confidence numbers on bars")
    print("🎯 Clean layout following Streamlit reference design")
    print("⚡ Full ML functionality with working predictions")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("🚀 Starting enhanced Flask application...")
    print("🌐 Server: http://localhost:5000")
    print("🎨 Features: Confidence numbers, clean UI, theme toggle")
    print("Press Ctrl+C to stop the server")
    print("=" * 70)

    # Start browser in a separate thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()

    # Start Flask app
    try:
        app.run(debug=False, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        print("Thanks for using the Enhanced Plankton Classifier!")