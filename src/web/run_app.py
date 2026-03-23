"""
Launcher script for the Oceanic Precision Flask Application
"""

import webbrowser
import time
import threading
from flask_app import app

def open_browser():
    """Open browser after a short delay"""
    time.sleep(1.5)  # Give Flask time to start
    webbrowser.open('http://localhost:5000')

if __name__ == '__main__':
    print("=" * 70)
    print("🌊 OCEANIC PRECISION - PREMIUM AI PLANKTON CLASSIFIER 🔬")
    print("=" * 70)
    print("🎨 Enhanced with sophisticated visual effects and animations")
    print("✨ Glass morphism • Neural aesthetics • Premium interactions")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("Starting premium Flask application...")
    print("🌐 Server will be available at: http://localhost:5000")
    print("⌨️  Keyboard shortcuts: Ctrl+U (upload), Escape (reset)")
    print("🎯 Features: Theme toggle, drag & drop, real-time analysis")
    print("Press Ctrl+C to stop the server")
    print("=" * 70)

    # Start browser in a separate thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()

    # Start Flask app
    try:
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        print("Thanks for using Oceanic Precision!")