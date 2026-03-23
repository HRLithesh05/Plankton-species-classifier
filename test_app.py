"""
Test the simplified Flask application
"""

print("Testing Flask application startup...")

try:
    from flask_app import app, model, idx_to_class
    print("Flask app imported successfully")

    if model is not None:
        print("Model loaded successfully")
        print(f"Number of classes: {len(idx_to_class) if idx_to_class else 'Unknown'}")

        # Test model info endpoint
        with app.test_client() as client:
            response = client.get('/api/model-info')
            print(f"Model info API status: {response.status_code}")
            if response.status_code == 200:
                data = response.get_json()
                print(f"Model available: {data.get('available', False)}")
                print(f"Classes: {data.get('classes', 'Unknown')}")

            # Test main route
            response = client.get('/')
            print(f"Main route status: {response.status_code}")

        print("All tests passed! Ready to run the application.")
        print("Run: python run_simple.py")

    else:
        print("ERROR: Model failed to load")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()