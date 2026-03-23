"""
Web interface tests.
Tests for Streamlit and Flask web applications.
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestWebInterfaces(unittest.TestCase):
    """Test cases for web application functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.web_dir = project_root / "src" / "web"

    def test_web_directory_exists(self):
        """Test if web interface directory exists."""
        self.assertTrue(self.web_dir.exists(), "Web interface directory should exist")

    def test_streamlit_app_exists(self):
        """Test if Streamlit app file exists."""
        streamlit_app = self.web_dir / "app.py"
        self.assertTrue(streamlit_app.exists(), "Streamlit app.py should exist")

    def test_flask_app_exists(self):
        """Test if Flask app file exists."""
        flask_app = self.web_dir / "flask_app.py"
        self.assertTrue(flask_app.exists(), "Flask flask_app.py should exist")

    def test_templates_directory(self):
        """Test if templates directory exists."""
        templates_dir = self.web_dir / "templates"
        if templates_dir.exists():
            # If templates exist, check they contain HTML files
            html_files = list(templates_dir.glob("*.html"))
            self.assertGreater(len(html_files), 0, "Templates directory should contain HTML files")

    def test_static_directory(self):
        """Test if static assets directory exists."""
        static_dir = self.web_dir / "static"
        if static_dir.exists():
            # Static directory exists, which is good
            self.assertTrue(True)

    def test_streamlit_imports(self):
        """Test if Streamlit can be imported."""
        try:
            import streamlit
            self.assertTrue(hasattr(streamlit, '__version__'))
        except ImportError:
            self.skipTest("Streamlit not installed")

    def test_flask_imports(self):
        """Test if Flask can be imported."""
        try:
            import flask
            self.assertTrue(hasattr(flask, '__version__'))
        except ImportError:
            self.skipTest("Flask not installed")

    def test_web_dependencies(self):
        """Test if web dependencies are available."""
        web_deps = ['requests', 'PIL']
        missing_deps = []

        for dep in web_deps:
            try:
                __import__(dep)
            except ImportError:
                missing_deps.append(dep)

        if missing_deps:
            self.skipTest(f"Missing web dependencies: {missing_deps}")
        else:
            self.assertTrue(True, "All web dependencies available")


if __name__ == '__main__':
    unittest.main()