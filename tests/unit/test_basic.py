"""
Basic functionality tests.
Simple tests to verify the testing framework is working.
"""

import unittest
from pathlib import Path


class TestBasicFunctionality(unittest.TestCase):
    """Basic functionality test cases."""

    def test_basic_math(self):
        """Test basic mathematical operations."""
        self.assertEqual(2 + 2, 4)
        self.assertEqual(5 * 3, 15)
        self.assertNotEqual(10, 11)

    def test_string_operations(self):
        """Test string operations."""
        test_string = "plankton classifier"
        self.assertIn("plankton", test_string)
        self.assertEqual(len("hello"), 5)
        self.assertTrue("classifier".isalpha())

    def test_project_directory_structure(self):
        """Test project directory exists."""
        project_root = Path(__file__).parent.parent.parent
        self.assertTrue(project_root.exists())

        # Check for important directories
        important_dirs = ["src", "docs", "tests"]
        for dir_name in important_dirs:
            dir_path = project_root / dir_name
            self.assertTrue(dir_path.exists(), f"{dir_name} directory should exist")

    def test_python_version(self):
        """Test Python version compatibility."""
        import sys
        version_info = sys.version_info
        self.assertGreaterEqual(version_info.major, 3)
        self.assertGreaterEqual(version_info.minor, 8)


if __name__ == '__main__':
    unittest.main()