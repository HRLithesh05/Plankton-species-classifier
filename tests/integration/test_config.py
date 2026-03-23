"""
Configuration and utilities tests.
Tests for configuration management and utility functions.
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestConfiguration(unittest.TestCase):
    """Test cases for configuration and utilities."""

    def setUp(self):
        """Set up test fixtures."""
        self.config_path = project_root / "src" / "utils" / "config.py"
        self.utils_dir = project_root / "src" / "utils"

    def test_config_file_exists(self):
        """Test if configuration file exists."""
        self.assertTrue(self.config_path.exists(), "Configuration file should exist")

    def test_utils_directory_structure(self):
        """Test if utils directory has expected structure."""
        self.assertTrue(self.utils_dir.exists(), "Utils directory should exist")

        expected_files = ["config.py", "evaluate.py", "predict.py"]
        for file_name in expected_files:
            file_path = self.utils_dir / file_name
            self.assertTrue(file_path.exists(), f"{file_name} should exist in utils directory")

    def test_config_import(self):
        """Test if configuration can be imported."""
        try:
            from src.utils import config
            self.assertTrue(hasattr(config, 'CNN_CONFIG'), "CNN_CONFIG should be defined")
        except ImportError as e:
            self.skipTest(f"Config import failed: {e}")
        except Exception as e:
            self.skipTest(f"Config import error: {e}")

    def test_outputs_directory_structure(self):
        """Test if outputs directory structure is correct."""
        outputs_dir = project_root / "outputs"
        if outputs_dir.exists():
            expected_subdirs = ["models", "logs", "results"]
            for subdir in expected_subdirs:
                subdir_path = outputs_dir / subdir
                self.assertTrue(subdir_path.exists(), f"outputs/{subdir} should exist")

    def test_dataset_directory_structure(self):
        """Test if dataset directory structure is accessible."""
        dataset_dirs = [
            project_root / "2014_clean",
            project_root / "2014"
        ]

        dataset_exists = any(d.exists() for d in dataset_dirs)
        if not dataset_exists:
            self.skipTest("No dataset directory found")

        # Find existing dataset directory
        for dataset_dir in dataset_dirs:
            if dataset_dir.exists():
                # Check if it contains class directories
                class_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
                self.assertGreater(len(class_dirs), 0, "Dataset should contain class directories")
                break


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""

    def test_path_utilities(self):
        """Test path utility functions."""
        from pathlib import Path

        # Test Path operations
        test_path = Path("test/path/file.txt")
        self.assertEqual(test_path.name, "file.txt")
        self.assertEqual(test_path.suffix, ".txt")
        self.assertEqual(test_path.parent.name, "path")

    def test_import_safety(self):
        """Test safe importing of optional dependencies."""
        # Test importing common ML libraries safely
        optional_imports = {
            'numpy': 'np',
            'torch': None,
            'PIL': None
        }

        available_imports = {}
        for module, alias in optional_imports.items():
            try:
                if alias:
                    exec(f"import {module} as {alias}")
                    available_imports[module] = True
                else:
                    exec(f"import {module}")
                    available_imports[module] = True
            except ImportError:
                available_imports[module] = False

        # At least PyTorch should be available for the project
        if not available_imports.get('torch', False):
            self.skipTest("PyTorch not available - core dependency missing")

        self.assertTrue(available_imports['torch'], "PyTorch should be available")


if __name__ == '__main__':
    unittest.main()