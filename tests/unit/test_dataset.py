"""
Unit tests for dataset functionality.
Tests data loading, preprocessing, and augmentation pipeline.
"""

import unittest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

class TestDataset(unittest.TestCase):
    """Test cases for dataset functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.data_dir = Path(__file__).parent.parent.parent / "2014_clean"

    def test_dataset_directory_exists(self):
        """Test if dataset directory exists."""
        self.assertTrue(self.data_dir.exists(), f"Dataset directory {self.data_dir} not found")

    def test_dataset_has_classes(self):
        """Test if dataset directory contains class folders."""
        if self.data_dir.exists():
            class_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
            self.assertGreater(len(class_dirs), 0, "No class directories found in dataset")

    # Additional dataset tests will be added here

if __name__ == '__main__':
    unittest.main()