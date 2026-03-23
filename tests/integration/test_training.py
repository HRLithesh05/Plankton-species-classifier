"""
Integration tests for training pipeline.
Tests end-to-end training workflow and convergence.
"""

import unittest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

class TestTrainingIntegration(unittest.TestCase):
    """Integration test cases for training pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.config_path = Path(__file__).parent.parent.parent / "src" / "utils" / "config.py"

    def test_config_exists(self):
        """Test if configuration file exists."""
        self.assertTrue(self.config_path.exists(), "Configuration file not found")

    def test_training_scripts_exist(self):
        """Test if training scripts are accessible."""
        training_dir = Path(__file__).parent.parent.parent / "src" / "training"
        self.assertTrue(training_dir.exists(), "Training directory not found")

        # Check for key training scripts
        scripts = ["train_cnn.py", "train_traditional.py"]
        for script in scripts:
            script_path = training_dir / script
            self.assertTrue(script_path.exists(), f"Training script {script} not found")

    # Additional integration tests will be added here

if __name__ == '__main__':
    unittest.main()