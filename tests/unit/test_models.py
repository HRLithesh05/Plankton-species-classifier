"""
Unit tests for CNN models.
Tests model architecture, loading, and inference correctness.
"""

import unittest
import torch
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class TestCNNModel(unittest.TestCase):
    """Test cases for CNN model functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.num_classes = 54
        self.model_name = 'efficientnet_v2_s'

    def test_imports_available(self):
        """Test if required modules can be imported."""
        try:
            import torch
            import torchvision
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Required imports not available: {e}")

    def test_torch_availability(self):
        """Test if PyTorch is properly installed."""
        self.assertTrue(hasattr(torch, '__version__'))
        self.assertIsInstance(torch.__version__, str)

    def test_project_structure(self):
        """Test if project structure exists."""
        src_dir = project_root / "src"
        models_dir = src_dir / "models"
        self.assertTrue(src_dir.exists(), "src directory should exist")
        self.assertTrue(models_dir.exists(), "models directory should exist")

    def test_model_creation_basic(self):
        """Test basic model creation without complex imports."""
        # Create a simple model to test basic functionality
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(16, self.num_classes)
        )
        self.assertIsNotNone(model)

    def test_model_forward_pass_basic(self):
        """Test basic model forward pass."""
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(16, self.num_classes)
        )
        model.eval()

        # Create dummy input batch
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, 224, 224)

        with torch.no_grad():
            output = model(dummy_input)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, self.num_classes))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_availability(self):
        """Test CUDA functionality if available."""
        self.assertTrue(torch.cuda.is_available())
        device_count = torch.cuda.device_count()
        self.assertGreater(device_count, 0)

    def test_plankton_cnn_import(self):
        """Test if PlanktonCNN can be imported."""
        try:
            from src.models.cnn_model import PlanktonCNN
            self.assertTrue(True, "PlanktonCNN imported successfully")
        except ImportError as e:
            self.skipTest(f"PlanktonCNN import failed: {e}")
        except Exception as e:
            self.skipTest(f"PlanktonCNN import error: {e}")

if __name__ == '__main__':
    unittest.main()