"""
Test package for the plankton classifier project.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test configuration
TEST_DATA_DIR = project_root / "2014_clean"
TEST_MODEL_DIR = project_root / "outputs" / "models"
TEST_CONFIG = {
    'timeout': 30,  # seconds
    'batch_size': 4,
    'num_test_classes': 5
}