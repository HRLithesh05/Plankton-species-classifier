"""
Test runner for the plankton classifier project.
Runs all test suites and generates coverage reports.
"""

import unittest
import sys
from pathlib import Path

# Add current directory to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def discover_and_run_tests():
    """Discover and run all tests in the tests directory."""
    # Test discovery
    loader = unittest.TestLoader()
    test_dir = project_root / "tests"

    # Discover all tests
    suite = loader.discover(str(test_dir), pattern='test_*.py')

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()

if __name__ == '__main__':
    print("Running Plankton Classifier Test Suite")
    print("=" * 50)

    success = discover_and_run_tests()

    if success:
        print("\n[PASS] All tests passed!")
        exit(0)
    else:
        print("\n[FAIL] Some tests failed!")
        exit(1)