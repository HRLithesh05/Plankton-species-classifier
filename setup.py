"""
Setup script for Plankton Species Classifier package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "Plankton Species Classifier using deep learning and traditional ML approaches"

# Define requirements (matching requirements.txt exactly)
base_requirements = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "torchaudio>=2.0.0",
    "scikit-learn>=1.3.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "Pillow>=10.0.0",
    "tqdm>=4.65.0",
    "joblib>=1.3.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
]

web_requirements = [
    "streamlit>=1.28.0",
    "plotly>=5.17.0",
    "Flask>=3.0.0",
    "Werkzeug>=3.0.0",
    "requests>=2.28.0",
]

advanced_requirements = [
    "opencv-python>=4.8.0",
    "scipy>=1.7.0",
    "scikit-image>=0.21.0",
]

dev_requirements = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=22.0.0",
    "flake8>=5.0.0",
    "mypy>=0.991",
]

all_requirements = base_requirements + web_requirements + advanced_requirements + dev_requirements

setup(
    name="plankton-classifier",
    version="2.0.0",
    author="Plankton Classifier Team",
    description="CNN vs Traditional ML for Plankton Species Classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=base_requirements,
    extras_require={
        "web": web_requirements,
        "advanced": advanced_requirements,
        "dev": dev_requirements,
        "all": all_requirements,
    },
    entry_points={
        "console_scripts": [
            "plankton-train=src.training.train_cnn:main",
            "plankton-predict=src.utils.predict:main",
            "plankton-evaluate=src.utils.evaluate:main",
            "plankton-web=src.web.app:main",
        ],
    },
)