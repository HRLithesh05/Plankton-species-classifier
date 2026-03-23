# 🚀 Installation Guide

## System Requirements

### Hardware
- **CPU**: Multi-core processor (Intel i5+ or AMD Ryzen 5+ recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB free space for project + dataset
- **GPU**: NVIDIA GPU with 4GB+ VRAM (recommended for training)

### Software
- **Python**: 3.8, 3.9, 3.10, or 3.11
- **Operating System**: Windows 10+, macOS 10.15+, or Ubuntu 18.04+
- **Git**: For version control and repository management

## 📦 Installation Methods

### Method 1: Quick Install (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd Plankton-species-classifier

# Install with web interface support
pip install -e ".[web]"

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### Method 2: Full Installation (All Features)

```bash
# Install all features including advanced processing
pip install -e ".[all]"

# This includes:
# - Base ML functionality
# - Web interfaces (Streamlit, Flask)
# - Advanced image processing (OpenCV, scikit-image)
# - Development tools (testing, linting)
```

### Method 3: Development Setup

```bash
# Clone and enter directory
git clone <repository-url>
cd Plankton-species-classifier

# Create virtual environment (recommended)
python -m venv plankton-env

# Activate virtual environment
# On Windows:
plankton-env\Scripts\activate
# On macOS/Linux:
source plankton-env/bin/activate

# Install in development mode
pip install -e ".[dev]"
```

## 🔧 Environment-Specific Instructions

### Windows

#### Prerequisites
```bash
# Install Visual Studio Build Tools (for some packages)
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Install Python from python.org or Microsoft Store
# Ensure "Add Python to PATH" is checked
```

#### GPU Setup (NVIDIA)
```bash
# Install CUDA 11.8 or 12.1 from NVIDIA website
# Verify CUDA installation
nvcc --version

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### macOS

#### Prerequisites
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Homebrew (optional but recommended)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python via Homebrew
brew install python@3.11
```

#### Note on Apple Silicon (M1/M2)
```bash
# PyTorch has native Apple Silicon support
pip install torch torchvision torchaudio

# For some packages, you may need:
conda install -c pytorch pytorch torchvision torchaudio
```

### Linux (Ubuntu/Debian)

#### Prerequisites
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python and development tools
sudo apt install python3 python3-pip python3-venv python3-dev

# Install system dependencies
sudo apt install build-essential libssl-dev libffi-dev
```

#### GPU Setup (NVIDIA)
```bash
# Install NVIDIA drivers
sudo apt install nvidia-driver-xxx  # Replace xxx with version

# Install CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install cuda

# Install cuDNN (optional, for optimization)
# Follow NVIDIA cuDNN installation guide
```

## 🐳 Docker Installation

### Option 1: Pre-built Image
```bash
# Pull and run the pre-built image
docker pull plankton-classifier:latest
docker run -p 8501:8501 plankton-classifier:latest
```

### Option 2: Build from Source
```dockerfile
# Dockerfile (already included in repository)
FROM python:3.10-slim

WORKDIR /app
COPY . .

# Install dependencies
RUN pip install -e ".[web]"

# Expose ports
EXPOSE 8501 5000

# Default command (Streamlit)
CMD ["streamlit", "run", "src/web/app.py", "--server.address", "0.0.0.0"]
```

```bash
# Build and run
docker build -t plankton-classifier .
docker run -p 8501:8501 plankton-classifier
```

### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'
services:
  plankton-web:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./outputs:/app/outputs
      - ./2014_clean:/app/2014_clean
    environment:
      - PYTHONPATH=/app
```

```bash
docker-compose up
```

## ☁️ Cloud Environment Setup

### Google Colab
```python
# In a Colab notebook
!git clone <repository-url>
%cd Plankton-species-classifier

# Install dependencies
!pip install -e ".[web]"

# For Streamlit in Colab
!npm install -g localtunnel
```

### Kaggle Kernels
```bash
# In Kaggle notebook
import os
os.chdir('/kaggle/working')

!git clone <repository-url>
%cd Plankton-species-classifier
!pip install -e .
```

### AWS SageMaker / EC2
```bash
# On EC2 instance
sudo yum update -y
sudo yum install -y python3 python3-pip git

git clone <repository-url>
cd Plankton-species-classifier
pip3 install -e ".[all]"
```

## 📊 Dataset Setup

### Download Dataset
```bash
# Dataset is included in repository as 2014_clean.zip
# Extract if needed
unzip 2014_clean.zip

# Verify dataset structure
ls 2014_clean/  # Should show species folders
```

### Custom Dataset
```bash
# To use your own dataset:
# 1. Create folder structure: dataset/species_name/images.jpg
# 2. Update config.py with new DATA_DIR path
# 3. Run preprocessing script
python src/data/preprocess_dataset.py --data-dir /path/to/your/dataset
```

## 🧪 Verification & Testing

### Quick Verification
```bash
# Test basic imports
python -c "
import torch
import torchvision
import streamlit
import flask
import numpy
import PIL
print('✅ All core packages imported successfully')
"

# Test CUDA (if available)
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA devices: {torch.cuda.device_count()}')
"
```

### Run Test Suite
```bash
# Run all tests
python run_tests.py

# Quick functionality test
python tests/unit/test_models.py
```

### Launch Web Interface
```bash
# Test Streamlit interface
streamlit run src/web/app.py

# Test Flask interface
python src/web/flask_app.py
```

## 🔍 Troubleshooting

### Common Issues

#### PyTorch Installation
```bash
# Issue: PyTorch not using GPU
# Solution: Reinstall with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Out of Memory Errors
```bash
# Issue: CUDA OOM during model loading
# Solution: Reduce batch size in config.py
# Edit src/utils/config.py:
CNN_CONFIG = {
    'batch_size': 16,  # Reduce from 32
    'gradient_accumulation_steps': 4  # Increase to maintain effective batch size
}
```

#### Port Conflicts
```bash
# Issue: Port 8501 already in use
# Solution: Use different port
streamlit run src/web/app.py --server.port 8502
```

#### Import Errors
```bash
# Issue: Module not found errors
# Solution: Add current directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or use editable install
pip install -e .
```

### Platform-Specific Issues

#### Windows
- **Long Path Issues**: Enable long path support in Windows settings
- **Visual Studio Build Tools**: Required for some native packages
- **Antivirus Interference**: Exclude project folder from real-time scanning

#### macOS
- **Apple Silicon Compatibility**: Some packages may need specific versions
- **Security Gatekeeper**: May need to allow apps in Security settings
- **Xcode CLI Tools**: Required for native compilations

#### Linux
- **Permission Issues**: Use `sudo` only when necessary, prefer user installs
- **Display Issues**: For GUI applications, ensure X11 forwarding if using SSH
- **Library Dependencies**: Install system packages before Python packages

## 📋 Installation Checklist

- [ ] Python 3.8+ installed and accessible
- [ ] Git installed and configured
- [ ] Repository cloned successfully
- [ ] Virtual environment created (recommended)
- [ ] Dependencies installed via pip
- [ ] CUDA setup completed (if using GPU)
- [ ] Dataset downloaded and extracted
- [ ] Basic functionality verified
- [ ] Web interface launches successfully
- [ ] Test suite passes

## 🆘 Getting Help

If you encounter issues:

1. **Check this guide** for common solutions
2. **Search existing issues** on GitHub
3. **Run diagnostics** with verification commands above
4. **Create new issue** with:
   - Operating system and version
   - Python version
   - Full error message and traceback
   - Steps to reproduce the problem
   - Output of verification commands

## 🔄 Updates & Maintenance

### Keeping Dependencies Updated
```bash
# Check for outdated packages
pip list --outdated

# Update specific packages
pip install --upgrade torch torchvision

# Update all packages (use with caution)
pip install --upgrade -e ".[all]"
```

### Version Management
```bash
# Pin current working versions
pip freeze > requirements-pinned.txt

# Restore from pinned versions
pip install -r requirements-pinned.txt
```