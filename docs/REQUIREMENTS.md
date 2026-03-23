# 📦 Requirements Management Guide

## Overview

The Plankton Species Classifier uses a unified, modular requirements system that allows you to install only the dependencies you need for your specific use case.

## 🎯 Installation Options

### Option 1: Modular Installation (Recommended)

Install only what you need using the setup.py extras system:

```bash
# Basic ML functionality (CNN training, evaluation, prediction)
pip install -e .

# Web interfaces (Streamlit + Flask applications)
pip install -e ".[web]"

# Advanced image processing (OpenCV, scikit-image, SciPy)
pip install -e ".[advanced]"

# Development tools (testing, linting, formatting)
pip install -e ".[dev]"

# Everything (all features)
pip install -e ".[all]"
```

### Option 2: Combined Installation

Install multiple feature sets together:

```bash
# Web interfaces + advanced processing
pip install -e ".[web,advanced]"

# Development setup with all tools
pip install -e ".[web,advanced,dev]"
```

### Option 3: Traditional Requirements File

Install all dependencies at once (legacy method):

```bash
pip install -r requirements.txt
```

### Option 4: Production/CI (Locked Versions)

For reproducible environments with exact versions:

```bash
pip install -r requirements-lock.txt
```

## 🔧 PyTorch with CUDA Support

PyTorch installation depends on your system's CUDA version:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only (not recommended for training)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Check your CUDA version:
```bash
nvidia-smi  # Shows CUDA driver version
nvcc --version  # Shows CUDA toolkit version (if installed)
```

## 📋 Dependency Categories

### Base Dependencies (Always Installed)
- **PyTorch**: Deep learning framework
- **scikit-learn**: Traditional ML algorithms
- **NumPy/Pandas**: Data manipulation
- **Pillow**: Basic image processing
- **Matplotlib/Seaborn**: Data visualization
- **tqdm**: Progress bars

### Web Dependencies (`.[web]`)
- **Streamlit**: Modern web interface
- **Flask**: Production web API
- **Plotly**: Interactive visualizations
- **Requests**: HTTP client for URL images

### Advanced Dependencies (`.[advanced]`)
- **OpenCV**: Advanced image processing
- **SciPy**: Scientific computing
- **scikit-image**: Image analysis algorithms

### Development Dependencies (`.[dev]`)
- **pytest**: Testing framework
- **black**: Code formatting
- **flake8**: Code linting
- **mypy**: Type checking

## 🚀 Quick Start Examples

### For Machine Learning Research
```bash
# Basic ML + advanced image processing
pip install -e ".[advanced]"
python src/training/train_cnn.py
```

### For Web Application Development
```bash
# Web interfaces + all tools
pip install -e ".[web,dev]"
streamlit run src/web/app.py
```

### For Production Deployment
```bash
# Locked versions for reproducibility
pip install -r requirements-lock.txt
python src/web/flask_app.py
```

### For Full Development Setup
```bash
# Everything for contributors
pip install -e ".[all]"
python run_tests.py
```

## 🔄 Updating Dependencies

### Check for Updates
```bash
pip list --outdated
```

### Update to Latest Compatible Versions
```bash
# Reinstall with latest versions
pip install -e ".[all]" --upgrade
```

### Generate New Lock File
```bash
# After updating, create new lock file
pip freeze > requirements-lock.txt
```

## 🐳 Docker Installation

The Docker image includes all dependencies:

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -e ".[all]"
```

## ⚠️ Common Issues & Solutions

### PyTorch CUDA Mismatch
**Problem**: PyTorch not using GPU or CUDA errors
**Solution**:
```bash
# Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"
# Reinstall with correct CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### OpenCV Import Issues
**Problem**:
```python
ImportError: libGL.so.1: cannot open shared object file
```
**Solution**:
```bash
# Ubuntu/Debian
sudo apt-get install libgl1-mesa-glx

# Alternative: Use headless OpenCV
pip install opencv-python-headless
```

### Memory Issues During Installation
**Problem**: Out of memory when installing large packages
**Solution**:
```bash
# Install with no cache
pip install -e ".[all]" --no-cache-dir

# Or install packages individually
pip install torch torchvision
pip install -e ".[web]"
```

## 📊 Dependency Matrix

| Feature | Base | Web | Advanced | Dev |
|---------|------|-----|----------|-----|
| **CNN Training** | ✅ | ✅ | ✅ | ✅ |
| **Traditional ML** | ✅ | ✅ | ✅ | ✅ |
| **Streamlit App** | ❌ | ✅ | ✅ | ✅ |
| **Flask API** | ❌ | ✅ | ✅ | ✅ |
| **Image Enhancement** | Basic | Basic | ✅ | ✅ |
| **Testing** | ❌ | ❌ | ❌ | ✅ |
| **Code Quality** | ❌ | ❌ | ❌ | ✅ |

## 🎯 Recommended Setups

### Research & Experimentation
```bash
pip install -e ".[advanced,dev]"
```
*Includes: ML training, advanced image processing, testing tools*

### Web Application
```bash
pip install -e ".[web]"
```
*Includes: Streamlit/Flask interfaces, visualization*

### Production API
```bash
pip install -r requirements-lock.txt
```
*Includes: Exact versions for reproducible deployment*

### Full Development
```bash
pip install -e ".[all]"
```
*Includes: Everything for contributors and advanced users*