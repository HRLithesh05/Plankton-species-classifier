# 🔬 Plankton Species Classifier

**An AI-powered system for automated identification of microscopic marine organisms using deep learning and traditional machine learning approaches.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

This project implements and compares multiple approaches to plankton species classification:
- **Deep Learning**: EfficientNetV2-S CNN with transfer learning
- **Traditional ML**: SVM and Random Forest with engineered features
- **Web Interface**: Professional Streamlit and Flask applications
- **Research Platform**: Comprehensive evaluation and visualization tools

### 🏆 Performance Metrics
- **Top-1 Accuracy**: 75.5%
- **Top-5 Accuracy**: 96.4%
- **Species Coverage**: 54 plankton classes
- **Dataset Size**: 20,000+ microscopic images
- **Inference Speed**: <200ms per prediction

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd Plankton-species-classifier

# Install dependencies
pip install -e ".[web]"  # For web interface
# OR
pip install -e ".[all]"  # For all features including advanced processing
```

### 2. Launch Web Application

```bash
# Streamlit Interface (Recommended)
streamlit run src/web/app.py

# Flask Interface (Alternative)
python src/web/flask_app.py
```

Open your browser to `http://localhost:8501` (Streamlit) or `http://localhost:5000` (Flask)

### 3. Quick Prediction

```bash
# Command line prediction
python src/utils/predict.py --image path/to/plankton_image.jpg

# Evaluate model performance
python src/utils/evaluate.py --model cnn
```

## 📁 Project Structure

```
Plankton-species-classifier/
├── src/                        # Source code
│   ├── models/                # Model definitions (CNN, traditional ML)
│   ├── data/                  # Dataset handling and preprocessing
│   ├── training/              # Training pipelines
│   ├── web/                   # Web applications (Streamlit, Flask)
│   └── utils/                 # Configuration, evaluation, prediction
├── tests/                     # Comprehensive test suite
│   ├── unit/                  # Model and component tests
│   ├── integration/           # End-to-end workflow tests
│   ├── performance/           # Accuracy and speed benchmarks
│   └── ui/                    # Frontend interface tests
├── docs/                      # Detailed documentation
├── configs/                   # Configuration files
├── outputs/                   # Generated models, logs, results
├── 2014_clean/               # Processed dataset (54 species)
└── setup.py                  # Package installation
```

## 🤖 Models & Performance

### CNN Models (Deep Learning)
| Model | Top-1 Acc | Top-5 Acc | Parameters | Training Time |
|-------|-----------|-----------|------------|---------------|
| **EfficientNetV2-S** | **75.5%** | **96.4%** | 21.5M | 1-2 hours |
| EfficientNetV2-M | 78.2%* | 97.1%* | 54.1M | 3-4 hours |

### Traditional ML Models
| Model | Top-1 Acc | Top-5 Acc | Training Time | Inference Speed |
|-------|-----------|-----------|---------------|-----------------|
| Random Forest | 60-70% | 80-88% | 2-4 hours | Fast (CPU) |
| SVM (RBF) | 55-65% | 75-85% | 4-8 hours | Medium |

*Performance may vary based on dataset size and configuration

## 🌐 Web Interfaces

### 🎨 Streamlit Application
- **Modern Design**: Professional glass morphism UI with dark/light themes
- **Real-time Classification**: Upload images or paste URLs for instant predictions
- **Interactive Visualizations**: Top-5 predictions with confidence charts
- **Model Statistics**: Performance metrics and training information
- **Responsive Design**: Works seamlessly on desktop and mobile

### ⚡ Flask Application
- **Production Ready**: Robust backend with advanced image processing
- **API Endpoints**: RESTful API for programmatic access
- **Batch Processing**: Handle multiple images efficiently
- **Advanced Features**: Image enhancement and quality assessment

## 📊 Dataset Information

### WHOI Plankton 2014 Dataset
- **Source**: Woods Hole Oceanographic Institution
- **Total Images**: 20,644 high-quality microscopic images
- **Species**: 54 distinct plankton classes (filtered from original 94)
- **Format**: 256x256 PNG images, preprocessed and cleaned
- **Split**: 70% training, 15% validation, 15% test

### Data Preprocessing Pipeline
- Corrupted image detection and removal
- Duplicate detection using image hashing
- Class balancing (20-2000 samples per class)
- Quality assessment and enhancement
- Standardized format and size normalization

## 🏃‍♂️ Training Your Own Models

### CNN Training
```bash
# Quick training (1-2 hours on RTX 4060 Ti)
python src/training/train_fast.py

# Optimized training (4-6 hours, higher accuracy)
python src/training/train_optimized.py

# Google Colab optimized
python src/training/train_colab.py
```

### Traditional ML Training
```bash
# Train both SVM and Random Forest
python src/training/train_traditional.py

# Train specific model
python src/training/train_traditional.py --model svm
```

### Custom Configuration
Edit `src/utils/config.py` to customize:
- Model architecture and hyperparameters
- Data augmentation strategies
- Training schedules and optimization
- Class balancing approaches

## 🧪 Testing

Run the comprehensive test suite:
```bash
# All tests
python run_tests.py

# Unit tests only
python -m pytest tests/unit/

# Performance benchmarks
python -m pytest tests/performance/
```

## 📖 Detailed Documentation

- **[Installation Guide](docs/INSTALLATION.md)** - Detailed setup instructions
- **[API Reference](docs/API.md)** - Complete API documentation
- **[Training Guide](docs/TRAINING.md)** - Model training walkthrough
- **[Deployment Guide](docs/DEPLOYMENT.md)** - Production deployment options
- **[Contributing](docs/CONTRIBUTING.md)** - Development guidelines

## 🚀 Advanced Features & Future Enhancements

### Planned Enhancements (See [Enhancement Plan](https://github.com/path/to/plan))
- **Vision Transformers**: Advanced attention mechanisms for microscopic details
- **Ensemble Methods**: Multi-model predictions for improved accuracy
- **Research Platform**: Collaborative annotation and species database integration
- **Production Deployment**: Docker, Kubernetes, and cloud deployment
- **Mobile Application**: Cross-platform mobile interface

### Research Applications
- Marine biodiversity monitoring
- Ecosystem health assessment
- Climate change impact studies
- Automated taxonomic classification
- Educational and research collaboration

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/CONTRIBUTING.md) for details on:
- Development workflow
- Code style guidelines
- Testing requirements
- Pull request process

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **WHOI-Plankton Dataset**: Woods Hole Oceanographic Institution
- **EfficientNet**: Google Research for the base architecture
- **PyTorch**: Facebook AI Research for the deep learning framework
- **Marine Biology Community**: For domain expertise and validation

## 📧 Contact

For questions, issues, or collaboration opportunities:
- **Issues**: [GitHub Issues](https://github.com/path/to/issues)
- **Documentation**: [Project Wiki](https://github.com/path/to/wiki)
- **Research Collaboration**: [Contact Form](mailto:research@example.com)

---

## 🎯 Quick Navigation

| Task | Command | Documentation |
|------|---------|---------------|
| **Launch App** | `streamlit run src/web/app.py` | [Web Interface](docs/WEB_INTERFACE.md) |
| **Train Model** | `python src/training/train_fast.py` | [Training Guide](docs/TRAINING.md) |
| **Run Tests** | `python run_tests.py` | [Testing Guide](docs/TESTING.md) |
| **Make Predictions** | `python src/utils/predict.py --image photo.jpg` | [API Reference](docs/API.md) |
| **Evaluate Models** | `python src/utils/evaluate.py` | [Evaluation Guide](docs/EVALUATION.md) |

**Start classifying plankton in under 2 minutes!** 🔬✨