# 🔬 Plankton Species Classifier

**An advanced AI-powered system for automated identification and classification of microscopic marine organisms using state-of-the-art deep learning techniques.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Project Overview

This project implements a production-ready plankton species classification system that combines cutting-edge deep learning with an intuitive web interface. The system can accurately identify and classify 67 different species of microscopic marine organisms from digital images.

### 🏆 Current Performance Highlights
- **Model Accuracy**: 89.51% (approach1_final_model)
- **Architecture**: EfficientNet-B2 with progressive training
- **Species Coverage**: 67 distinct plankton classes
- **Training Approach**: 3-stage progressive learning (Foundation → Refinement → Fine-tuning)
- **Inference Speed**: Real-time classification with batch processing support
- **Web Interface**: Professional Flask application with dark/light mode support

## 🚀 Quick Start

### 1. Installation & Setup

```bash
# Clone the repository
git clone <repository-url>
cd Plankton-species-classifier

# Install dependencies
pip install -r requirements.txt

# Ensure you have the dataset
# Dataset should be in ./2014_clean/ directory
```

### 2. Launch the Web Application

```bash
# Start the Flask web application
python flask_app.py
```

Open your browser to `http://localhost:5000` to access the web interface.

### 3. Using the Application

**Single Image Classification:**
- Upload a plankton image through the web interface
- Get instant classification with confidence scores
- View detailed species information

**Batch Processing:**
- Upload multiple images at once
- View results in a responsive grid layout
- Export results to Excel format

**Model Information:**
- Click the "Model Info" button in the navigation bar
- View comprehensive details about the current model architecture
- See training methodology and performance metrics

## 📁 Current Project Structure

```
Plankton-species-classifier/
├── flask_app.py                    # Main Flask web application
├── species_database.json           # Comprehensive species database (67 species)
├── requirements.txt                # Python dependencies
├── train_approach1_improved.py     # Current training script
├──
├── templates/
│   └── index_enhanced_fixed.html   # Modern web interface template
├──
├── static/
│   ├── app_enhanced_fixed.js       # Frontend JavaScript logic
│   ├── favicon.svg                 # Application favicon
│   └── favicon.png                 # Fallback favicon
├──
├── outputs/models/                 # All trained models
│   ├── approach1_final_model.pth   # 🎯 Current best model (89.51%)
│   ├── approach1_stage1_foundation_best.pth
│   ├── approach1_stage2_refinement_best.pth
│   ├── approach1_stage3_fine-tuning_best.pth
│   └── [other model files...]
├──
├── models/                         # Model architecture definitions
│   └── cnn_model.py               # CNN model classes (PlanktonCNN)
├──
├── 2014_clean/                    # Processed dataset
│   └── [67 species directories with images]
├──
└── REMOVABLE_STUFF/               # Archived/unused files
    └── [backup files, old versions, alternative approaches]
```

## 🤖 Model Architecture & Training

### Current Production Model: `approach1_final_model.pth`

**Architecture Details:**
- **Base Model**: EfficientNet-B2
- **Input Size**: 224×224 RGB images
- **Output Classes**: 67 plankton species
- **Training Strategy**: Progressive 3-stage approach

**Training Methodology:**
1. **Stage 1 - Foundation** (`train_approach1_colab.py`)
   - Initial feature learning with frozen backbone
   - Focus on basic plankton morphology recognition

2. **Stage 2 - Refinement** (`train_approach1_improved.py`)
   - Fine-tuning specific layers
   - Enhanced feature discrimination

3. **Stage 3 - Fine-tuning**
   - End-to-end optimization
   - Species-specific characteristic learning

**Performance Metrics:**
- **Validation Accuracy**: 89.51%
- **Model Size**: ~40MB
- **Inference Time**: <100ms per image
- **Training Time**: ~6-8 hours (3 stages combined)

### Training Your Own Models

```bash
# Stage 1: Foundation training (Colab optimized)
python train_approach1_colab.py

# Stage 2 & 3: Refinement and fine-tuning (Local)
python train_approach1_improved.py
```

## 🌐 Web Application Features

### 🎨 User Interface
- **Modern Design**: Clean, responsive interface with Tailwind CSS
- **Dark Mode Support**: Toggle between light and dark themes
- **Marine Theme**: Ocean-inspired color palette and favicon
- **Mobile Responsive**: Works seamlessly on all device sizes

### 🔍 Classification Features
- **Single Image Upload**: Drag-and-drop or click to upload
- **Batch Processing**: Upload and process multiple images simultaneously
- **Real-time Results**: Instant classification with confidence scores
- **Species Information**: Detailed information for each classified species

### 📊 Advanced Features
- **Model Information Modal**: Comprehensive model details and performance metrics
- **Species Database Browser**: Explore all 67 supported species
- **Excel Export**: Export batch processing results to Excel format
- **Confidence Visualization**: Clear confidence score displays
- **Error Handling**: Robust error handling with user-friendly messages

### 🔧 Technical Features
- **API Endpoints**: RESTful API for programmatic access
- **Image Processing**: Automatic image preprocessing and validation
- **Memory Management**: Efficient handling of large batch uploads
- **Caching**: Smart caching for improved performance

## 📊 Species Database

The system supports classification of **67 distinct plankton species**, including:

**Major Groups Covered:**
- Copepods (various species and life stages)
- Diatoms and phytoplankton
- Protozoans and other microorganisms
- Marine larvae and juvenile forms
- Radiolarians and foraminifera

**Database Features:**
- Comprehensive species information
- Scientific nomenclature
- Morphological characteristics
- Confidence thresholds for each species

## 🧪 API Usage

### Programmatic Classification

```python
import requests
import json

# Single image classification
files = {'image': open('plankton_image.jpg', 'rb')}
response = requests.post('http://localhost:5000/predict', files=files)
result = response.json()

print(f"Species: {result['species']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Batch Processing API

```python
# Batch processing
files = [
    ('images', open('image1.jpg', 'rb')),
    ('images', open('image2.jpg', 'rb')),
    # ... more images
]
response = requests.post('http://localhost:5000/batch_predict', files=files)
results = response.json()

for result in results['results']:
    print(f"File: {result['filename']} → Species: {result['species']}")
```

## 🔧 Configuration & Customization

### Model Configuration
Edit `config.py` to customize:
- Model selection and paths
- Image preprocessing parameters
- Classification confidence thresholds
- Training hyperparameters

### Web Interface Customization
- **Template**: Modify `templates/index_enhanced_fixed.html`
- **Styling**: Tailwind CSS classes for easy customization
- **JavaScript**: `static/app_enhanced_fixed.js` for frontend logic
- **Species Database**: Update `species_database.json` for new species

## 📈 Performance Benchmarks

### Model Comparison
| Model | Accuracy | Model Size | Inference Time | Training Approach |
|-------|----------|------------|----------------|-------------------|
| **approach1_final_model** | **89.51%** | 40MB | <100ms | Progressive 3-stage |
| approach1_stage3 | 87.2% | 40MB | <100ms | 3-stage fine-tuning |
| approach1_stage2 | 84.1% | 40MB | <100ms | 2-stage refinement |
| approach1_stage1 | 78.9% | 40MB | <100ms | Foundation training |

### System Requirements
- **Minimum RAM**: 4GB (8GB+ recommended for training)
- **GPU**: Optional for inference, required for training
- **Storage**: 2GB+ (includes dataset and models)
- **CPU**: Modern multi-core processor

## 🛠️ Development & Testing

### Running Tests
```bash
# Install development dependencies
pip install -r requirements.txt

# Run the Flask application in development mode
python flask_app.py

# Test prediction functionality
python predict.py --image test_image.jpg
```

### Development Workflow
1. Make changes to the Flask app (`flask_app.py`)
2. Update frontend code (`templates/` and `static/`)
3. Test with sample images
4. Verify batch processing and export functionality

## 📚 Advanced Usage

### Custom Model Integration
To integrate a new trained model:

1. Place the model file in `outputs/models/`
2. Update the model path in `flask_app.py`
3. Ensure class mappings match `species_database.json`
4. Test prediction accuracy

### Extending Species Database
To add new species:

1. Update `species_database.json` with new species information
2. Retrain the model with additional species data
3. Update class mappings accordingly
4. Verify web interface compatibility

## 🐛 Troubleshooting

### Common Issues

**Model Loading Errors:**
- Ensure PyTorch and torchvision are properly installed
- Verify model file integrity and path
- Check CUDA availability for GPU acceleration

**Web Interface Issues:**
- Clear browser cache and cookies
- Ensure Flask server is running on correct port
- Check for JavaScript console errors

**Classification Accuracy:**
- Verify image quality and format (RGB, proper resolution)
- Ensure images contain clear plankton specimens
- Check confidence thresholds in configuration

## 🤝 Contributing

We welcome contributions to improve the plankton classifier! Areas for contribution:

- **Model Improvements**: Better architectures, training strategies
- **Web Interface**: Enhanced UI/UX, additional features
- **Species Database**: Expanded species coverage, better annotations
- **Performance**: Optimization, faster inference, memory efficiency
- **Documentation**: Tutorials, examples, API documentation

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **WHOI-Plankton Dataset**: Woods Hole Oceanographic Institution for providing the comprehensive plankton dataset
- **EfficientNet**: Google Research for the efficient neural network architecture
- **PyTorch Community**: For the excellent deep learning framework and ecosystem
- **Marine Biology Research**: Community contributions to plankton taxonomy and identification

## 📧 Contact & Support

For questions, issues, or collaboration:
- **GitHub Issues**: Report bugs and request features
- **Model Performance**: Share your results and improvements
- **Research Collaboration**: Contact for academic partnerships

---

**🌊 Start classifying marine plankton with cutting-edge AI accuracy in minutes! 🔬**

*This project advances marine biology research through accessible AI technology, supporting biodiversity monitoring and ecosystem health assessment worldwide.*