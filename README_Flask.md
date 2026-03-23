# 🌊 Oceanic Precision - Plankton Species Classifier

A modern, professional web application for classifying plankton species using deep learning. This application seamlessly integrates beautiful HTML frontend designs with PyTorch ML models.

## ✨ Features

### 🎨 Beautiful Modern UI
- **Dark/Light Theme Toggle**: Seamless switching between professional dark and light themes
- **Material Design**: Clean, modern interface using Tailwind CSS and Material Icons
- **Responsive Layout**: Works perfectly on desktop and mobile devices
- **Professional Animations**: Smooth transitions and scanning effects during analysis

### 🧠 Advanced ML Integration
- **EfficientNetV2-S Architecture**: State-of-the-art deep learning model
- **54 Species Classification**: Comprehensive plankton species identification
- **High Accuracy**: 96.4% top-5 accuracy on test dataset
- **Real-time Predictions**: Instant analysis with confidence scores

### 🚀 Interactive Features
- **Drag & Drop Upload**: Easy image uploading with visual feedback
- **URL Image Loading**: Load images directly from web URLs
- **Live Progress**: Real-time analysis progress with scanning animation
- **Results Visualization**: Interactive charts showing top 5 predictions
- **Session History**: Track recent analyses with thumbnails
- **Detailed Results**: Expandable view with complete prediction breakdown

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r flask_requirements.txt
```

### 2. Run the Application
```bash
python run_app.py
```

The application will automatically:
- Start the Flask server on `http://localhost:5000`
- Open your default browser
- Load the beautiful Oceanic Precision interface

### 3. Start Classifying!
1. **Upload an Image**: Either drag & drop or click "Upload Image"
2. **Or Use URL**: Paste an image URL in the input field
3. **Analyze**: Click "Analyze Species" to start the ML prediction
4. **View Results**: See the predicted species with confidence scores

## 🎯 Model Information

- **Architecture**: EfficientNetV2-S with custom classifier
- **Training Data**: WHOI-Plankton dataset (20,000+ images)
- **Classes**: 54 marine plankton species
- **Performance**:
  - Top-1 Accuracy: 75.5%
  - Top-3 Accuracy: 92.3%
  - Top-5 Accuracy: 96.4%

## 🔧 Technical Details

### Backend (Flask)
- **Framework**: Flask 3.0.0
- **ML Framework**: PyTorch 2.0+
- **Image Processing**: PIL/Pillow
- **API Endpoints**:
  - `POST /api/predict`: Image classification
  - `GET /api/model-info`: Model statistics

### Frontend
- **Styling**: Tailwind CSS with custom Material Design theme
- **Charts**: Chart.js for interactive visualizations
- **Icons**: Material Symbols
- **Fonts**: Inter + Space Grotesk for professional typography

### File Structure
```
├── flask_app.py          # Main Flask application
├── run_app.py           # Application launcher
├── templates/
│   └── index.html       # Main HTML template
├── static/
│   └── app.js          # Frontend JavaScript
├── models/              # ML model files
├── outputs/models/      # Trained model checkpoints
└── flask_requirements.txt
```

## 🎨 Theme System

The application features a sophisticated dual-theme system:

### 🌙 Dark Theme
- Deep ocean blues (#001627)
- Cyan accents (#56f5f8)
- Professional dark surfaces
- Optimal for low-light environments

### ☀️ Light Theme
- Clean whites and blues
- Navy accents (#001627)
- Bright, accessible interface
- Perfect for daytime use

Switch themes instantly using the floating toggle button in the top-right corner.

## 🔍 Supported Species

The model can identify 54 different plankton species including:
- Radiolaria
- Copepods
- Chaetoceros
- Diatoms
- Dinoflagellates
- And many more marine microorganisms

## 📱 Browser Compatibility

- ✅ Chrome/Chromium (Recommended)
- ✅ Firefox
- ✅ Safari
- ✅ Edge
- ⚠️ IE11+ (Limited support)

## 🚨 Troubleshooting

### Model Not Loading
1. Ensure model files exist in `outputs/models/`
2. Check that PyTorch is properly installed
3. Verify Python version compatibility (3.8+)

### Upload Issues
1. Check file size (max 500MB)
2. Ensure supported format (PNG, JPG, JPEG, WEBP, TIFF)
3. Try using URL upload as alternative

### Performance
1. Use Chrome for best performance
2. Ensure stable internet connection for CDN resources
3. Consider GPU acceleration for faster predictions

## 📄 License

© 2024 Oceanic Precision AI. All rights reserved.

---

**Enjoy exploring the microscopic world with AI! 🔬🌊**