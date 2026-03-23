# 🌐 Web Interface Documentation

## Overview

The Plankton Species Classifier provides multiple web interface options designed for different use cases and deployment scenarios.

## 🎨 Streamlit Application

### Features
- **Modern Design**: Glass morphism UI with professional theming
- **Real-time Classification**: Instant predictions with confidence visualization
- **Interactive Charts**: Top-5 predictions with Plotly visualizations
- **Theme Switching**: Seamless dark/light mode toggle
- **Model Information**: Live performance metrics and architecture details

### Usage
```bash
streamlit run src/web/app.py
```

### Interface Components
- **Upload Section**: Drag & drop or file picker with URL support
- **Results Display**: Species identification with confidence scores
- **Top-5 Chart**: Interactive bar chart of prediction probabilities
- **Model Stats**: Performance metrics and architecture information
- **About Page**: Project documentation and technical details

### Customization
Edit `src/web/app.py` to modify:
- Color themes and styling
- Chart configurations
- Model loading behavior
- UI layout and components

## ⚡ Flask Application

### Features
- **Production Ready**: Robust backend with comprehensive error handling
- **REST API**: Programmatic access via JSON endpoints
- **Advanced Processing**: Image enhancement and quality assessment
- **Batch Support**: Multiple image processing capabilities
- **Template System**: Flexible HTML template rendering

### API Endpoints

#### Prediction Endpoint
```http
POST /api/predict
Content-Type: multipart/form-data

file: image file (JPG, PNG, WEBP, TIFF)
```

**Response:**
```json
{
  "predictions": [
    {"species": "copepod_calanoid", "confidence": 85.3},
    {"species": "diatom_chain", "confidence": 8.7},
    ...
  ],
  "status": "success"
}
```

#### Model Information
```http
GET /api/model-info
```

**Response:**
```json
{
  "model_name": "EfficientNetV2-S",
  "num_classes": 54,
  "accuracy": {
    "top1": 75.5,
    "top5": 96.4
  },
  "status": "loaded"
}
```

### Usage
```bash
# Development server
python src/web/flask_app.py

# Production deployment
gunicorn src.web.flask_app:app
```

### Configuration
Environment variables:
- `FLASK_ENV`: development/production
- `MODEL_PATH`: Custom model file path
- `UPLOAD_FOLDER`: Image upload directory
- `MAX_CONTENT_LENGTH`: Maximum file size

## 🚀 Deployment Options

### Local Development
```bash
# Install dependencies
pip install streamlit flask plotly

# Run Streamlit
streamlit run src/web/app.py

# Run Flask
python src/web/flask_app.py
```

### Docker Deployment
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install -e ".[web]"

# Streamlit
EXPOSE 8501
CMD ["streamlit", "run", "src/web/app.py"]

# Flask
EXPOSE 5000
CMD ["python", "src/web/flask_app.py"]
```

### Cloud Deployment

#### Streamlit Cloud
1. Push code to GitHub
2. Connect at [share.streamlit.io](https://share.streamlit.io)
3. Deploy with one click

#### Heroku
```bash
# Create Procfile
echo "web: streamlit run src/web/app.py --server.port $PORT" > Procfile

# Deploy
git add .
git commit -m "Deploy to Heroku"
heroku create your-app-name
git push heroku main
```

#### AWS/GCP/Azure
- Use container services (ECS, Cloud Run, Container Instances)
- Configure auto-scaling and load balancing
- Set up CI/CD pipelines for automated deployment

## 🎛️ Interface Customization

### Theme Configuration
```python
# Custom colors (Streamlit)
def get_custom_theme():
    return {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'success': '#2ca02c',
        'background': '#ffffff'
    }
```

### Adding New Pages
```python
# In src/web/app.py
def new_page():
    st.title("New Feature")
    # Add your functionality here

# Add to navigation
pages = {
    "Classify": classify_page,
    "Model Stats": model_stats_page,
    "New Feature": new_page,  # Add here
    "About": about_page
}
```

### Custom Visualization
```python
# Add new chart types
def create_species_distribution_chart(predictions):
    fig = go.Figure(data=[
        go.Scatter(x=species, y=confidence, mode='markers+lines')
    ])
    return fig
```

## 🔧 Troubleshooting

### Common Issues

**Streamlit Port Already in Use:**
```bash
streamlit run src/web/app.py --server.port 8502
```

**Flask Model Loading Errors:**
- Check model file paths in `config.py`
- Verify PyTorch version compatibility
- Ensure sufficient memory for model loading

**Image Upload Issues:**
- Verify supported file formats (JPG, PNG, WEBP, TIFF)
- Check file size limits (default 10MB)
- Ensure proper PIL/Pillow installation

### Performance Optimization

**Streamlit:**
- Use `@st.cache_resource` for model loading
- Implement session state for efficiency
- Optimize image processing pipeline

**Flask:**
- Enable response compression
- Use async processing for batch uploads
- Implement caching for frequent predictions
- Configure proper WSGI server (Gunicorn, uWSGI)

## 📱 Mobile Responsiveness

Both interfaces are mobile-optimized:
- **Responsive layouts**: Adapt to different screen sizes
- **Touch-friendly**: Large buttons and touch targets
- **Mobile navigation**: Streamlined interface on small screens
- **Progressive Web App**: Streamlit supports PWA features

## 🔐 Security Considerations

### File Upload Security
- Validate file types and extensions
- Scan for malicious content
- Limit file sizes and upload rates
- Sanitize filenames and paths

### API Security
- Implement rate limiting
- Add authentication for production use
- Validate and sanitize all inputs
- Use HTTPS in production

### Deployment Security
- Keep dependencies updated
- Use security headers
- Configure proper CORS policies
- Monitor for vulnerabilities

## 📊 Analytics & Monitoring

### Usage Tracking
```python
# Track predictions
def log_prediction(species, confidence, timestamp):
    # Implement logging logic
    pass

# Monitor performance
def track_inference_time(start_time, end_time):
    # Log performance metrics
    pass
```

### Error Monitoring
- Implement structured logging
- Set up error tracking (Sentry, etc.)
- Monitor system resources
- Track user interactions and feedback

## 🎯 Best Practices

### Code Organization
- Separate UI components into modules
- Keep business logic in utilities
- Use configuration files for settings
- Implement proper error handling

### User Experience
- Provide clear loading indicators
- Show helpful error messages
- Include example images and instructions
- Optimize for fast loading times

### Maintenance
- Regular dependency updates
- Monitor application performance
- Collect user feedback
- Test on multiple browsers and devices