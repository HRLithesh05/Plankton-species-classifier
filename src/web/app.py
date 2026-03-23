"""
Plankton Species Classifier - Modern Web App
Clean, minimal, professional design with excellent UX
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import json
from pathlib import Path
import plotly.graph_objects as go
from torchvision import transforms, models
import requests
from io import BytesIO

from models.cnn_model import PlanktonCNN

# Page configuration
st.set_page_config(
    page_title="Plankton Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

def get_modern_css():
    is_dark = st.session_state.theme == 'dark'

    if is_dark:
        # Dark mode - professional dark
        bg_primary = '#1a1d29'
        bg_secondary = '#252836'
        bg_card = '#2a2d3a'
        text_primary = '#ffffff'
        text_secondary = '#b4b7c1'
        text_muted = '#8b8fa3'
        border = '#3a3d4a'
        accent = '#5b9bd5'
        accent_light = 'rgba(91, 155, 213, 0.1)'
        success = '#4ade80'
        warning = '#fb923c'
        error = '#f87171'
        shadow = 'rgba(0, 0, 0, 0.3)'
    else:
        # Light mode - clean white
        bg_primary = '#ffffff'
        bg_secondary = '#f8fafc'
        bg_card = '#ffffff'
        text_primary = '#1e293b'
        text_secondary = '#64748b'
        text_muted = '#94a3b8'
        border = '#e2e8f0'
        accent = '#3b82f6'
        accent_light = 'rgba(59, 130, 246, 0.1)'
        success = '#22c55e'
        warning = '#f59e0b'
        error = '#ef4444'
        shadow = 'rgba(0, 0, 0, 0.1)'

    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        * {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }}

        .stApp {{
            background: {bg_primary};
            color: {text_primary};
        }}

        /* Hide default elements */
        #MainMenu, footer, header {{visibility: hidden;}}
        .stDeployButton {{display: none;}}

        /* Modern Header */
        .modern-header {{
            background: {bg_card};
            padding: 2rem 0;
            text-align: center;
            margin-bottom: 3rem;
            border-bottom: 1px solid {border};
        }}

        .app-title {{
            font-size: 2.5rem;
            font-weight: 700;
            color: {text_primary};
            margin: 0;
            letter-spacing: -0.025em;
        }}

        .app-subtitle {{
            font-size: 1.125rem;
            color: {text_secondary};
            margin-top: 0.5rem;
            font-weight: 400;
        }}

        .header-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: {accent_light};
            color: {accent};
            padding: 0.5rem 1rem;
            border-radius: 100px;
            font-size: 0.875rem;
            font-weight: 500;
            margin-top: 1rem;
        }}

        /* Cards */
        .modern-card {{
            background: {bg_card};
            border: 1px solid {border};
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px {shadow};
            transition: all 0.2s ease;
        }}

        .modern-card:hover {{
            box-shadow: 0 4px 12px {shadow};
            transform: translateY(-2px);
        }}

        .card-header {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1.25rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid {border};
        }}

        .card-icon {{
            width: 40px;
            height: 40px;
            background: {accent_light};
            color: {accent};
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.25rem;
        }}

        .card-title {{
            font-size: 1.125rem;
            font-weight: 600;
            color: {text_primary};
            margin: 0;
        }}

        /* Result Display */
        .result-container {{
            background: {bg_card};
            border: 2px solid {accent};
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            margin: 1.5rem 0;
        }}

        .result-badge {{
            background: {accent_light};
            color: {accent};
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 0.375rem 0.75rem;
            border-radius: 6px;
            display: inline-block;
            margin-bottom: 1rem;
        }}

        .result-species {{
            font-size: 1.875rem;
            font-weight: 700;
            color: {text_primary};
            margin: 0.75rem 0;
            line-height: 1.2;
        }}

        .result-confidence {{
            font-size: 3rem;
            font-weight: 800;
            color: {accent};
            margin: 0.5rem 0;
        }}

        .confidence-label {{
            font-size: 0.875rem;
            color: {text_secondary};
            font-weight: 500;
        }}

        /* Buttons */
        .stButton > button {{
            background: {accent} !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            font-size: 0.875rem !important;
            padding: 0.75rem 1.5rem !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 1px 3px rgba(59, 130, 246, 0.3) !important;
        }}

        .stButton > button:hover {{
            background: #2563eb !important;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4) !important;
            transform: translateY(-1px) !important;
        }}

        .stButton > button:active {{
            transform: translateY(0) !important;
        }}

        /* Inputs */
        .stTextInput > div > div > input {{
            background: {bg_card} !important;
            border: 1px solid {border} !important;
            border-radius: 8px !important;
            color: {text_primary} !important;
            padding: 0.75rem 1rem !important;
            font-size: 0.875rem !important;
        }}

        .stTextInput > div > div > input:focus {{
            border-color: {accent} !important;
            box-shadow: 0 0 0 3px {accent_light} !important;
        }}

        .stTextInput > div > div > input::placeholder {{
            color: {text_muted} !important;
        }}

        .stTextInput > label {{
            color: {text_primary} !important;
            font-weight: 500 !important;
            font-size: 0.875rem !important;
        }}

        /* File Upload */
        [data-testid="stFileUploadDropzone"] {{
            background: {bg_secondary} !important;
            border: 2px dashed {border} !important;
            border-radius: 12px !important;
            padding: 2rem !important;
        }}

        [data-testid="stFileUploadDropzone"]:hover {{
            border-color: {accent} !important;
            background: {accent_light} !important;
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background: {bg_secondary} !important;
            border-right: 1px solid {border};
        }}

        section[data-testid="stSidebar"] * {{
            color: {text_primary} !important;
        }}

        section[data-testid="stSidebar"] .stMarkdown h3 {{
            color: {text_primary} !important;
            font-size: 0.875rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
            margin-bottom: 1rem !important;
        }}

        /* Radio buttons */
        .stRadio > div {{
            gap: 0.5rem !important;
        }}

        .stRadio > div > label {{
            background: {bg_card} !important;
            border: 1px solid {border} !important;
            border-radius: 8px !important;
            padding: 0.75rem 1rem !important;
            margin: 0.25rem 0 !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
        }}

        .stRadio > div > label:hover {{
            border-color: {accent} !important;
            background: {accent_light} !important;
        }}

        .stRadio > div > label[data-checked="true"] {{
            background: {accent} !important;
            color: white !important;
            border-color: {accent} !important;
        }}

        /* Metrics */
        [data-testid="stMetricValue"] {{
            color: {accent} !important;
            font-weight: 700 !important;
        }}

        [data-testid="stMetricLabel"] {{
            color: {text_secondary} !important;
        }}

        /* Status indicators */
        .stSuccess {{
            background: rgba(34, 197, 94, 0.1) !important;
            color: {success} !important;
            border: 1px solid rgba(34, 197, 94, 0.2) !important;
            border-radius: 8px !important;
        }}

        .stWarning {{
            background: rgba(245, 158, 11, 0.1) !important;
            color: {warning} !important;
            border: 1px solid rgba(245, 158, 11, 0.2) !important;
            border-radius: 8px !important;
        }}

        .stError {{
            background: rgba(239, 68, 68, 0.1) !important;
            color: {error} !important;
            border: 1px solid rgba(239, 68, 68, 0.2) !important;
            border-radius: 8px !important;
        }}

        .stInfo {{
            background: {accent_light} !important;
            color: {accent} !important;
            border: 1px solid rgba(59, 130, 246, 0.2) !important;
            border-radius: 8px !important;
        }}

        /* Expander */
        .streamlit-expanderHeader {{
            background: {bg_card} !important;
            border: 1px solid {border} !important;
            border-radius: 8px !important;
            color: {text_primary} !important;
        }}

        .streamlit-expanderContent {{
            background: {bg_card} !important;
            border: 1px solid {border} !important;
            border-top: none !important;
            border-radius: 0 0 8px 8px !important;
        }}

        /* Headers */
        h1, h2, h3, h4, h5, h6 {{
            color: {text_primary} !important;
            font-weight: 600 !important;
        }}

        p {{
            color: {text_secondary} !important;
        }}

        /* Clean scrollbar */
        ::-webkit-scrollbar {{
            width: 6px;
        }}

        ::-webkit-scrollbar-track {{
            background: {bg_secondary};
        }}

        ::-webkit-scrollbar-thumb {{
            background: {border};
            border-radius: 3px;
        }}

        ::-webkit-scrollbar-thumb:hover {{
            background: {accent};
        }}

        /* Image styling */
        .stImage > img {{
            border-radius: 8px;
            box-shadow: 0 1px 3px {shadow};
        }}

        /* Divider */
        .divider {{
            display: flex;
            align-items: center;
            gap: 1rem;
            margin: 1.5rem 0;
        }}

        .divider::before,
        .divider::after {{
            content: '';
            flex: 1;
            height: 1px;
            background: {border};
        }}

        .divider-text {{
            font-size: 0.875rem;
            color: {text_muted};
            font-weight: 500;
        }}
    </style>
    """

st.markdown(get_modern_css(), unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained CNN model."""
    model_path = Path("outputs/models/cnn_final_colab.pth")
    mapping_path = Path("outputs/models/class_mapping_colab.json")

    if not model_path.exists():
        model_path = Path("outputs/models/cnn_final.pth")
        mapping_path = Path("outputs/models/class_mapping.json")
    if not model_path.exists():
        model_path = Path("outputs/models/best_model_finetune.pth")

    if not model_path.exists():
        return None, None, None

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    if mapping_path.exists():
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
            class_to_idx = mapping['class_to_idx']
            idx_to_class = {int(k): v for k, v in mapping['idx_to_class'].items()}
            num_classes = len(class_to_idx)
    else:
        class_to_idx = checkpoint.get('class_to_idx', {})
        idx_to_class = checkpoint.get('idx_to_class', {})
        if isinstance(list(idx_to_class.keys())[0], str):
            idx_to_class = {int(k): v for k, v in idx_to_class.items()}
        num_classes = checkpoint.get('num_classes', len(class_to_idx))

    config = checkpoint.get('config', {'model_name': 'efficientnet_v2_s', 'dropout': 0.25})
    state_dict = checkpoint['model_state_dict']
    has_backbone = any(k.startswith('backbone.classifier') for k in state_dict.keys())

    if has_backbone:
        model = models.efficientnet_v2_s(weights=None)
        in_features = model.classifier[1].in_features
        dropout = config.get('dropout', 0.25)
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(512, num_classes)
        )
        new_state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        model.eval()

        class Wrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.model = m
            def forward(self, x):
                return self.model(x)

        model = Wrapper(model)
    else:
        model = PlanktonCNN(
            num_classes=num_classes,
            model_name=config['model_name'],
            pretrained=False,
            dropout=config.get('dropout', 0.25),
            freeze_backbone=False
        )
        model.load_state_dict(state_dict)
        model.eval()

    return model, idx_to_class, class_to_idx

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def enhance_image(image):
    from PIL import ImageEnhance, ImageOps
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = ImageOps.autocontrast(image, cutoff=2)
    image = ImageEnhance.Contrast(image).enhance(1.2)
    image = ImageEnhance.Sharpness(image).enhance(1.3)
    return image

def load_image_from_url(url):
    try:
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except:
        return None

def predict_image(model, image, idx_to_class, top_k=5):
    transform = get_transforms()
    image = enhance_image(image)
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, top_k)

    predictions = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        predictions.append({
            'species': idx_to_class[idx.item()],
            'confidence': prob.item() * 100
        })

    return predictions

def main():
    # Modern Header
    st.markdown("""
    <div class="modern-header">
        <h1 class="app-title">Plankton Classifier</h1>
        <p class="app-subtitle">AI-powered microscopic organism identification</p>
        <div class="header-badge">
            <span>🧠</span>
            <span>Powered by EfficientNetV2-S</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### Navigation")
        page = st.radio(
            "",
            ["Classify", "Model Stats", "About"],
            label_visibility="collapsed"
        )

        st.markdown("---")

        st.markdown("### Theme")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("☀️ Light", use_container_width=True):
                st.session_state.theme = 'light'
                st.rerun()
        with col2:
            if st.button("🌙 Dark", use_container_width=True):
                st.session_state.theme = 'dark'
                st.rerun()

        st.markdown("---")

        st.markdown("### Model Info")
        st.info("""
        **Architecture:** EfficientNetV2-S
        **Classes:** 54 species
        **Top-1 Accuracy:** 75.5%
        **Top-5 Accuracy:** 96.4%
        """)

    # Load model
    with st.spinner("Loading model..."):
        model, idx_to_class, class_to_idx = load_model()

    if model is None:
        st.error("Model not found. Please train the model first.")
        return

    # Page routing
    if page == "Classify":
        classify_page(model, idx_to_class, class_to_idx)
    elif page == "Model Stats":
        model_stats_page()
    else:
        about_page()

def classify_page(model, idx_to_class, class_to_idx):
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("""
        <div class="modern-card">
            <div class="card-header">
                <div class="card-icon">📤</div>
                <h3 class="card-title">Upload Image</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # File upload
        uploaded_file = st.file_uploader(
            "Choose a plankton image",
            type=['png', 'jpg', 'jpeg', 'webp', 'tiff'],
            label_visibility="collapsed"
        )

        st.markdown('<div class="divider"><span class="divider-text">or</span></div>', unsafe_allow_html=True)

        # URL input
        url = st.text_input(
            "Image URL",
            placeholder="https://example.com/plankton.jpg"
        )

        # Get image
        image = None
        if uploaded_file:
            image = Image.open(uploaded_file)
        elif url:
            with st.spinner("Loading image..."):
                image = load_image_from_url(url)
                if image is None:
                    st.error("Could not load image from URL")

        if image:
            st.image(image, caption="Uploaded Image", use_container_width=True)

            if st.button("Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing..."):
                    predictions = predict_image(model, image, idx_to_class)
                    st.session_state.predictions = predictions
                    st.rerun()

    with col2:
        st.markdown("""
        <div class="modern-card">
            <div class="card-header">
                <div class="card-icon">🎯</div>
                <h3 class="card-title">Results</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.predictions:
            predictions = st.session_state.predictions
            top_pred = predictions[0]

            species_display = top_pred['species'].replace('_', ' ').title()
            confidence = top_pred['confidence']

            # Result display
            st.markdown(f"""
            <div class="result-container">
                <div class="result-badge">Predicted Species</div>
                <div class="result-species">{species_display}</div>
                <div class="result-confidence">{confidence:.1f}%</div>
                <div class="confidence-label">Confidence</div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence status
            if confidence >= 70:
                st.success("High confidence prediction")
            elif confidence >= 50:
                st.warning("Medium confidence prediction")
            else:
                st.error("Low confidence prediction")

            # Top 5 chart
            st.markdown("#### Top 5 Predictions")

            species_names = [p['species'].replace('_', ' ').title() for p in predictions]
            confidences = [p['confidence'] for p in predictions]

            # Create clean chart
            colors = ['#3b82f6', '#6366f1', '#8b5cf6', '#a855f7', '#c084fc']

            fig = go.Figure(go.Bar(
                x=confidences,
                y=species_names,
                orientation='h',
                marker=dict(
                    color=colors,
                    cornerradius=4
                ),
                text=[f"{c:.1f}%" for c in confidences],
                textposition='auto',
                textfont=dict(color='white', size=12, weight='600'),
                hovertemplate='<b>%{y}</b><br>Confidence: %{x:.1f}%<extra></extra>'
            ))

            fig.update_layout(
                height=280,
                margin=dict(l=0, r=20, t=10, b=20),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    range=[0, 100],
                    showgrid=True,
                    gridcolor='rgba(0,0,0,0.05)',
                    title="Confidence (%)",
                    titlefont=dict(size=12),
                    tickfont=dict(size=10)
                ),
                yaxis=dict(
                    showgrid=False,
                    autorange='reversed',
                    tickfont=dict(size=11)
                ),
                font=dict(family='Inter')
            )

            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            # Detailed results
            with st.expander("Detailed Results"):
                for i, pred in enumerate(predictions, 1):
                    st.write(f"**{i}.** {pred['species'].replace('_', ' ').title()} — {pred['confidence']:.2f}%")

        else:
            st.info("Upload an image to see classification results")

def model_stats_page():
    st.markdown("## Model Performance")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Top-1 Accuracy", "75.5%")
    with col2:
        st.metric("Top-3 Accuracy", "92.3%")
    with col3:
        st.metric("Top-5 Accuracy", "96.4%")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="modern-card">
            <div class="card-header">
                <div class="card-icon">🏗️</div>
                <h3 class="card-title">Architecture</h3>
            </div>
            <p><strong>Base Model:</strong> EfficientNetV2-S</p>
            <p><strong>Parameters:</strong> 21.5M</p>
            <p><strong>Input Size:</strong> 224×224×3</p>
            <p><strong>Classes:</strong> 54 species</p>
            <p><strong>Training:</strong> Transfer Learning</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="modern-card">
            <div class="card-header">
                <div class="card-icon">📊</div>
                <h3 class="card-title">Dataset</h3>
            </div>
            <p><strong>Source:</strong> WHOI-Plankton</p>
            <p><strong>Images:</strong> 20,000+</p>
            <p><strong>Species:</strong> 54 classes</p>
            <p><strong>Split:</strong> 70/15/15</p>
            <p><strong>Augmentation:</strong> Yes</p>
        </div>
        """, unsafe_allow_html=True)

def about_page():
    st.markdown("## About This Project")

    st.markdown("""
    <div class="modern-card">
        <div class="card-header">
            <div class="card-icon">🔬</div>
            <h3 class="card-title">Plankton Classification System</h3>
        </div>
        <p>This application uses deep learning to automatically identify microscopic plankton species from images.</p>

        <h4>Key Features</h4>
        <ul>
            <li><strong>54 Species:</strong> Comprehensive classification across marine plankton taxa</li>
            <li><strong>High Accuracy:</strong> 96.4% top-5 accuracy on test dataset</li>
            <li><strong>Real-time:</strong> Instant predictions with confidence scores</li>
            <li><strong>Easy to Use:</strong> Simple upload interface with URL support</li>
        </ul>

        <h4>Technology</h4>
        <ul>
            <li><strong>Model:</strong> EfficientNetV2-S with transfer learning</li>
            <li><strong>Framework:</strong> PyTorch for deep learning</li>
            <li><strong>Interface:</strong> Streamlit for web application</li>
            <li><strong>Visualization:</strong> Plotly for interactive charts</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()