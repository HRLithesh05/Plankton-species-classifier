"""
Plankton Species Classifier - Streamlit Web App
Real-time plankton identification using deep learning
"""

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from torchvision import transforms

# Import your model
from models.cnn_model import PlanktonCNN


# Page configuration
st.set_page_config(
    page_title="Plankton Species Classifier",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained CNN model."""
    model_path = Path("outputs/models/cnn_final.pth")

    if not model_path.exists():
        st.error(f"Model not found at {model_path}")
        return None, None, None

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')

    # Get class mapping
    class_to_idx = checkpoint['class_to_idx']
    idx_to_class = checkpoint['idx_to_class']
    num_classes = checkpoint['num_classes']

    # Create model
    config = checkpoint['config']
    model = PlanktonCNN(
        num_classes=num_classes,
        model_name=config['model_name'],
        pretrained=False,
        dropout=config['dropout'],
        freeze_backbone=False
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, idx_to_class, class_to_idx


@st.cache_data
def load_statistics():
    """Load training statistics."""
    stats_path = Path("outputs/results/cnn_training_history.json")

    if not stats_path.exists():
        return None

    with open(stats_path, 'r') as f:
        stats = json.load(f)

    return stats


def get_transforms():
    """Get image preprocessing transforms."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def predict_image(model, image, idx_to_class, top_k=5):
    """Predict plankton species from image."""
    transform = get_transforms()

    # Preprocess
    if image.mode != 'RGB':
        image = image.convert('RGB')

    img_tensor = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, top_k)

    # Get predictions
    predictions = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        predictions.append({
            'species': idx_to_class[str(idx.item())],
            'confidence': prob.item() * 100
        })

    return predictions


def main():
    """Main Streamlit app."""

    # Header
    st.markdown('<div class="main-header">🦠 Plankton Species Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Identify microscopic marine organisms using AI</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/streamlit/streamlit/develop/docs/_static/img/streamlit-mark-color.png", width=50)
        st.title("Navigation")
        page = st.radio("Go to", ["🔍 Classify", "📊 Model Stats", "ℹ️ About"])

        st.markdown("---")
        st.markdown("### Model Info")
        st.info("""
        **Architecture:** EfficientNetV2-S
        **Classes:** 51 plankton species
        **Accuracy:** 57% (Top-1)
        **Training:** Transfer Learning
        """)

    # Load model
    with st.spinner("Loading AI model..."):
        model, idx_to_class, class_to_idx = load_model()

    if model is None:
        st.error("Failed to load model. Please train the model first using `python train_cnn.py`")
        return

    # Page routing
    if page == "🔍 Classify":
        classify_page(model, idx_to_class, class_to_idx)
    elif page == "📊 Model Stats":
        stats_page()
    else:
        about_page()


def classify_page(model, idx_to_class, class_to_idx):
    """Classification page."""
    st.header("Upload Image for Classification")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Plankton Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a microscopic plankton image"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Prediction button
            if st.button("🔬 Identify Species", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    predictions = predict_image(model, image, idx_to_class, top_k=5)

                # Store in session state
                st.session_state.predictions = predictions

    with col2:
        st.subheader("🎯 Prediction Results")

        if 'predictions' in st.session_state and st.session_state.predictions:
            predictions = st.session_state.predictions

            # Top prediction
            top_pred = predictions[0]
            st.markdown(f"""
            <div class="prediction-box">
                <h2>Predicted Species</h2>
                <h1>{top_pred['species'].replace('_', ' ').title()}</h1>
                <h3>Confidence: {top_pred['confidence']:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)

            # Top-5 predictions
            st.subheader("📊 Top 5 Predictions")

            # Create bar chart
            species_names = [p['species'].replace('_', ' ').title() for p in predictions]
            confidences = [p['confidence'] for p in predictions]

            fig = go.Figure(data=[
                go.Bar(
                    x=confidences,
                    y=species_names,
                    orientation='h',
                    marker=dict(
                        color=confidences,
                        colorscale='Viridis',
                        showscale=True
                    ),
                    text=[f"{c:.1f}%" for c in confidences],
                    textposition='auto',
                )
            ])

            fig.update_layout(
                xaxis_title="Confidence (%)",
                yaxis_title="Species",
                height=400,
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

            # Detailed results
            with st.expander("📋 View Detailed Results"):
                for i, pred in enumerate(predictions, 1):
                    st.markdown(f"""
                    **{i}. {pred['species'].replace('_', ' ').title()}**
                    Confidence: {pred['confidence']:.2f}%
                    """)
        else:
            st.info("👆 Upload an image and click 'Identify Species' to see results")


def stats_page():
    """Model statistics page."""
    st.header("📊 Model Performance Statistics")

    stats = load_statistics()

    if stats is None:
        st.warning("Training statistics not found. Train the model first.")
        return

    # Test results
    st.subheader("🎯 Test Set Performance")

    col1, col2, col3 = st.columns(3)

    test_results = stats['test_results']

    with col1:
        st.metric(
            "Top-1 Accuracy",
            f"{test_results['accuracy']*100:.2f}%",
            help="Percentage of correct species predictions"
        )

    with col2:
        st.metric(
            "Top-3 Accuracy",
            f"{test_results['top3_accuracy']*100:.2f}%",
            help="Correct species in top 3 predictions"
        )

    with col3:
        st.metric(
            "Top-5 Accuracy",
            f"{test_results['top5_accuracy']*100:.2f}%",
            help="Correct species in top 5 predictions"
        )

    # Training history
    st.subheader("📈 Training Progress")

    tab1, tab2 = st.tabs(["📉 Loss", "📈 Accuracy"])

    with tab1:
        # Loss plot
        frozen_loss = stats['frozen']['train_loss']
        frozen_val_loss = stats['frozen']['val_loss']
        finetune_loss = stats.get('finetune', {}).get('train_loss', [])
        finetune_val_loss = stats.get('finetune', {}).get('val_loss', [])

        epochs_frozen = list(range(1, len(frozen_loss) + 1))
        epochs_finetune = list(range(len(frozen_loss) + 1, len(frozen_loss) + len(finetune_loss) + 1))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs_frozen, y=frozen_loss, name='Train Loss (Frozen)', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=epochs_frozen, y=frozen_val_loss, name='Val Loss (Frozen)', line=dict(color='lightblue', dash='dash')))

        if finetune_loss:
            fig.add_trace(go.Scatter(x=epochs_finetune, y=finetune_loss, name='Train Loss (Fine-tune)', line=dict(color='red')))
            fig.add_trace(go.Scatter(x=epochs_finetune, y=finetune_val_loss, name='Val Loss (Fine-tune)', line=dict(color='lightcoral', dash='dash')))

        fig.update_layout(
            xaxis_title="Epoch",
            yaxis_title="Loss",
            height=500,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Accuracy plot
        frozen_acc = [a*100 for a in stats['frozen']['train_acc']]
        frozen_val_acc = [a*100 for a in stats['frozen']['val_acc']]
        finetune_acc = [a*100 for a in stats.get('finetune', {}).get('train_acc', [])]
        finetune_val_acc = [a*100 for a in stats.get('finetune', {}).get('val_acc', [])]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs_frozen, y=frozen_acc, name='Train Acc (Frozen)', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=epochs_frozen, y=frozen_val_acc, name='Val Acc (Frozen)', line=dict(color='lightgreen', dash='dash')))

        if finetune_acc:
            fig.add_trace(go.Scatter(x=epochs_finetune, y=finetune_acc, name='Train Acc (Fine-tune)', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=epochs_finetune, y=finetune_val_acc, name='Val Acc (Fine-tune)', line=dict(color='yellow', dash='dash')))

        fig.update_layout(
            xaxis_title="Epoch",
            yaxis_title="Accuracy (%)",
            height=500,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)


def about_page():
    """About page."""
    st.header("ℹ️ About This Project")

    st.markdown("""
    ## Plankton Species Classification using Deep Learning

    This project uses state-of-the-art deep learning to automatically identify microscopic plankton species
    from images captured under a microscope.

    ### 🎯 Project Highlights

    - **51 Plankton Species** classified
    - **12,886 Training Images** curated and preprocessed
    - **57% Top-1 Accuracy** achieved
    - **85% Top-5 Accuracy** - correct species in top 5 predictions

    ### 🧠 Model Architecture

    **EfficientNetV2-S** - A state-of-the-art convolutional neural network optimized for:
    - Efficient training and inference
    - Transfer learning from ImageNet
    - Superior accuracy with fewer parameters

    ### 🔬 Training Strategy

    **Two-Phase Training:**
    1. **Phase 1 (Frozen):** Train only classifier head (35 epochs)
    2. **Phase 2 (Fine-tune):** Unfreeze backbone layers (35 epochs)

    ### 📊 Dataset Preprocessing

    - Statistical threshold calculation for class balancing
    - Removal of classes with < 20 samples
    - Data augmentation: rotation, flip, color jitter
    - Stratified train/val/test split (70/15/15)

    ### 🛠️ Technologies Used

    - **PyTorch** - Deep learning framework
    - **EfficientNetV2** - Model architecture
    - **Streamlit** - Web interface
    - **Plotly** - Interactive visualizations

    ### 👨‍💻 Developed By

    **College Project** - Machine Learning & Computer Vision

    ### 📚 Use Cases

    - Marine biology research
    - Environmental monitoring
    - Educational tool for oceanography
    - Automated plankton surveys

    ---

    ### 🚀 Future Improvements

    - [ ] Increase accuracy to 70%+ with ensemble models
    - [ ] Add real-time video classification
    - [ ] Deploy as REST API
    - [ ] Mobile app development
    - [ ] Explainable AI (Grad-CAM visualizations)

    """)


if __name__ == "__main__":
    main()
