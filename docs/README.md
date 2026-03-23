# Plankton Species Classifier

**CNN vs Traditional Machine Learning - A Head-to-Head Comparison**

This project compares deep learning (EfficientNetV2) with traditional machine learning (SVM, Random Forest) approaches for classifying plankton species using the WHOI Plankton 2014 dataset.

## Dataset

- **Source**: WHOI Plankton 2014
- **Species**: 94 classes
- **Images**: ~330,000 PNG images
- **Challenge**: Highly imbalanced (1 to 266K images per class)

## System Requirements

- **Python**: 3.11+
- **GPU**: NVIDIA GPU with 8GB+ VRAM (tested on RTX 4060 Ti)
- **CUDA**: 12.1
- **RAM**: 16GB+ recommended

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Plankton-species-classifier
```

### 2. Install PyTorch with CUDA

```bash
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Other Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Project Structure

```
Plankton-species-classifier/
├── 2014/                      # Dataset directory (94 species folders)
├── outputs/                   # Generated outputs
│   ├── models/               # Saved model files
│   ├── logs/                 # Training logs
│   └── results/              # Evaluation results and plots
├── models/                    # Model definitions
│   ├── __init__.py
│   ├── cnn_model.py          # EfficientNetV2 CNN model
│   └── traditional_model.py  # SVM and Random Forest
├── config.py                  # Configuration settings
├── dataset.py                 # Data loading and preprocessing
├── train_cnn.py              # CNN training script
├── train_traditional.py      # Traditional ML training script
├── evaluate.py               # Model evaluation and comparison
├── requirements.txt
└── README.md
```

## Usage

### Training the CNN Model (Recommended)

```bash
# Full training with default settings (EfficientNetV2-S)
python train_cnn.py

# Custom training
python train_cnn.py --batch-size 16 --epochs 20

# Skip fine-tuning phase (faster, slightly lower accuracy)
python train_cnn.py --no-finetune
```

**Training Phases:**
1. **Frozen Phase**: Train only the classifier head (15 epochs)
2. **Fine-tuning Phase**: Unfreeze last 50 layers (15 epochs)

**Expected Results:**
- Training time: 1-2 hours on RTX 4060 Ti
- Accuracy: 85-95% (Top-1), 95-99% (Top-5)

### Training Traditional ML Models

```bash
# Train both SVM and Random Forest
python train_traditional.py

# Train only SVM
python train_traditional.py --model svm

# Train only Random Forest
python train_traditional.py --model rf

# Limit samples (faster training)
python train_traditional.py --max-samples 10000
```

**Note:** Traditional ML training can take several hours due to feature extraction.

**Expected Results:**
- Training time: 2-8 hours
- Accuracy: 60-75% (Top-1)

### Evaluating Models

```bash
# Evaluate all trained models
python evaluate.py

# Evaluate only CNN
python evaluate.py --model cnn

# Generate visualizations
python evaluate.py --visualize
```

## Configuration

Edit `config.py` to customize settings:

### CNN Settings
```python
CNN_CONFIG = {
    'model_name': 'efficientnet_v2_s',  # Model architecture
    'batch_size': 32,                    # Reduce if OOM errors
    'epochs_frozen': 15,                 # Phase 1 epochs
    'epochs_finetune': 15,               # Phase 2 epochs
    'learning_rate_frozen': 1e-3,
    'learning_rate_finetune': 1e-5,
    'mixed_precision': True,             # Faster training on RTX cards
}
```

### Class Balancing
```python
# Handle imbalanced dataset
CLASS_BALANCE_STRATEGY = 'sqrt_weighted'  # Options: 'undersample', 'oversample', 'weighted', 'sqrt_weighted'
MIN_SAMPLES_PER_CLASS = 5                 # Skip classes with fewer samples
MAX_SAMPLES_PER_CLASS = 5000              # Cap samples per class
```

## Troubleshooting

### Out of Memory (OOM) Errors

1. Reduce batch size:
   ```bash
   python train_cnn.py --batch-size 16
   ```

2. In `config.py`, set:
   ```python
   'gradient_accumulation_steps': 4  # Increase this
   ```

### Slow Training

1. Ensure CUDA is being used:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   ```

2. Enable mixed precision (already enabled by default):
   ```python
   'mixed_precision': True
   ```

### Class Imbalance Issues

The dataset is highly imbalanced. The project handles this through:
- **Weighted sampling**: Oversample minority classes during training
- **Class weights**: Apply inverse frequency weights to loss function
- **Square root weighting**: Balance between uniform and frequency-based weights

## Model Comparison

| Metric | CNN (EfficientNetV2) | SVM | Random Forest |
|--------|---------------------|-----|---------------|
| Top-1 Accuracy | 85-95% | 60-70% | 55-65% |
| Top-5 Accuracy | 95-99% | 80-88% | 75-85% |
| Training Time | 1-2 hours | 4-8 hours | 2-4 hours |
| Inference Speed | Fast (GPU) | Medium | Fast |
| Model Size | ~30 MB | ~5-50 MB | ~50-200 MB |

## Key Files

- **`train_cnn.py`**: Main CNN training script with two-phase training
- **`train_traditional.py`**: Traditional ML training with feature extraction
- **`evaluate.py`**: Compare all models and generate reports
- **`dataset.py`**: Data loading with augmentation and class balancing
- **`models/cnn_model.py`**: EfficientNetV2 with custom classifier head

## License

MIT License
