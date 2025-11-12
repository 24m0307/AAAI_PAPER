# 🍎 Apple Leaf Disease Segmentation — Complete Training Suite

A comprehensive deep learning framework for apple leaf disease segmentation and severity analysis using state-of-the-art neural network architectures.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🌟 Features

- **Multi-Model Architecture Support**: U-Net, FCN, PSPNet, and SegFormer implementations
- **Enhanced Disease Severity Analysis**: Detailed metrics with treatment recommendations
- **Comprehensive Visualizations**: Training curves, confusion matrices, and performance comparisons
- **Experiment Tracking**: Integrated Weights & Biases support
- **Paper-Ready Outputs**: Publication-quality reports and visualizations
- **Advanced Data Augmentation**: Disease-aware oversampling and robust preprocessing

## 📋 Table of Contents

- [Model Architectures](#model-architectures)
- [Installation](#installation)
- [Dataset Structure](#dataset-structure)
- [Configuration](#configuration)
- [Usage](#usage)
- [Severity Analysis](#severity-analysis)
- [Output Structure](#output-structure)
- [Citation](#citation)

## 🏗️ Model Architectures

### 1. U-Net with MobileNetV2 Encoder
- **Encoder**: Pre-trained MobileNetV2 for efficient feature extraction
- **Decoder**: Symmetric expanding path with skip connections
- **Best For**: Lightweight deployment, medical imaging tasks

### 2. FCN (Fully Convolutional Network)
- **Backbone**: ResNet50 for robust feature extraction
- **Architecture**: Fully convolutional with upsampling layers
- **Best For**: Dense prediction tasks, spatial information preservation

### 3. PSPNet (Pyramid Scene Parsing Network)
- **Backbone**: ResNet50 with pyramid pooling module
- **Multi-Scale**: Captures contextual information at different scales
- **Best For**: Scene understanding, multi-scale object detection

### 4. SegFormer (Simplified)
- **Multi-Scale**: Feature extraction at multiple resolutions
- **Feature Fusion**: Intelligent combination of multi-scale features
- **Best For**: Modern architecture with balanced accuracy and efficiency

## 🚀 Installation

### System Requirements

- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: 16GB minimum (32GB recommended for larger batch sizes)
- **Storage**: 10GB+ for datasets and outputs

### Core Dependencies

```bash
# Core deep learning framework
pip install tensorflow==2.13.0

# Image processing and visualization
pip install opencv-python matplotlib seaborn

# Scientific computing and utilities
pip install scikit-learn tqdm pandas numpy

# Enhanced visualizations
pip install scikit-image plotly
```

### Optional Dependencies

```bash
# Experiment tracking (recommended)
pip install wandb

# For PDF generation and LaTeX tables
pip install pdflatex  # System package, or use included functionality
```

### GPU Setup

The framework automatically detects and configures GPU:

```python
def set_gpu_growth():
    """
    Enables memory growth for GPU to prevent OOM errors.
    Automatically called during initialization.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
```

### Environment Variables

The script automatically sets optimal TensorFlow configurations:

```python
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"      # Reduce TF logging
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"     # Disable oneDNN warnings
os.environ["TF_DISABLE_PROFILER"] = "1"       # Disable profiler
```

## 📁 Dataset Structure

Organize your dataset in the following structure:

```
Apple/
├── Brown spot/
│   ├── image/
│   │   ├── IMG_001.jpg
│   │   ├── IMG_002.jpg
│   │   └── ...
│   └── label/
│       ├── IMG_001.png
│       ├── IMG_002.png
│       └── ...
├── Alternaria leaf spot/
│   ├── image/
│   └── label/
├── Gray spot/
│   ├── image/
│   └── label/
├── Rust/
│   ├── image/
│   └── label/
└── Healthy/
    ├── image/
    └── label/
```

### Class Definitions

| Class | RGB Value | Description |
|-------|-----------|-------------|
| Background | `(0, 0, 0)` | Background pixels |
| Healthy | `(128, 0, 0)` | Healthy leaf tissue |
| Brown spot | `(128, 0, 128)` | Brown spot disease |
| Alternaria leaf spot | `(128, 128, 0)` | Alternaria infection |
| Gray spot | `(0, 0, 128)` | Gray spot disease |
| Rust | `(0, 128, 0)` | Rust disease |

## ⚙️ Configuration

Key configuration parameters in the training script:

```python
# Image and training parameters
IMG_SIZE = 256           # Input image size (256x256)
BATCH_SIZE = 8          # Training batch size
EPOCHS = 50             # Number of training epochs
NUM_CLASSES = 6         # Background + 5 disease classes

# Model selection
RUN_ONLY = ['UNet_MobileNetV2', 'FCN', 'PSPNet', 'SegFormer']

# Training optimization
AUTO_CLASS_WEIGHTS = True      # Auto-compute class weights from data
OVERSAMPLE_DISEASE = True      # Disease-aware oversampling
SKIP_TRAIN_IF_CKPT = True     # Skip training if checkpoint exists
TOLERANT_LABEL_COLORS = True  # Allow color tolerance in labels

# Visualization options
USE_TTA_IN_VIZ = True         # Use Test-Time Augmentation
OVERLAY_ALPHA = 0.45          # Overlay transparency

# Augmentation probabilities
A_ROT90_PROB = 0.75           # 90° rotation
A_FLIP_H_PROB = 0.5           # Horizontal flip
A_FLIP_V_PROB = 0.5           # Vertical flip
A_JITTER_PROB = 0.6           # Color jittering
A_NOISE_PROB = 0.3            # Gaussian noise
A_CROP_PROB = 0.7             # Random crop
CROP_MIN_FRAC = 0.65          # Minimum crop fraction (aggressive zoom)
```

### Stratified Data Split

The framework uses intelligent stratification:
- Creates discrete labels based on dominant disease class
- Ensures balanced representation across train/validation sets
- Falls back to random split if classes have <2 samples
- Default split: 80% train, 20% validation

### Automatic Class Weight Computation

When `AUTO_CLASS_WEIGHTS=True`:
```python
# Inverse frequency weighting with smoothing
weight_i = 1.0 / (class_freq_i + 1e-7)

# Normalized to sum to NUM_CLASSES
weights = weights / weights.sum() * NUM_CLASSES
```

This helps handle severe class imbalance in medical imaging datasets.

## 💻 Usage

### Configuration Parameters

Edit the configuration section at the top of the script:

```python
# Key Configuration
BASE_DIR = "/path/to/your/Apple"  # Dataset directory
IMG_SIZE = 256                     # Image size (try 384/512 for higher IoU)
BATCH_SIZE = 8                     # Batch size
EPOCHS = 50                        # Training epochs
SEED = 2025                        # Random seed for reproducibility

# Training options
SKIP_TRAIN_IF_CKPT = True         # Skip training if checkpoint exists
RUN_ONLY = ['UNet_MobileNetV2', 'FCN', 'PSPNet', 'SegFormer']
AUTO_CLASS_WEIGHTS = True          # Automatically compute class weights
OVERSAMPLE_DISEASE = True          # Disease-aware oversampling
USE_TTA_IN_VIZ = True             # Test-time augmentation in visualization

# Augmentation parameters
A_ROT90_PROB = 0.75               # Rotation probability
A_FLIP_H_PROB = 0.5               # Horizontal flip probability
A_FLIP_V_PROB = 0.5               # Vertical flip probability
A_CROP_PROB = 0.7                 # Random crop probability
CROP_MIN_FRAC = 0.65              # Minimum crop fraction (aggressive zoom)

# Inference options
DO_INFERENCE_AFTER_TRAIN = True   # Run inference after training
TEST_IMAGE_PATH = "/path/to/test/image.jpg"  # Test image path
```

### Basic Training

```python
# Run the complete training suite
python train.py

# The script will automatically:
# - Load and preprocess data with stratified split
# - Train all models specified in RUN_ONLY
# - Generate comprehensive metrics and visualizations
# - Create confusion matrices for all models
# - Perform severity analysis on validation set
# - Save all outputs to the outputs/ directory
```

### Weights & Biases Integration

```python
# Set your W&B API key in the script
wandb.login(key="YOUR_WANDB_API_KEY")

# W&B automatically tracks:
# - Training metrics (loss, accuracy, IoU)
# - Model architectures and parameters
# - Confusion matrices and visualizations
# - Severity analysis results
# - Performance comparisons
```

### Model Evaluation

```python
# Comprehensive metrics computation
from training_script import compute_model_metrics

metrics = compute_model_metrics(model, X_val, Y_val)
# Returns:
# - mean_iou: Mean Intersection over Union
# - accuracy: Overall pixel accuracy
# - disease_accuracy: Binary disease detection accuracy
# - iou_per_class: Per-class IoU scores
# - confusion_matrix: Full confusion matrix
# - predictions: Prediction masks
```

### Severity Analysis

```python
from training_script import SeverityIndexCalculator

severity_calculator = SeverityIndexCalculator()
result = severity_calculator.compute_detailed_severity(pred_mask)

# Comprehensive severity metrics
print(f"Severity Level: {result['severity_level']}")
print(f"Severity Score: {result['weighted_severity_score']:.2f}")
print(f"Dominant Disease: {result['dominant_disease']}")
print(f"Leaf Quality Score: {result['leaf_quality_score']}/100")
print(f"Risk Assessment: {result['risk_assessment']}")
print(f"Treatment Priority: {result['treatment_priority']}")

# Disease distribution analysis
for disease, stats in result['disease_distribution'].items():
    print(f"{disease}: {stats['num_spots']} spots, "
          f"avg size: {stats['average_spot_size']:.1f} pixels")
```

### Custom Inference with Severity Analysis

```python
# Run inference on a single image with all trained models
from training_script import inference_on_test_image

results = inference_on_test_image(trained_models, "path/to/image.jpg")

# Results include:
# - Prediction masks for each model
# - Comprehensive severity analysis
# - Labeled visualizations with percentages
# - Disease distribution reports
# - Treatment recommendations
```

## 📊 Severity Classification

The system classifies disease severity into five levels with detailed analysis:

| Level | Description | Disease Area % | Treatment |
|-------|-------------|----------------|-----------|
| 🟢 Healthy | No disease detected | 0-5% | Preventive care only |
| 🟡 Mild | Early stage infection | 5-15% | Monitor closely, consider preventive treatment |
| 🟠 Moderate | Noticeable symptoms | 15-30% | Active treatment recommended |
| 🔴 Severe | Significant damage | 30-50% | Immediate treatment required |
| ⚫ Critical | Extensive damage | >50% | Emergency treatment - consider plant removal |

### Disease Severity Weights

Different diseases have different impact levels (used in weighted severity score):

```python
disease_weights = {
    'Brown spot': 1.0,         # Moderate impact
    'Alternaria leaf spot': 1.2, # High impact (aggressive pathogen)
    'Gray spot': 0.8,          # Mild impact
    'Rust': 1.5                # Very high impact (spreading pathogen)
}
```

### Output Metrics

- **Weighted Severity Score**: Composite disease severity metric (0-100+)
- **Dominant Disease Type**: Primary infection identified
- **Disease Distribution**: Percentage breakdown by disease class
- **Disease Spots Analysis**: 
  - Number of infection spots per disease
  - Average spot size (pixels)
  - Largest spot size
  - Spatial distribution patterns
- **Risk Assessment**: 
  - High disease burden detection
  - Multiple disease co-infection analysis
  - Pathogen aggressiveness evaluation
- **Treatment Priority**: Actionable recommendations
- **Leaf Quality Score**: Overall health rating (0-100, inverse of severity)

### Connected Components Analysis

For each disease type, the system analyzes:
- **Number of Spots**: Individual infection sites
- **Average Spot Size**: Mean area of infection regions
- **Largest Spot**: Maximum contiguous infected area
- **Total Affected Pixels**: Complete disease coverage

This helps distinguish between:
- Few large lesions vs. many small spots
- Concentrated vs. distributed infections
- Early vs. advanced disease stages

## 🎨 Advanced Features

### Loss Functions

The framework uses a combined loss function for optimal training:

```python
# Combined Loss = α * Weighted CE + (1-α) * Focal Tversky
loss = combined_loss(class_weights, alpha=0.5)

# Components:
# 1. Weighted Categorical Cross-Entropy
#    - Handles class imbalance
#    - Custom weights per class
#
# 2. Focal Tversky Loss
#    - alpha=0.7, beta=0.3, gamma=2.0
#    - Focuses on hard-to-segment regions
#    - Better handling of false negatives
```

### Data Augmentation Pipeline

- **Geometric Transforms**: Rotation, flipping (75% probability)
- **Random Cropping**: Aggressive zoom with 65% minimum fraction
- **Color Jittering**: Brightness, contrast, saturation adjustments
- **Noise Injection**: Gaussian noise for model robustness
- **Disease-Aware Oversampling**: Balanced class representation

### Training Callbacks

- Model checkpointing (best validation IoU)
- Early stopping with patience
- Learning rate reduction on plateau
- Weights & Biases integration

## 📂 Output Structure

```
outputs/
├── paper_report/                 # Publication-ready materials
│   ├── training_comparison.png   # Multi-model training curves
│   ├── class_distribution.png    # Dataset class balance
│   ├── performance_radar.png     # Model comparison radar chart
│   ├── architecture_*.png        # Model architecture diagrams
│   ├── attention_*.png           # Attention heatmaps
│   ├── model_comparison.tex      # LaTeX table for papers
│   └── model_summary.csv         # Performance summary
├── model_best_weights/           # Trained model checkpoints
│   ├── UNet_MobileNetV2_best.h5
│   ├── FCN_best.h5
│   ├── PSPNet_best.h5
│   └── SegFormer_best.h5
├── confusion_matrices/           # Comprehensive evaluation matrices
│   ├── confusion_matrix_*.txt         # Text format
│   ├── confusion_matrix_*.png         # Normalized visualization
│   ├── detailed_confusion_matrix_*.png # Raw counts + percentages
│   ├── confusion_matrices_comparison_grid.png
│   └── all_confusion_matrices.json    # JSON data
├── severity_analysis/            # Disease severity reports
│   ├── comprehensive_report_*.png     # Full severity analysis
│   ├── labeled_prediction_*.png       # Predictions with % labels
│   ├── severity_data_*.json           # Severity metrics (JSON)
│   └── severity_report_*.txt          # Text reports
├── viz_with_severity_*.png       # Severity visualizations (3-column)
├── panel_severity_*.png          # Grid panel visualizations
├── comprehensive_metrics.png     # All training metrics dashboard
├── loss_evolution.png            # Loss curves comparison
├── accuracy_evolution.png        # Accuracy curves comparison
├── iou_evolution.png            # IoU curves comparison
├── final_performance_comparison.png   # Bar chart comparison
├── performance_summary.png       # Summary statistics
├── predictions_comparison.png    # Model prediction grids
├── model_comparison.csv          # Performance comparison table
├── model_summary_statistics.csv  # Detailed statistics
└── model_comparison_labeled.png  # Labeled comparison grid
```

### Key Output Files

- **Training Metrics**: Comprehensive plots tracking loss, accuracy, and IoU across all models
- **Confusion Matrices**: Both raw counts and normalized percentages for detailed error analysis
- **Severity Reports**: Complete disease analysis with treatment recommendations
- **Labeled Visualizations**: Predictions with percentage breakdowns overlaid
- **Paper-Ready Materials**: High-quality figures and LaTeX tables for publication

## 📈 Model Performance

Comprehensive evaluation metrics include:

- **Mean IoU** (Intersection over Union)
- **Per-Class Accuracy** with confusion matrices
- **Disease Detection Accuracy** (Binary: healthy vs diseased)
- **Training/Validation Metrics** tracking
- **Model Comparison Visualizations**
- **Per-Class IoU Scores** for each disease type

### Evaluation Features

#### Confusion Matrix Analysis
- **Raw Counts**: Absolute pixel-level predictions
- **Normalized Percentages**: Accuracy per class
- **Multi-Model Comparison**: Side-by-side visualization
- **JSON Export**: Machine-readable confusion matrix data

#### Custom Metrics
```python
class IoUMetric(keras.metrics.Metric):
    """
    Custom IoU metric that:
    - Maintains confusion matrix across batches
    - Computes per-class IoU
    - Returns mean IoU (excluding background)
    """
```

#### Test-Time Augmentation (TTA)
The framework supports TTA for improved inference accuracy:
- Horizontal flip
- Vertical flip
- 90° rotation
- Average ensemble of predictions

#### Small Component Cleanup
Post-processing removes noise:
- Identifies connected components
- Removes tiny isolated regions (<0.1% of image)
- Reassigns to "Healthy" class
- Improves visual quality and accuracy

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to new functions and classes
- Include unit tests for new features
- Update documentation for API changes

### Areas for Contribution

- Additional model architectures
- New augmentation techniques
- Enhanced severity metrics
- Performance optimizations
- Mobile deployment support

## 🐛 Troubleshooting

### Common Issues

**Out of Memory (OOM) Errors**
```python
# Reduce batch size
BATCH_SIZE = 4  # or 2

# Reduce image size
IMG_SIZE = 128  # or 224
```

**Color Mapping Issues**
```python
# Enable tolerant color matching
TOLERANT_LABEL_COLORS = True  # Allows ±10 RGB tolerance
```

**Training Too Slow**
```python
# Reduce augmentation probability
A_ROT90_PROB = 0.5
A_CROP_PROB = 0.5

# Disable TTA in visualization
USE_TTA_IN_VIZ = False
```

**Model Not Learning**
```python
# Check class weights
AUTO_CLASS_WEIGHTS = True

# Enable disease oversampling
OVERSAMPLE_DISEASE = True

# Increase learning rate
optimizer = keras.optimizers.Adam(learning_rate=5e-4)
```

## 🔬 Technical Details

### Model Architecture Details

**U-Net with MobileNetV2**
- Encoder: MobileNetV2 (ImageNet pretrained)
- Skip connections at 5 scales: 128×128, 64×64, 32×32, 16×16, 8×8
- Decoder: Symmetric upsampling with concatenation
- Parameters: ~5-10M (lightweight)

**FCN-32s**
- Backbone: ResNet50 (ImageNet pretrained)
- Single-scale upsampling (32×)
- Parameters: ~25M

**PSPNet**
- Backbone: ResNet50
- Pyramid pooling at 4 scales: 1×1, 2×2, 4×4, 8×8
- Global context aggregation
- Parameters: ~50M

**SegFormer (Simplified)**
- Multi-scale feature extraction at 4 levels
- Feature fusion with upsampling
- Lightweight encoder-decoder
- Parameters: ~10-15M

### Performance Optimization Tips

1. **Mixed Precision Training** (for RTX GPUs):
```python
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
```

2. **XLA Compilation**:
```python
model.compile(
    optimizer=...,
    loss=...,
    jit_compile=True  # Enable XLA
)
```

3. **Prefetching Optimization**:
```python
AUTOTUNE = tf.data.AUTOTUNE
dataset = dataset.prefetch(buffer_size=AUTOTUNE)
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{apple_leaf_segmentation_2024,
  title = {Apple Leaf Disease Segmentation with Enhanced Severity Analysis},
  author = {IXRBHIII},
  year = {2024},
  url = {https://github.com/ixrbhiii/apple-leaf-segmentation},
  note = {Comprehensive multi-model framework for plant disease analysis}
}
```

## 🙏 Acknowledgments

- Pre-trained models from TensorFlow/Keras Applications
- Inspired by medical imaging segmentation research
- Built with contributions from the deep learning community

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Made with ❤️ for advancing agricultural AI**
