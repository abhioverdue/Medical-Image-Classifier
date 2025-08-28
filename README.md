# Medical Image Classification Using ResNet18

A deep learning pipeline for classifying chest X-ray images into multiple categories using PyTorch and transfer learning with ResNet18. This project focuses on medical diagnosis assistance by automatically categorizing X-ray images as **Normal**, **Pneumonia**, or **Tuberculosis (TB)**.

## Overview

This project implements a robust medical image classification system that leverages transfer learning to achieve high accuracy on chest X-ray diagnosis. The model uses a pre-trained ResNet18 architecture, fine-tuned specifically for medical imaging tasks with **>93% accuracy**.

## Key Features

- **Transfer Learning**: Fine-tuned ResNet18 with strategically frozen layers for efficient training
- **Data Augmentation**: Comprehensive augmentation pipeline including rotation, flips, affine transforms, and color jitter
- **Class Imbalance Handling**: Weighted cross-entropy loss to address uneven dataset distribution
- **Comprehensive Evaluation**: Multiple metrics including accuracy, F1-score, precision, recall, and confusion matrices
- **Training Visualization**: Real-time plotting of loss, accuracy, and F1-score across epochs
- **Smart Checkpointing**: Automatic model saving with early stopping based on validation performance
- **Medical-Grade Accuracy**: Optimized for reliable medical diagnosis assistance achieving 93% accuracy

## Project Structure

```
Medical-Image-Classification/
│
├── data/                           # Dataset directory (download separately)
│   ├── train/
│   │   ├── NORMAL/
│   │   ├── PNEUMONIA/
│   │   └── TUBERCULOSIS/
│   ├── val/
│   │   ├── NORMAL/
│   │   ├── PNEUMONIA/
│   │   └── TUBERCULOSIS/
│   └── test/
│       ├── NORMAL/
│       ├── PNEUMONIA/
│       └── TUBERCULOSIS/
│
├── notebooks/
│   └── medical_classification.ipynb    # Main Jupyter notebook
│
├── src/
│   ├── data_loader.py                  # Dataset loading and preprocessing
│   ├── model.py                        # ResNet18 model architecture
│   ├── train.py                        # Training loop with validation
│   ├── evaluate.py                     # Evaluation metrics and analysis
│   └── utils.py                        # Utility functions and visualizations
│
├── requirements.txt                    # Python dependencies
├── README.md                          # Project documentation
└── .gitignore                         # Git ignore rules
```

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 2GB+ available disk space

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/Medical-Image-Classification.git
   cd Medical-Image-Classification
   ```

2. **Create virtual environment**
   ```bash
   python -m venv medical_env
   source medical_env/bin/activate  # On Windows: medical_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Dataset Setup

1. **Download chest X-ray dataset** from one of these sources:
   - [Kaggle Chest X-Ray Images](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
   - [NIH Chest X-ray Dataset](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community)

2. **Organize the dataset** according to the project structure shown above

3. **Verify dataset structure**
   ```bash
   python src/utils.py --verify-data
   ```

## Usage

### Training the Model

#### Option 1: Using Jupyter Notebook (Recommended)
```bash
jupyter notebook notebooks/medical_classification.ipynb
```

#### Option 2: Using Python Scripts
```bash
# Train the model
python src/train.py --epochs 50 --batch-size 32 --learning-rate 0.001

# Evaluate the model
python src/evaluate.py --model-path checkpoints/best_model.pth
```

### Key Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 50 | Number of training epochs |
| `--batch-size` | 32 | Training batch size |
| `--learning-rate` | 0.001 | Initial learning rate |
| `--weight-decay` | 1e-4 | L2 regularization strength |
| `--early-stopping` | 10 | Early stopping patience |

## Model Architecture

The project uses **ResNet18** as the backbone architecture with the following modifications:

- **Input Layer**: Adapted for grayscale chest X-ray images (1 channel)
- **Frozen Layers**: Early convolutional layers frozen to preserve low-level feature extraction
- **Custom Classifier**: Final fully connected layer modified for 3-class classification
- **Dropout**: Added for regularization (0.5 dropout rate)

## Data Preprocessing & Augmentation

### Training Augmentations
- Random rotation (±15 degrees)
- Random horizontal flip (50% probability)
- Random affine transforms
- Color jitter (brightness, contrast adjustment)
- Normalization using ImageNet statistics

### Validation/Test Preprocessing
- Resize to 224×224 pixels
- Tensor conversion
- Normalization only

## Performance Metrics

The model is evaluated using multiple metrics appropriate for medical diagnosis:

- **Accuracy**: Overall classification accuracy
- **Precision**: True positive rate per class
- **Recall (Sensitivity)**: Ability to identify positive cases
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown
- **ROC Curves**: Performance across different thresholds

### Achieved Performance
- **Overall Accuracy**: 93%
- **Normal X-rays**: High precision and recall for healthy cases
- **Pneumonia**: Reliable detection of pneumonia cases
- **Tuberculosis**: Accurate identification of TB cases


## Important Medical Disclaimer

**This model is for research and educational purposes only. It should NOT be used as the sole basis for medical diagnosis. Always consult qualified healthcare professionals for medical decisions.**

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation as needed
- Ensure HIPAA compliance for any medical data handling

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Reduce batch size
python src/train.py --batch-size 16
```

**Dataset Not Found**
```bash
# Verify data structure
python src/utils.py --check-dataset
```

**Poor Model Performance**
- Ensure dataset is properly balanced
- Check data preprocessing pipeline
- Verify learning rate scheduling
- Consider increasing training epochs

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


**Made for advancing medical AI research**
