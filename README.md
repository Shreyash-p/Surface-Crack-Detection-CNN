# Industrial Surface Crack Detection using Deep CNN

An end-to-end computer vision solution designed for automated surface defect and crack inspection in civil and industrial infrastructure. Built with TensorFlow/Keras, this system processes raw surface imagery, applies real-time data augmentations, and trains a deep 4-block Convolutional Neural Network to perform robust binary classification (Crack vs. No Crack).

## Key Features

- **Automated Pipeline**: Organizes raw images into reproducible 70% Train, 15% Validation, and 15% Test splits with zero data leakage.
- **Robust Feature Extraction**: 4-stage convolutional backbone (32 to 256 filters) coupled with Batch Normalization and MaxPooling for stable spatial representations.
- **Overfitting Prevention**: Real-time image augmentation (rotations, spatial shifts, zooming, horizontal flips) combined with deep Dropout regularization (0.5 and 0.3).
- **Industrial Training Guards**: Employs `EarlyStopping`, `ModelCheckpoint`, and adaptive learning rate decay via `ReduceLROnPlateau`.
- **Diagnostic Metrics**: Evaluates model performance via Confusion Matrix, Precision, Recall, and F1-score to prioritize defect capture (minimizing false negatives).
- **Single-Image Inference**: Standalone preprocessing and prediction module for classifying individual test samples.

## Tech Stack

- **Framework**: TensorFlow 2.x / Keras
- **Computer Vision & Imaging**: OpenCV / PIL, ImageDataGenerator
- **Data & Evaluation**: NumPy, Scikit-Learn
- **Visualization**: Matplotlib
