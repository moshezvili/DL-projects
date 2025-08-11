# 🏋️ Workout Fitness Classification

A computer vision project that classifies workout fitness activities using MediaPipe pose estimation and deep learning.

## 🎯 Project Overview

This project demonstrates advanced computer vision techniques for real-time fitness activity recognition. Using MediaPipe for pose detection and a custom PyTorch neural network, the system can accurately classify different workout activities from video data.

## ✨ Key Features

- **Pose Estimation**: MediaPipe integration for accurate human pose detection
- **Custom Neural Network**: PyTorch-based classification model
- **Video Processing**: Real-time frame analysis and activity recognition
- **Data Visualization**: Comprehensive plotting and analysis tools
- **Model Evaluation**: Detailed performance metrics and confusion matrices

## 🛠️ Technologies Used

- **Deep Learning**: PyTorch, TorchVision
- **Computer Vision**: MediaPipe, OpenCV
- **Data Science**: NumPy, Pandas, Matplotlib
- **Model Training**: Scikit-learn, Custom loss functions
- **Data Processing**: Video frame extraction, pose landmark detection

## 📊 Dataset

The project uses the Kaggle Workout Fitness Video dataset, featuring multiple exercise categories for classification.

## 🚀 Getting Started

1. **Install Dependencies**:
   ```bash
   pip install torch torchvision mediapipe opencv-python numpy pandas matplotlib scikit-learn
   ```

2. **Download Dataset**:
   ```python
   import kagglehub
   path = kagglehub.dataset_download("hasyimabdillah/workoutfitness-video")
   ```

3. **Run the Notebook**:
   Open `clalit_home_task.ipynb` and execute the cells sequentially.

## 🏗️ Architecture

The project implements a complete machine learning pipeline:

1. **Data Loading**: Video dataset processing and organization
2. **Pose Detection**: MediaPipe-based landmark extraction
3. **Feature Engineering**: Pose coordinate preprocessing and normalization
4. **Model Training**: Custom PyTorch neural network with appropriate loss functions
5. **Evaluation**: Comprehensive metrics and visualization

## 📈 Results

The model achieves strong performance in classifying various workout activities, demonstrating the effectiveness of pose-based activity recognition.

## 🔍 Key Components

- **MediaPipe Integration**: Efficient pose detection and landmark extraction
- **Custom Dataset Class**: PyTorch-compatible data loading and preprocessing
- **Neural Network Architecture**: Tailored for pose-based classification
- **Training Pipeline**: Complete training loop with validation and monitoring
- **Evaluation Metrics**: Confusion matrices, classification reports, and visualizations

## 📝 Notes

This project was developed as part of a home assignment, showcasing practical applications of computer vision in fitness and health monitoring applications.

---

*For more details, please refer to the complete notebook implementation.*
