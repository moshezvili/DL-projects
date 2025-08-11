# 🚀 Setup Guide

This guide will help you set up the environment to run the deep learning projects in this repository.

## 📋 Prerequisites

- Python 3.8 or higher
- Git
- CUDA-compatible GPU (recommended for faster training)

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/DL-projects.git
cd DL-projects
```

### 2. Create Virtual Environment

```bash
# Using conda (recommended)
conda create -n dl-portfolio python=3.9
conda activate dl-portfolio

# Or using venv
python -m venv dl-portfolio
# On Windows:
dl-portfolio\Scripts\activate
# On macOS/Linux:
source dl-portfolio/bin/activate
```

### 3. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# For GPU support (if you have CUDA-compatible GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Additional Setup for Specific Projects

#### LogGPT Project
```bash
cd wp-log-ai
# Install additional dependencies for LogGPT
pip install unsloth
pip install lm-format-enforcer
```

#### Workout Classification Project
```bash
# Install Kaggle API for dataset download
pip install kaggle
# Configure Kaggle credentials (follow Kaggle API documentation)
```

## 🎯 Running the Projects

### Jupyter Notebooks

1. Start Jupyter Lab:
```bash
jupyter lab
```

2. Navigate to the project notebook:
   - `clalit_home_task.ipynb` - Workout Classification
   - `Home_Assignment.ipynb` - Empathetic AI Agent
   - `wp-log-ai/LogGPT_Training.ipynb` - LogGPT Training

### Python Scripts

For LogGPT project:
```bash
cd wp-log-ai
python train.py
python create_data_set.py
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# API Keys (if needed)
OPENAI_API_KEY=your_openai_key_here
HUGGINGFACE_TOKEN=your_hf_token_here

# Kaggle API (for dataset downloads)
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key
```

### GPU Setup

To verify GPU availability:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
```

## 📊 Dataset Setup

### Workout Classification Dataset

```python
import kagglehub
path = kagglehub.dataset_download("hasyimabdillah/workoutfitness-video")
```

### LogGPT Dataset

The WordPress logs are included in the `wp-log-ai` directory as `wordpress-logs.txt`.

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size in training scripts
   - Use gradient accumulation
   - Enable mixed precision training

2. **MediaPipe Installation Issues**:
   ```bash
   pip install mediapipe --no-deps
   pip install opencv-python
   ```

3. **Transformers Version Conflicts**:
   ```bash
   pip install transformers==4.36.0 --force-reinstall
   ```

### System Requirements

- **RAM**: Minimum 8GB, recommended 16GB+
- **Storage**: At least 5GB free space for datasets and models
- **GPU**: NVIDIA GPU with 8GB+ VRAM for optimal performance

## 📞 Support

If you encounter any issues:

1. Check the [Issues](../../issues) page for known problems
2. Create a new issue with detailed error messages
3. Include your environment information:
   ```bash
   python --version
   pip list | grep torch
   nvidia-smi  # if using GPU
   ```

## 🎓 Learning Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [MediaPipe Documentation](https://mediapipe.dev/)
- [Jupyter Lab User Guide](https://jupyterlab.readthedocs.io/)

---

Happy coding! 🚀
