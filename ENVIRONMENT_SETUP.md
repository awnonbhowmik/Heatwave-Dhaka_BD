# Heatwave Analysis Environment Setup

## 🚀 Environment Overview

**Environment Name:** `heatwave-analysis`  
**Python Version:** 3.11.13  
**TensorFlow Version:** 2.20.0 (with GPU support)  
**CUDA Version:** 12.2 (compatible)  
**GPU:** NVIDIA GeForce RTX 2060 (4.7GB VRAM)

## ✅ Installation Status

✅ **TensorFlow GPU** - Working correctly with CUDA 12.2  
✅ **Core Data Science** - pandas, numpy, scipy, matplotlib, seaborn  
✅ **Machine Learning** - scikit-learn, xgboost  
✅ **Time Series** - statsmodels, pmdarima  
✅ **File Support** - openpyxl, xlrd for Excel files  
✅ **Jupyter** - Full notebook support with registered kernel

## 🔧 Activation Commands

### Activate Environment

```bash
conda activate heatwave-analysis
```

### Start Jupyter Notebook

```bash
conda activate heatwave-analysis
jupyter notebook
```

### Start JupyterLab

```bash
conda activate heatwave-analysis
jupyter lab
```

## 📦 Key Packages Installed

### Deep Learning & GPU

- tensorflow[and-cuda]==2.20.0
- keras==3.11.2
- Full CUDA 12.x support

### Data Science Core

- pandas==2.3.1
- numpy==1.26.4
- scipy==1.16.1
- matplotlib==3.10.5
- seaborn==0.13.2

### Machine Learning

- scikit-learn==1.7.1
- xgboost==3.0.4

### Time Series Analysis

- statsmodels==0.14.5
- pmdarima==2.0.4

### File Handling

- openpyxl==3.1.5
- xlrd==2.0.2

### Jupyter Support

- jupyter==1.1.1
- ipykernel==6.30.1
- jupyterlab==4.4.5

## 🧪 GPU Test Results

```python
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
print('GPU available:', tf.test.is_gpu_available())
print('GPU devices:', tf.config.list_physical_devices('GPU'))
```

**Results:**

- TensorFlow version: 2.20.0
- GPU available: True
- GPU devices: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
- GPU: NVIDIA GeForce RTX 2060 with 4785 MB memory

## 📂 Environment Files

- `environment.yml` - Complete conda environment export
- `requirements.txt` - pip freeze output
- This setup supports your full heatwave analysis project!

## 🔄 Recreation Commands

### From environment.yml

```bash
conda env create -f environment.yml
conda activate heatwave-analysis
```

### From requirements.txt (after creating base environment)

```bash
conda create -n heatwave-analysis python=3.11 -y
conda activate heatwave-analysis
pip install -r requirements.txt
```

## 🎯 Next Steps

1. Your environment is ready for data science work
2. TensorFlow GPU is configured and working
3. All your project dependencies are installed
4. Jupyter kernel is registered as "Heatwave Analysis (GPU)"

**Happy analyzing! 🌡️📊**
