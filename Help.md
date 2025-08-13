# Heatwave Analysis - Dhaka, Bangladesh Setup Guide

This guide helps you set up a conda environment for the heatwave analysis project.

## Prerequisites

1. Check if conda is installed:

   ```bash
   conda --version
   ```

   If not installed, install Miniconda or Anaconda first.

2. Create conda environment (Tensorflow supports only up to 3.11):

   ```bash
   conda create -n research python=3.11 -y
   ```

3. Activate the environment:

   ```bash
   conda activate research
   ```

4. Install core data science packages:

   ```bash
   conda install -c conda-forge -y pandas numpy matplotlib seaborn scipy scikit-learn statsmodels openpyxl jupyter xgboost
   ```

5. Install TensorFlow GPU support:

   ```bash
   conda install -c conda-forge -y tensorflow-gpu
   ```

6. Install development tools:
   ```bash
   pip install pytest black flake8 isort
   ```

## Verify Installation

Check that key packages are installed:

```bash
python -c "import pandas, numpy, matplotlib, tensorflow; print('All packages imported successfully')"
```

### GPU Support Verification

To ensure TensorFlow can use your GPU for accelerated computation:

1. **Check GPU Detection**:
   ```bash
   python -c "
   import tensorflow as tf
   print('TensorFlow version:', tf.__version__)
   gpus = tf.config.list_physical_devices('GPU')
   print('Number of GPUs detected:', len(gpus))
   if gpus:
       for i, gpu in enumerate(gpus):
           print(f'GPU {i}: {gpu}')
   else:
       print('No GPU devices found')
   "
   ```

2. **Test GPU Functionality**:
   ```bash
   python -c "
   import tensorflow as tf
   print('Built with CUDA:', tf.test.is_built_with_cuda())
   if tf.config.list_physical_devices('GPU'):
       with tf.device('/GPU:0'):
           test_tensor = tf.constant([1.0, 2.0, 3.0])
           print('GPU tensor creation: SUCCESS')
       print('TensorFlow can use GPU!')
   else:
       print('GPU not available - using CPU only')
   "
   ```

3. **Check NVIDIA System Status**:
   ```bash
   nvidia-smi
   ```

#### Expected Output for Working GPU Setup:
- TensorFlow should detect at least 1 GPU
- `Built with CUDA: True`
- `GPU tensor creation: SUCCESS`
- `nvidia-smi` should show your GPU and CUDA version

#### Troubleshooting GPU Issues:
- **No GPU detected**: Ensure NVIDIA drivers are installed and CUDA is compatible
- **CUDA errors**: Check that your CUDA version matches TensorFlow requirements
- **Memory errors**: Monitor GPU memory usage with `nvidia-smi`
- **Driver issues**: Update NVIDIA drivers if needed

#### GPU Memory Management:
```python
# Limit GPU memory growth (recommended for development)
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

## Usage

1. Always activate the environment before working:

   ```bash
   conda activate research
   ```
   ```

## Notes

- **pmdarima excluded**: Conflicts with statsmodels. Install separately if needed after removing statsmodels.
- **GPU Support**: tensorflow-gpu requires CUDA drivers on your system.
- **Environment Removal**: When done with the project:
  ```bash
  conda env remove -n research
  ```
