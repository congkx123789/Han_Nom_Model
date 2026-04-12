# Project Scripts 📜

This directory contains the core logic for the Han-Nom OCR and Document Understanding pipeline. The scripts are organized into subdirectories by their function in the machine learning lifecycle.

## 📂 Subdirectories

### [data_prep/](./data_prep/) 🛠️
Everything needed to harvest and prepare training data.
- **Scrapers**: Tools to pull data from online archives like the Nom Foundation.
- **Generators**: `generate_synthetic_nom.py` creates thousands of training images from `.ttf` fonts.
- **Converters**: Tools to format raw data into YOLO or Transformer-ready datasets.

### [training/](./training/) 🧠
Scripts for model training and fine-tuning.
- **YOLO Training**: Logic for training the detection models.
- **Fine-tuning**: specialized scripts for TrOCR and Qwen2.5-VL (using LoRA).
- **Quantization**: Utilities to compress models to 4-bit for faster inference and lower VRAM usage.

### [inference/](./inference/) 🚀
Production-ready scripts for running models on new documents.
- **Pipelines**: `yolo_qwen_pipeline.py` integrates detection and transcription into a single flow.
- **Batch Processing**: Run OCR on large folders of images at once.

### [tools/](./tools/) 🔧
Utilities and debugging instruments.
- **Vocabulary Analysis**: Analyze character distributions and identify "missing" characters.
- **Debugging**: Tools for visualizing font rendering and image rotations.
- **Comparison**: Compare model performance across different checkpoints.
