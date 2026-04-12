# Han-Nom OCR & Document Understanding Pipeline

This repository contains a full machine learning pipeline for recognizing, processing, and understanding **Han-Nom** (ancient Vietnamese script) texts. It combines object detection (YOLO), Transformer-based OCR (TrOCR), and modern Vision-Language Models (Qwen2.5-VL) to digitize and translate historical documents.

## 🚀 Quick Starts
- If you are looking to get started immediately with our **Qwen2.5-VL-3B** model for Han-Nom Document Understanding, please see the [**Qwen Quickstart Guide**](QUICKSTART.md).

## 🗄️ Dataset
The complete raw and processed Han-Nom dataset used in this project is hosted on Hugging Face:
👉 **[Cong123779/Han_Nom_Dataset](https://huggingface.co/datasets/Cong123779/Han_Nom_Dataset)**

### Where the data comes from
The dataset is primarily composed of:
1. **Nom Foundation (chunom.org)**: A massive source of digitized Luc Bat poetry, historical texts, and dictionaries from the Vietnamese Nom Preservation Foundation.
2. **Tu Dien Han Viet (Thieu Chuu)**: Textual and character structure data processed from standard Han-Viet historical dictionaries.
3. **KanjiVG / NomNaOCR**: Stroke-order and character visualization data utilized for generating massive synthetic text corpora for the models to pre-train on. 

### How to use the Data
To download the dataset locally into your project, you can use the HuggingFace CLI or Python library:
```python
from huggingface_hub import snapshot_download

# This will download the dataset to your local `data/` directory
snapshot_download(repo_id="Cong123779/Han_Nom_Dataset", repo_type="dataset", local_dir="./data")
```

---

## 🏗️ Architecture & Pipeline

The pipeline is designed to handle the complexities of historical texts, including vertical writing layouts, degraded document quality, and complex character sets.

1. **Text Detection & Cropping (YOLOv8 & YOLO11)** 
   Detects characters or columns of text in a source document.
   - `scripts/training/train_yolo.py`: Train custom YOLO detection models.
   - `scripts/inference/batch_yolo_inference.py`: Run batch detection on documents.

2. **Optical Character Recognition (TrOCR)**
   Specialized Transformer models fine-tuned to read cropped Han-Nom images.
   - `scripts/training/finetune_trocr.py`: Fine-tune TrOCR models for Han-Nom.
   - `scripts/inference/predict_trocr.py`: Run OCR inference on character images.

3. **End-to-End Vision-Language Modeling (Qwen2.5-VL)**
   Heavyweight VLM used for reading complex crops, translating, and document understanding.
   - `scripts/training/finetune_qwen.py`: LoRA fine-tuning for Qwen2.5-VL.
   - `scripts/inference/yolo_qwen_pipeline.py`: The complete end-to-end pipeline (YOLO Detection -> Crop -> Qwen2.5-VL OCR).

---

## 🛠️ Data Preparation & Synthetic Generation

Training historical OCR models requires massive amounts of data. This project includes extensive tooling for data harvesting and generation:

- **Synthetic Data Generation**: 
  - `scripts/data_prep/generate_synthetic_nom.py`: Generate synthetic training images using `.ttf` fonts.
  - `scripts/data_prep/generate_images.py`: Image augmentation and generation.
- **Scraping Tools**: 
  - `scripts/data_prep/scrape_nom_foundation.py` / `scripts/data_prep/scrape_kieu_1902.py`: Tools to harvest raw data.
- **Han-Viet Dictionaries**:
  - `scripts/tools/parse_unihan.py`, `scripts/tools/hanviet_utils.py`, `scripts/data_prep/convert_prc_to_csv.py`: Utilities to process structural Han-Viet lookup dictionaries.

---

## 📂 Directory Structure

```text
Han_Nom_Model/
├── assets/                  # Fonts, dictionaries, and test images
├── checkpoints/             # Model training checkpoints
├── data/                    # Project data (raw, processed, metadata)
├── models/                  # Base models and standalone weights
├── output/                  # Final outputs, logs, and adapter weights
└── scripts/                 # Organized Python/Shell scripts
    ├── data_prep/           # Scraping and generation
    ├── training/            # Fine-tuning and training
    ├── inference/           # Prediction and evaluation
    └── tools/               # Utilities and analysis
```

---

## 💻 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/congkx123789/Han_Nom_Model.git
   cd Han_Nom_Model
   ```

2. **Install PyTorch:**
   Ensure you have a GPU-enabled version of PyTorch installed for your system constraints.
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Install Dependencies:**
   Install required libraries for Ultralytics (YOLO), HuggingFace Transformers, PEFT, and Qwen-VL.
   ```bash
   pip install ultralytics transformers accelerate peft qwen-vl-utils pillow tqdm bs4 rapidfuzz
   ```
   *(For 4-bit quantization, you must also install `bitsandbytes`)*

---

## 📝 License
This project is open-source. Datasets and models downloaded via scripts may be subject to their original curators' licenses.
