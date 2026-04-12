# Project Assets 🎨

This directory contains static resources used for character rendering, data lookup, and testing.

## 📂 Subdirectories

### [fonts/](./fonts/) 🖋️
Contains TrueType fonts used to render Han-Nom characters. 
- **NomNaTong-Regular.ttf**: The primary font used for generating synthetic training images.

### [dictionary/](./dictionary/) 📖
Project-specific lookup resources.
- **Han-Viet Tu-dien - Thieu Chuu.prc**: A comprehensive Sino-Vietnamese dictionary used for cross-referencing glyphs.

### [tests/](./tests/) 🖼️
A collection of sample images used to quickly test model performance.
- Use these with `scripts/inference/predict_trocr.py` or the full `yolo_qwen_pipeline.py`.
