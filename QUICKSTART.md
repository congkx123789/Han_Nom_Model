# Qwen2.5-VL-3B Quick Reference

## Model Location
```
~/Documents/Cursor/Han_Nom_Model/models/Qwen2.5-VL-3B/
```

## Quick Start

### Test Model
```bash
cd ~/Documents/Cursor/Han_Nom_Model
python scripts/test_qwen_model.py
```

### Load Model in Python
```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "./models/Qwen2.5-VL-3B",
    torch_dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("./models/Qwen2.5-VL-3B")
```

### Load with 4-bit Quantization (Save VRAM)
```python
from transformers import BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "./models/Qwen2.5-VL-3B",
    quantization_config=quantization_config,
    device_map="auto"
)
```

## Model Specs
- **Size**: 7.1GB (3B parameters)
- **VRAM**: ~7-8GB (full) or ~2.5-3GB (4-bit)
- **Type**: Vision-Language Model
- **Best for**: OCR, Document Understanding, Image-to-Text

## Directory Structure
```
Han_Nom_Model/
├── models/Qwen2.5-VL-3B/      # Model files
├── output/qwen3b_hannom_lora/ # Training output
├── data/                       # Han-Nom dataset
└── scripts/test_qwen_model.py # Verification
```

## Next Steps
1. Run `python scripts/test_qwen_model.py` to verify
2. Prepare training data format
3. Configure LoRA for fine-tuning
4. Start training on Han-Nom dataset

## Resources
- Model README: `models/README.md`
- Full Walkthrough: See artifacts
- Official Docs: https://qwen.readthedocs.io/
