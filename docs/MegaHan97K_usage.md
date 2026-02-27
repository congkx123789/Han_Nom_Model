# MegaHan97K Dataset Usage Guide

## Overview

MegaHan97K is a mega-category, large-scale Chinese character recognition dataset containing **97,455 character categories** - the largest available dataset for Chinese character recognition.

### Dataset Characteristics

- **Total Size**: 15.4GB (7.2GB General CCR + 8.2GB Zero-Shot CCR)
- **Categories**: 97,455 Chinese characters
- **Standard**: Supports GB18030-2022 (latest Chinese standard)
- **Subsets**: Handwritten, Historical, and Synthetic
- **Format**: LMDB (Lightning Memory-Mapped Database)

## Directory Structure

```
data/MegaHan97K/
├── General_CCR/
│   ├── train/
│   │   ├── Handwritten/
│   │   │   ├── Original/          # Original handwritten samples
│   │   │   └── Augmented/         # Augmented handwritten samples
│   │   ├── Historical/
│   │   │   ├── M5HisDoc/          # Historical document samples
│   │   │   └── Kangxi_Dictionary/ # Kangxi dictionary samples
│   │   └── Synthetic/             # Synthetic samples
│   └── test/
│       ├── Handwritten/
│       ├── Handwritten_2/
│       └── Historical/
└── Zero_Shot_CCR/
    └── [similar structure for zero-shot learning]
```

Each LMDB directory contains:
- `data.mdb` - Main database file
- `lock.mdb` - Lock file

## Loading Data

### Using the Dataloader

```python
from scripts.megahan_dataloader import get_megahan_dataloader

# Create dataloader
dataloader = get_megahan_dataloader(
    data_root='data/MegaHan97K/General_CCR/train/Handwritten/Original',
    codebook_path='MegaHan97K/MegaHan_codebook.txt',
    batch_size=32,
    num_workers=4,
    shuffle=True,
    img_size=(96, 96)
)

# Iterate through batches
for images, labels, characters in dataloader:
    # images: torch.Tensor of shape (batch_size, 3, 96, 96)
    # labels: torch.Tensor of shape (batch_size,) - integer labels
    # characters: list of Chinese characters
    print(f"Batch: {images.shape}, Labels: {labels}, Chars: {characters}")
```

### Extract Sample Images

```bash
python3 scripts/megahan_dataloader.py \
    --data_root data/MegaHan97K/General_CCR/train/Handwritten/Original \
    --codebook MegaHan97K/MegaHan_codebook.txt \
    --num_samples 100 \
    --save_dir sample_images
```

## Character Codebook

The codebook file `MegaHan97K/MegaHan_codebook.txt` maps label IDs to Chinese characters:

```
一:U+4E00
二:U+4E8C
三:U+4E09
...
```

Format: `character:unicode_codepoint`

## Integration with Qwen2.5-VL-3B

### For OCR Training

```python
from transformers import Qwen2VLForConditionalGeneration
from scripts.megahan_dataloader import MegaHanLMDBDataset

# Load MegaHan97K dataset
train_dataset = MegaHanLMDBDataset(
    root='data/MegaHan97K/General_CCR/train/Handwritten/Original',
    img_size=(224, 224),  # Qwen2.5-VL input size
    codebook_path='MegaHan97K/MegaHan_codebook.txt'
)

# Fine-tune Qwen2.5-VL-3B for Han Nom OCR
# Combine with NomNaOCR dataset for better Han Nom coverage
```

## Comparison with NomNaOCR

| Feature | MegaHan97K | NomNaOCR |
|---------|------------|----------|
| **Categories** | 97,455 | ~3,000-5,000 |
| **Focus** | Modern Chinese (GB18030-2022) | Han Nom (Vietnamese) |
| **Size** | 15.4GB | ~500MB-1GB |
| **Format** | LMDB | JSON + Images |
| **Use Case** | General Chinese CCR | Historical Vietnamese texts |

### Combined Training Strategy

1. **Pre-train** on MegaHan97K for general Chinese character recognition
2. **Fine-tune** on NomNaOCR for Han Nom specific characters
3. **Benefit**: Leverage large-scale Chinese data + specialized Han Nom knowledge

## Dataset Subsets

### Handwritten
- **Original**: Raw handwritten samples
- **Augmented**: Data-augmented versions (rotation, scaling, etc.)
- **Writers**: Multiple writers (HandWT_1 through HandWT_8)

### Historical
- **M5HisDoc**: Historical document images
- **Kangxi Dictionary**: Characters from Kangxi dictionary

### Synthetic
- Generated using font rendering and style transfer

## Performance Tips

1. **LMDB Caching**: LMDB provides memory-mapped access for fast I/O
2. **Batch Size**: Use larger batches (64-128) for better GPU utilization
3. **Num Workers**: Set to number of CPU cores for parallel data loading
4. **Image Size**: 96x96 for fast training, 224x224 for better accuracy

## References

- **Paper**: "MegaHan97K: A Large-Scale Dataset for Mega-Category Chinese Character Recognition"
- **Repository**: https://github.com/SCUT-DLVCLab/MegaHan97K
- **License**: CC BY-NC-ND 4.0 (Non-commercial use only)
