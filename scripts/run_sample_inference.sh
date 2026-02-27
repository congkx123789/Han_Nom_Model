#!/usr/bin/env bash
# Wrapper to run OCR on 10 sample images using the latest checkpoint
# Using absolute path for checkpoint to avoid Hugging Face Hub validation issues
CHECKPOINT=$(readlink -f ./checkpoints/qwen2.5-vl-han-nom/checkpoint-1000)
IMAGES=(
    "data/raw/images/1321_001.jpg"
    "data/raw/images/1321_002.jpg"
    "data/raw/images/1321_003.jpg"
    "data/raw/images/1321_004.jpg"
    "data/raw/images/1321_005.jpg"
    "data/raw/images/1321_006.jpg"
    "data/raw/images/1321_007.jpg"
    "data/raw/images/1321_008.jpg"
    "data/raw/images/1321_009.jpg"
    "data/raw/images/1321_010.jpg"
)

OUTPUT_FILE="sample_inference_results.txt"
> "$OUTPUT_FILE"

for IMG in "${IMAGES[@]}"; do
    echo "--- $IMG ---" >> "$OUTPUT_FILE"
    # Ensure image path is also absolute or correctly relative
    IMG_PATH=$(readlink -f "$IMG")
    
    python scripts/test_qwen_ocr.py \
        --checkpoint_path "$CHECKPOINT" \
        --image_path "$IMG_PATH" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo "Finished $IMG"
    sleep 1

done

echo "All done. Results saved to $OUTPUT_FILE"
