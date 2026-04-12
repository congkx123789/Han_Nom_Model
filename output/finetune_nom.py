import os
from transformers import AutoModel, AutoTokenizer
from swift.llm import sft_main, SftArguments

# 1. Load base tokenizer and model natively to add tokens
model_id = "stepfun-ai/GOT-OCR2_0"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

# Read the explicitly missing Nôm tokens 
with open("data/missing_nom_chars.txt", "r", encoding="utf-8") as f:
    oov_chars = [line.strip() for line in f if line.strip()]

# Add to tokenizer
num_added = tokenizer.add_tokens(oov_chars)
print(f"Added {num_added} Nôm characters to the vocabulary!")

# Resize model embeddings to match new vocabulary size
model.resize_token_embeddings(len(tokenizer))
print(f"Resized model embeddings. New vocab size: {len(tokenizer)}")

# Save the updated base model and tokenizer locally
local_model_path = "output/got_ocr2_nom_base"
model.save_pretrained(local_model_path)
tokenizer.save_pretrained(local_model_path)
print(f"Saved expanded base model to {local_model_path}")

# Now, we could run sft_main using this local_model_path as the base model
