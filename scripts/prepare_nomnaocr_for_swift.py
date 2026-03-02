import json
import os
from tqdm import tqdm

def convert_to_swift_jsonl(input_file, output_file, image_dir):
    """
    Convert image_path\ttext format to ms-swift JSONL format:
    {"messages": [{"role": "user", "content": "<image>OCR this image"}, {"role": "assistant", "content": "extracted text"}]}
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, desc=f"Converting {os.path.basename(input_file)}"):
            parts = line.strip().split('\t')
            if len(parts) == 2:
                img_rel_path, text = parts
                img_abs_path = os.path.join(image_dir, img_rel_path)
                
                if os.path.exists(img_abs_path):
                    # ms-swift multimodal format for GOT-OCR
                    item = {
                        "messages": [
                            {
                                "role": "user",
                                "content": f"<image>OCR this image:\n"
                            },
                            {
                                "role": "assistant",
                                "content": text
                            }
                        ],
                        "images": [img_abs_path]
                    }
                    f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    # NomNaOCR Patches use relative paths to Patches/Images/
    BASE_DIR = "/home/alida/Documents/Cursor/Han_Nom_Model/data/NomNaOCR_dataset/Patches"
    IMAGE_DIR = os.path.join(BASE_DIR, "Images")
    
    # Train
    convert_to_swift_jsonl(
        os.path.join(BASE_DIR, "Train.txt"),
        "/home/alida/Documents/Cursor/Han_Nom_Model/data/got_swift_dataset/train.jsonl",
        IMAGE_DIR
    )
    
    # Val
    convert_to_swift_jsonl(
        os.path.join(BASE_DIR, "Validate.txt"),
        "/home/alida/Documents/Cursor/Han_Nom_Model/data/got_swift_dataset/val.jsonl",
        IMAGE_DIR
    )
