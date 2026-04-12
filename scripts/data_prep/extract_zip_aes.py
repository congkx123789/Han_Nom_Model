#!/usr/bin/env python3
"""
Extract large ZIP files with AES encryption support using pyzipper.
"""

import sys
import os
from pathlib import Path

try:
    import pyzipper
except ImportError:
    print("Error: pyzipper not installed. Install with: pip install pyzipper")
    sys.exit(1)

def extract_zip_with_pyzipper(zip_path, output_dir, password=None):
    """
    Extract ZIP file using pyzipper which supports AES encryption.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Opening {zip_path}...")
    
    try:
        with pyzipper.AESZipFile(zip_path, 'r') as zf:
            if password:
                zf.setpassword(password.encode())
            
            file_list = zf.namelist()
            total_files = len(file_list)
            print(f"Found {total_files} files/directories in archive")
            print(f"Extracting to {output_dir}...\n")
            
            for i, filename in enumerate(file_list, 1):
                try:
                    zf.extract(filename, output_dir)
                    if i % 10 == 0 or i == total_files:
                        print(f"Progress: {i}/{total_files} ({100*i//total_files}%) - {filename}")
                except Exception as e:
                    print(f"Error extracting {filename}: {e}")
            
            print(f"\n✓ Extraction complete!")
            print(f"  Total files: {total_files}")
            return total_files
            
    except Exception as e:
        print(f"Error opening ZIP file: {e}")
        print("\nIf the file is password protected, run with:")
        print(f"  python {sys.argv[0]} <zip_file> <output_dir> <password>")
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python extract_zip_aes.py <zip_file> <output_directory> [password]")
        sys.exit(1)
    
    zip_file = sys.argv[1]
    output_dir = sys.argv[2]
    password = sys.argv[3] if len(sys.argv) > 3 else None
    
    if not os.path.exists(zip_file):
        print(f"Error: {zip_file} not found")
        sys.exit(1)
    
    extract_zip_with_pyzipper(zip_file, output_dir, password)
