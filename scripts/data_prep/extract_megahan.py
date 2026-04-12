#!/usr/bin/env python3
"""
Extract ZIP files with AES encryption support using pyzipper.
Handles large files and provides progress tracking.
"""

import sys
import os
from pathlib import Path

try:
    import pyzipper
except ImportError:
    print("Error: pyzipper not installed.")
    print("Installing pyzipper...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyzipper"])
    import pyzipper

def extract_zip_with_progress(zip_path, output_dir, password=None):
    """
    Extract ZIP file using pyzipper with progress tracking.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Opening {zip_path}...")
    print(f"Output directory: {output_dir}\n")
    
    try:
        with pyzipper.AESZipFile(zip_path, 'r', compression=pyzipper.ZIP_STORED, allowZip64=True) as zf:
            if password:
                zf.setpassword(password.encode())
            
            file_list = zf.namelist()
            total_files = len(file_list)
            print(f"Found {total_files} files/directories in archive")
            print("Starting extraction...\n")
            
            extracted_count = 0
            total_size = 0
            
            for i, filename in enumerate(file_list, 1):
                try:
                    # Extract file
                    zf.extract(filename, output_dir)
                    
                    # Get file info
                    info = zf.getinfo(filename)
                    file_size = info.file_size
                    total_size += file_size
                    
                    if not filename.endswith('/'):
                        extracted_count += 1
                    
                    # Print progress every 10 files or for last file
                    if i % 10 == 0 or i == total_files:
                        progress_pct = 100 * i // total_files
                        size_gb = total_size / (1024**3)
                        print(f"Progress: {i}/{total_files} ({progress_pct}%) - {extracted_count} files - {size_gb:.2f} GB")
                        print(f"  Current: {filename}")
                        
                except Exception as e:
                    print(f"Error extracting {filename}: {e}")
                    continue
            
            print(f"\n✓ Extraction complete!")
            print(f"  Total files extracted: {extracted_count}")
            print(f"  Total size: {total_size / (1024**3):.2f} GB")
            print(f"  Output directory: {output_dir}")
            return extracted_count
            
    except Exception as e:
        print(f"Error opening ZIP file: {e}")
        print(f"\nTrying alternative method...")
        
        # Try with standard ZipFile as fallback
        try:
            import zipfile
            with zipfile.ZipFile(zip_path, 'r', allowZip64=True) as zf:
                file_list = zf.namelist()
                total_files = len(file_list)
                print(f"Found {total_files} files using standard zipfile")
                
                for i, filename in enumerate(file_list, 1):
                    try:
                        zf.extract(filename, output_dir)
                        if i % 10 == 0 or i == total_files:
                            print(f"Progress: {i}/{total_files} ({100*i//total_files}%)")
                    except Exception as e2:
                        print(f"Error: {e2}")
                        continue
                        
                print(f"\n✓ Extraction complete using fallback method!")
                return total_files
        except Exception as e2:
            print(f"Fallback method also failed: {e2}")
            sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python extract_megahan.py <zip_file> <output_directory> [password]")
        sys.exit(1)
    
    zip_file = sys.argv[1]
    output_dir = sys.argv[2]
    password = sys.argv[3] if len(sys.argv) > 3 else None
    
    if not os.path.exists(zip_file):
        print(f"Error: {zip_file} not found")
        sys.exit(1)
    
    print("="*60)
    print("MegaHan97K Dataset Extractor")
    print("="*60)
    
    extract_zip_with_progress(zip_file, output_dir, password)
