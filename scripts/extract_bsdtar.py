#!/usr/bin/env python3
"""
Extract ZIP using bsdtar with automated password handling.
"""

import subprocess
import sys
import os

def extract_with_bsdtar(zip_file, output_dir, password=""):
    """Extract ZIP file using bsdtar with password."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Extracting {zip_file} to {output_dir} using bsdtar...")
    print("This may take a while for large files...\n")
    
    # Try extraction with password piped to stdin
    cmd = ['bsdtar', '-xvf', zip_file, '-C', output_dir]
    
    try:
        # Run with password input
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Send password multiple times (for each encrypted file)
        password_input = (password + '\n') * 1000  # Send password 1000 times
        stdout, _ = process.communicate(input=password_input, timeout=3600)  # 1 hour timeout
        
        print(stdout)
        
        if process.returncode == 0:
            print("\n✓ Extraction completed successfully!")
            return True
        else:
            print(f"\n✗ Extraction failed with return code: {process.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        process.kill()
        print("\n✗ Extraction timed out after 1 hour")
        return False
    except Exception as e:
        print(f"\n✗ Error during extraction: {e}")
        return False

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python extract_bsdtar.py <zip_file> <output_dir> [password]")
        sys.exit(1)
    
    zip_file = sys.argv[1]
    output_dir = sys.argv[2]
    password = sys.argv[3] if len(sys.argv) > 3 else ""
    
    if not os.path.exists(zip_file):
        print(f"Error: {zip_file} not found")
        sys.exit(1)
    
    success = extract_with_bsdtar(zip_file, output_dir, password)
    sys.exit(0 if success else 1)
