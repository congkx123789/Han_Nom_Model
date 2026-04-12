#!/usr/bin/env python3
"""
Extract large ZIP files that may have corrupted or missing central directory.
This script reads the local file headers directly instead of relying on the central directory.
"""

import struct
import os
import sys
from pathlib import Path

def extract_zip_without_central_dir(zip_path, output_dir):
    """
    Extract ZIP file by reading local file headers directly.
    Works even if central directory is missing or corrupted.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(zip_path, 'rb') as f:
        file_count = 0
        total_extracted = 0
        
        while True:
            # Read local file header signature
            sig = f.read(4)
            if len(sig) < 4:
                break
                
            # Check for local file header signature (PK\x03\x04)
            if sig != b'PK\x03\x04':
                # Try to find next header
                if sig == b'PK\x01\x02' or sig == b'PK\x05\x06':
                    # Central directory or end marker - we're done with files
                    break
                # Skip this byte and try again
                f.seek(-3, 1)
                continue
            
            # Read local file header (30 bytes after signature, not 26!)
            # ZIP local file header structure:
            # - version needed (2 bytes)
            # - flags (2 bytes)
            # - compression method (2 bytes)
            # - last mod time (2 bytes)
            # - last mod date (2 bytes)
            # - crc-32 (4 bytes)
            # - compressed size (4 bytes)
            # - uncompressed size (4 bytes)
            # - filename length (2 bytes)
            # - extra field length (2 bytes)
            # Total: 26 bytes
            
            header = f.read(26)
            if len(header) < 26:
                break
            
            # Parse: H=2bytes, I=4bytes
            # HHHHHIIHH = 2+2+2+2+2+4+4+2+2 = 22 bytes (WRONG!)
            # Correct: HHHHHIIII = 2+2+2+2+2+4+4+4+4 = 26 bytes (WRONG!)
            # Actually: HHHHHIIHH but need to read 26 bytes
            # Let me recalculate: version(2) + flags(2) + method(2) + time(2) + date(2) + crc(4) + comp(4) + uncomp(4) + name(2) + extra(2) = 26
            # Format should be: HHHHHIIHH = 2+2+2+2+2+4+4+2+2 = 22 bytes
            # We need 4 more bytes! The issue is comp_size and uncomp_size are both 4 bytes (I not H)
            # Correct format: HHHHH II II HH = 2+2+2+2+2 + 4+4+4+4 + 2+2 = 10+16+4 = 30 bytes!
            
            # Let's read 26 bytes and parse what we can
            version = struct.unpack('<H', header[0:2])[0]
            flags = struct.unpack('<H', header[2:4])[0]
            method = struct.unpack('<H', header[4:6])[0]
            mod_time = struct.unpack('<H', header[6:8])[0]
            mod_date = struct.unpack('<H', header[8:10])[0]
            crc32 = struct.unpack('<I', header[10:14])[0]
            comp_size = struct.unpack('<I', header[14:18])[0]
            uncomp_size = struct.unpack('<I', header[18:22])[0]
            name_len = struct.unpack('<H', header[22:24])[0]
            extra_len = struct.unpack('<H', header[24:26])[0]
            
            # Read filename
            filename = f.read(name_len).decode('utf-8', errors='replace')
            
            # Read extra field
            extra = f.read(extra_len)
            
            # Create output path
            out_file = output_path / filename
            
            if filename.endswith('/'):
                # Directory entry
                out_file.mkdir(parents=True, exist_ok=True)
                print(f"Created directory: {filename}")
            else:
                # File entry
                out_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Read compressed data
                data = f.read(comp_size)
                
                if method == 0:
                    # Stored (no compression)
                    with open(out_file, 'wb') as out:
                        out.write(data)
                    total_extracted += len(data)
                    file_count += 1
                    print(f"Extracted ({file_count}): {filename} ({len(data)} bytes)")
                elif method == 8:
                    # Deflate compression
                    import zlib
                    try:
                        uncompressed = zlib.decompress(data, -15)
                        with open(out_file, 'wb') as out:
                            out.write(uncompressed)
                        total_extracted += len(uncompressed)
                        file_count += 1
                        print(f"Extracted ({file_count}): {filename} ({len(uncompressed)} bytes)")
                    except Exception as e:
                        print(f"Error decompressing {filename}: {e}")
                else:
                    print(f"Unsupported compression method {method} for {filename}")
        
        print(f"\n✓ Extraction complete!")
        print(f"  Files extracted: {file_count}")
        print(f"  Total size: {total_extracted / (1024**3):.2f} GB")
        return file_count

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python extract_zip.py <zip_file> <output_directory>")
        sys.exit(1)
    
    zip_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    if not os.path.exists(zip_file):
        print(f"Error: {zip_file} not found")
        sys.exit(1)
    
    print(f"Extracting {zip_file} to {output_dir}...")
    print("This may take a while for large files...\n")
    
    extract_zip_without_central_dir(zip_file, output_dir)
