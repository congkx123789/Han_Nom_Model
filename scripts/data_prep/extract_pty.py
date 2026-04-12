import pty
import os
import subprocess
import sys

def extract_with_password(zip_file, output_dir, password):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cmd = ["bsdtar", "-xvf", zip_file, "-C", output_dir]
    
    def read(fd):
        data = os.read(fd, 1024)
        if b"Enter passphrase:" in data:
            os.write(fd, (password + "\n").encode())
        return data

    print(f"Extracting {zip_file} with password...")
    pty.spawn(cmd, read)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 extract_pty.py <zip_file> <output_dir> <password>")
        sys.exit(1)
    
    extract_with_password(sys.argv[1], sys.argv[2], sys.argv[3])
