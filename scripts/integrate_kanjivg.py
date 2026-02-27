import requests
import zipfile
import io
import os

def download_kanjivg(url, output_dir):
    print(f"Downloading KanjiVG from {url}...")
    r = requests.get(url)
    if r.status_code == 200:
        z = zipfile.ZipFile(io.BytesIO(r.content))
        print(f"Extracting to {output_dir}...")
        z.extractall(output_dir)
        print("Extraction complete.")
    else:
        print(f"Failed to download KanjiVG. Status code: {r.status_code}")

def main():
    # Latest release URL as of my search
    url = "https://github.com/KanjiVG/kanjivg/releases/download/r20250816/kanjivg-20250816-main.zip"
    output_dir = "/home/alida/Documents/Cursor/Han_Nom_Model/data/KanjiVG"
    
    os.makedirs(output_dir, exist_ok=True)
    download_kanjivg(url, output_dir)

if __name__ == "__main__":
    main()
