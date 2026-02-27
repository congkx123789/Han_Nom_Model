#!/usr/bin/env python3
"""
Scraper for Tale of Kieu 1902 version from Nom Foundation
Since the site uses JavaScript for navigation, we'll extract all visible content from the main page
"""

import os
import json
import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import urljoin
import re

BASE_URL = "https://nomfoundation.org"
KIEU_1902_URL = "https://nomfoundation.org/nom-project/tale-of-kieu/tale-of-kieu-version-1902?uiLang=vn"

def fetch_url(url, retries=3):
    """Fetch URL with retry logic"""
    for i in range(retries):
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            if response.status_code == 200:
                return response
            time.sleep(1)
        except Exception as e:
            print(f"  Error fetching {url}: {e}")
            time.sleep(2)
    return None

def download_image(img_url, filepath):
    """Download image to filepath"""
    if os.path.exists(filepath):
        print(f"    Skipping (exists): {os.path.basename(filepath)}")
        return True
    
    response = fetch_url(img_url)
    if response:
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "wb") as f:
                f.write(response.content)
            print(f"    Downloaded: {os.path.basename(filepath)}")
            return True
        except Exception as e:
            print(f"    Failed to save: {e}")
    return False

def extract_page_data(soup, base_url):
    """Extract all text and image data from the page"""
    all_entries = []
    
    # Find all text links that might represent pages
    # These are the clickable text snippets on the page
    text_links = soup.find_all("a", href=lambda h: h and h == base_url)
    
    print(f"Found {len(text_links)} text entries on the page")
    
    for idx, link in enumerate(text_links, start=1):
        text_content = link.get_text(strip=True)
        
        # Skip empty or very short entries
        if not text_content or len(text_content) < 3:
            continue
        
        # Skip navigation links
        if text_content.isdigit() or text_content in ['đã', 'với']:
            continue
            
        entry = {
            "text": text_content,
            "url": f"/data/image_kieu_1902/page_{idx:03d}.jpg",  # Placeholder
            "page_num": idx
        }
        all_entries.append(entry)
    
    return all_entries

def main():
    # Setup directories
    base_dir = "/home/alida/Documents/Cursor/Han_Nom_Model/data/NomNaOCR_dataset"
    output_dir = os.path.join(base_dir, "Raw", "Tale of Kieu 1902")
    pages_dir = os.path.join(base_dir, "Pages", "Tale of Kieu 1902")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(pages_dir, exist_ok=True)
    
    print("=" * 70)
    print("Tale of Kieu 1902 Scraper (Text Extraction)")
    print("=" * 70)
    print(f"\nFetching main page: {KIEU_1902_URL}")
    
    response = fetch_url(KIEU_1902_URL)
    if not response:
        print("ERROR: Failed to fetch main page")
        return
    
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Extract all text entries
    all_entries = extract_page_data(soup, KIEU_1902_URL)
    
    if not all_entries:
        print("\nWARNING: No text entries found. Extracting all visible text...")
        # Fallback: extract all visible text
        body = soup.find("body")
        if body:
            text = body.get_text(separator="\n", strip=True)
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            all_entries = [{"text": line, "url": "", "page_num": i+1} for i, line in enumerate(lines)]
    
    print(f"\nExtracted {len(all_entries)} text entries")
    
    # Save to JSON files (10 entries per file, matching existing structure)
    entries_per_file = 10
    
    for file_idx in range(0, len(all_entries), entries_per_file):
        chunk = all_entries[file_idx:file_idx + entries_per_file]
        json_filename = f"{file_idx // entries_per_file + 1}.json"
        json_path = os.path.join(output_dir, json_filename)
        
        # Format to match existing structure
        formatted_chunk = [{"text": entry["text"], "url": entry["url"]} for entry in chunk]
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(formatted_chunk, f, ensure_ascii=False, indent=2)
        
        print(f"Saved: {json_filename} ({len(chunk)} entries)")
    
    # Also save a complete text file for reference
    text_file = os.path.join(output_dir, "complete_text.txt")
    with open(text_file, "w", encoding="utf-8") as f:
        for entry in all_entries:
            f.write(f"{entry['text']}\n")
    print(f"\nSaved complete text to: complete_text.txt")
    
    print("\n" + "=" * 70)
    print(f"Extraction complete!")
    print(f"Total entries: {len(all_entries)}")
    print(f"JSON files: {(len(all_entries) + entries_per_file - 1) // entries_per_file}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    print("\nNOTE: This extracted text content only. To get images, you may need")
    print("to use a browser automation tool like Selenium or Playwright.")

if __name__ == "__main__":
    main()
