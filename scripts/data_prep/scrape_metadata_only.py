import os
import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import time
import argparse
from tqdm import tqdm

def fetch_url(url, retries=5):
    """Fetch URL with retry logic."""
    for i in range(retries):
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            if response.status_code == 200:
                return response
            elif response.status_code in [500, 502, 503, 504]:
                time.sleep(2 * (i + 1))
                continue
            else:
                return response
        except Exception:
            time.sleep(2 * (i + 1))
    return None

def scrape_volume_metadata(url):
    """Scrape comprehensive metadata from a volume page."""
    response = fetch_url(url)
    if not response or response.status_code != 200:
        return None
    
    soup = BeautifulSoup(response.content, "html.parser")
    metadata = {}
    
    # Extract Titles and Headers
    h1s = [h.get_text(strip=True) for h in soup.find_all("h1")]
    h2s = [h.get_text(strip=True) for h in soup.find_all("h2")]
    h3s = [h.get_text(strip=True) for h in soup.find_all("h3")]
    
    if h1s: metadata["H1s"] = h1s
    if h2s: metadata["H2s"] = h2s
    if h3s: metadata["H3s"] = h3s
    
    # Preferred combined title for easier mapping
    metadata["CombinedTitle"] = " | ".join(h1s + h2s)

    # Extract DL list (structured metadata)
    dl = soup.find("dl", class_="volume")
    if dl:
        dt_list = dl.find_all("dt")
        dd_list = dl.find_all("dd")
        for dt, dd in zip(dt_list, dd_list):
            key = dt.get_text(strip=True).replace(":", "").strip()
            value = dd.get_text(strip=True).strip()
            if key and value:
                metadata[key] = value
                
    # Capture the entire textual content of the metadata div (or body) to be sure
    # Finding the main content area
    content_div = soup.find("div", id="content") or soup.find("div", class_="main") or soup.body
    if content_div:
        # Get text while preserving some structure
        metadata["FullText"] = content_div.get_text(separator="\n", strip=True)
    
    return metadata

def main():
    parser = argparse.ArgumentParser(description="Scrape metadata for Han-Nom volumes.")
    parser.add_argument("--url", help="Single volume URL to scrape")
    parser.add_argument("--csv", default="data/raw/labels.csv", help="Path to labels.csv to extract unique volumes")
    parser.add_argument("--output", default="data/raw/volume_metadata.json", help="Output JSON file")
    parser.add_argument("--limit", type=int, help="Limit number of volumes to scrape")
    args = parser.parse_args()

    if args.url:
        print(f"Scraping metadata for: {args.url}")
        meta = scrape_volume_metadata(args.url)
        print(json.dumps(meta, indent=2, ensure_ascii=False))
        return

    if not os.path.exists(args.csv):
        print(f"Error: {args.csv} not found.")
        return

    import csv
    unique_urls = set()
    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        try:
            vol_url_idx = header.index("volume_url")
        except ValueError:
            print("Error: 'volume_url' column not found in CSV.")
            return
            
        for row in reader:
            if row:
                unique_urls.add(row[vol_url_idx])
    
    unique_urls = list(unique_urls)
    if args.limit:
        unique_urls = unique_urls[:args.limit]
        
    print(f"Found {len(unique_urls)} unique volumes.")
    
    results = {}
    
    # Load existing results if any
    if os.path.exists(args.output):
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"Loaded {len(results)} existing entries from {args.output}")
        except Exception:
            pass

    for url in tqdm(unique_urls, desc="Scraping"):
        # Normalize URL (trailing slash)
        n_url = url.rstrip("/") + "/"
        if n_url in results and results[n_url]:
            continue
            
        meta = scrape_volume_metadata(n_url)
        if meta:
            results[n_url] = meta
            # Save every 10 iterations to prevent data loss
            if len(results) % 10 == 0:
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
        
        time.sleep(1) # Be polite to the server

    # Final save
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Done. Saved metadata for {len(results)} volumes to {args.output}")

if __name__ == "__main__":
    main()
