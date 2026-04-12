import os
import requests
from bs4 import BeautifulSoup
import csv
import time
from urllib.parse import urljoin
import argparse
import sys
import concurrent.futures
import threading

csv_lock = threading.Lock()


BASE_URL = "http://lib.nomfoundation.org"

def fetch_url(url, retries=5):
    """Fetch URL with retry logic for timeouts and errors."""
    for i in range(retries):
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            if response.status_code in [500, 502, 503, 504]:
                print(f"  Server error {response.status_code} for {url}. Retrying ({i+1}/{retries})...")
                time.sleep(2 * (i + 1))
                continue
            return response
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            print(f"  Timeout/Connection error for {url}: {e}. Retrying ({i+1}/{retries})...")
            time.sleep(2 * (i + 1))
        except Exception as e:
             print(f"  Error fetching {url}: {e}. Retrying ({i+1}/{retries})...")
             time.sleep(2 * (i + 1))
    return None

    return None

def download_worker(img_url, filepath, filename, save_dir, writer, row_data):
    """Worker function to download image and log to CSV."""
    if not os.path.exists(filepath):
        img_response = fetch_url(img_url)
        if img_response and img_response.status_code == 200:
            try:
                with open(filepath, "wb") as f:
                    f.write(img_response.content)
                print(f"    Downloaded {filename} to {save_dir}")
            except Exception as e:
                print(f"    Failed to write {filename}: {e}")
        else:
            print(f"    Failed to download image {img_url}")
    pass

    # Write metadata safely
    with csv_lock:
        writer.writerow(row_data)

def setup_directories():

    os.makedirs("data/raw/images", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)

def get_soup(url):
    response = fetch_url(url)
    if response and response.status_code == 200:
        return BeautifulSoup(response.content, "html.parser")
    elif response and response.status_code == 404:
        return None
    else:
        if response:
            print(f"Failed to fetch {url}: {response.status_code}")
        return None


def scrape_volume(volume_url, writer):
    print(f"Scraping volume: {volume_url}")
    
    # Parse collection and volume IDs
    # Expected format: http://lib.nomfoundation.org/collection/{cid}/volume/{vid}
    try:
        parts = volume_url.split("/")
        if "collection" in parts and "volume" in parts:
            col_idx = parts.index("collection")
            collection_id = parts[col_idx + 1]
            vol_idx = parts.index("volume")
            volume_id = parts[vol_idx + 1]
        else:
            # Fallback
            collection_id = "unknown"
            volume_id = volume_url.split("/")[-1]
    except:
        collection_id = "unknown"
        volume_id = volume_url.split("/")[-1]

    vol_soup = None # Initialize to avoid UnboundLocalError


    # Create specific directory
    save_dir = os.path.join("data/raw/images", f"collection_{collection_id}", f"volume_{volume_id}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Try to guess the first page URL or find the 'View contents' link
    first_page_url = volume_url.rstrip("/") + "/page/1"
    
    soup = get_soup(first_page_url)
    if not soup:
        # Try finding the link in the volume page
        vol_soup = get_soup(volume_url)
        if vol_soup:
            link = vol_soup.find("a", string=lambda t: t and "View contents" in t)
            if link:
                first_page_url = urljoin(volume_url, link['href'])
                soup = get_soup(first_page_url)
    
    if not soup:
        print(f"Could not access viewer for {volume_url}")
        return

    # Extract Metadata
    if not vol_soup:
        vol_soup = get_soup(volume_url)

    metadata = {}

    # Title
    h1s = vol_soup.find_all("h1") if vol_soup else []
    if not h1s and soup: h1s = soup.find_all("h1") # Try current page if vol page failed
    if h1s:
        metadata["Title"] = " | ".join([h.get_text(strip=True) for h in h1s])
    
    # DL list
    dl = vol_soup.find("dl", class_="volume") if vol_soup else None
    if not dl and soup: dl = soup.find("dl", class_="volume")
    
    if dl:
        dt_list = dl.find_all("dt")
        dd_list = dl.find_all("dd")
        for dt, dd in zip(dt_list, dd_list):
            key = dt.get_text(strip=True).replace(":", "")
            value = dd.get_text(strip=True)
            if key and value:
                metadata[key] = value
    
    metadata_str = str(metadata)

    # Get list of existing files to speed up checks using scandir
    existing_files = set()
    if os.path.exists(save_dir):
        with os.scandir(save_dir) as it:
            for entry in it:
                if entry.is_file():
                    existing_files.add(entry.name)
    
    def process_page(p_num, p_url, p_soup=None):
        # Predict filename
        f_name = f"{volume_id}_{p_num:03d}.jpg".replace(":", "").replace("?", "")
        
        # For page 1, we ALWAYS want to process it to get total_pages and metadata
        # For other pages, we skip if the file already exists
        if p_num != 1 and f_name in existing_files:
            return True, None
            
        if not p_soup:
            p_soup = get_soup(p_url)
        if not p_soup:
            return False, None
            
        # Find image
        i_url = None
        i_tag = p_soup.find("img", {"ismap": "ismap"})
        if i_tag:
            i_url = urljoin(BASE_URL, i_tag['src'])
        else:
            large_link = p_soup.find("a", href=lambda h: h and "large" in h)
            if large_link:
                i_url = urljoin(BASE_URL, large_link['href'])
        
        if i_url:
            f_path = os.path.join(save_dir, f_name)
            r_data = [volume_url, p_num, i_url, f_path, "", metadata_str]
            download_worker(i_url, f_path, f_name, save_dir, writer, r_data)
            
        # Extract total pages
        t_pages = None
        p_info = p_soup.find(string=lambda t: t and "Page" in t and "of" in t)
        if p_info:
            try:
                parts = p_info.strip().split()
                if "of" in parts:
                    idx = parts.index("of")
                    if idx + 1 < len(parts):
                        t_pages = int(parts[idx+1])
            except:
                pass
        return True, t_pages

    # Process first page (already have soup from line 104 or 112)
    success, total_pages = process_page(1, first_page_url, p_soup=soup)
    if not success:
        print(f"  Failed to process first page of {volume_url}")
        return

    processed_count = 1
    if not total_pages:
        # Fallback to sequential if total pages unknown
        print(f"  Total pages unknown for {volume_url}, falling back to sequential...")
        page_num = 2
        while True:
            base_part = first_page_url.rsplit("/page/", 1)[0]
            next_page_url = f"{base_part}/page/{page_num}"
            success, _ = process_page(page_num, next_page_url)
            if not success: break
            page_num += 1
        processed_count = page_num - 1
    else:
        # Use ThreadPoolExecutor for parallel page metadata fetching (only for missing images)
        # Increased workers to 20 for faster scanning
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as page_executor:
            futures = []
            for p_idx in range(2, total_pages + 1):
                f_name = f"{volume_id}_{p_idx:03d}.jpg".replace(":", "").replace("?", "")
                if f_name in existing_files:
                    continue
                
                print(f"  Processing page {p_idx}...")
                base_part = first_page_url.rsplit("/page/", 1)[0]
                next_page_url = f"{base_part}/page/{p_idx}"
                futures.append(page_executor.submit(process_page, p_idx, next_page_url))
            
            if futures:
                print(f"  Volume {volume_id} has {total_pages} pages. Fetching {len(futures)} missing images...")
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"  Error processing page: {e}")
        processed_count = total_pages


    # Verification
    expected_pages = None
    if "Pages" in metadata:
        try:
            expected_pages = int(metadata["Pages"])
        except:
            pass
    
    if expected_pages is None and total_pages:
        expected_pages = total_pages

    if expected_pages:
        if processed_count >= expected_pages:
            print(f"  [VERIFIED] Volume {volume_id}: Processed {processed_count} pages (Expected: {expected_pages})")
        else:
            print(f"  [WARNING] Volume {volume_id}: Processed {processed_count} pages but expected {expected_pages}")
    else:
        print(f"  [INFO] Volume {volume_id}: Processed {processed_count} pages (No expected count found)")


def crawl(url, writer, limit=None, count=0):
    if limit and count >= limit:
        return count

    if "/volume/" in url and "/page/" not in url:
        scrape_volume(url, writer)
        return count + 1

    print(f"Crawling: {url}")
    soup = get_soup(url)
    if not soup:
        return count

    # Find volumes directly on this page
    # Match "volume" in href, be more lenient
    volume_links = soup.find_all("a", href=lambda h: h and "volume" in h)
    volume_urls = set()
    for link in volume_links:
        # Use current page URL as base for relative links
        full_url = urljoin(url, link['href'])
        if "/volume/" in full_url and "/page/" not in full_url:
            volume_urls.add(full_url)
    
    for vol_url in volume_urls:
        if limit and count >= limit:
            return count
        scrape_volume(vol_url, writer)
        count += 1

    # Find subjects/topics if this is a collection page
    # Match "subject" in href
    subject_links = soup.find_all("a", href=lambda h: h and "subject" in h)
    subject_urls = set()
    for link in subject_links:
        # Use current page URL as base for relative links
        full_url = urljoin(url, link['href'])
        if "/subject/" in full_url:
            subject_urls.add(full_url)
    
    for sub_url in subject_urls:
        if limit and count >= limit:
            return count
        # Avoid infinite recursion if subject links back to collection (unlikely but good practice)
        if sub_url != url:
            count = crawl(sub_url, writer, limit, count)
            
    return count

def main():
    parser = argparse.ArgumentParser(description="Scrape Han-Nom images.")
    parser.add_argument("--url", help="Starting URL (collection or volume)", default="http://lib.nomfoundation.org/collection/2/subject/5")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of volumes to scrape")
    args = parser.parse_args()

    setup_directories()
    
    csv_path = "data/raw/labels.csv"
    file_exists = os.path.exists(csv_path)
    
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    if not file_exists:
        writer.writerow(["volume_url", "page_num", "image_url", "local_path", "text_content", "metadata"])

    
    crawl(args.url, writer, args.limit)
            
    csv_file.close()

if __name__ == "__main__":
    main()
