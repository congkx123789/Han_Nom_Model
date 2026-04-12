import re
import subprocess
import os

def scrape_from_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Regex to find markdown links: [text](url)
    links = re.findall(r'\[.*?\]\((http://lib\.nomfoundation\.org/.*?)\)', content)
    
    unique_links = sorted(list(set(links)))
    
    print(f"Found {len(unique_links)} unique Nom Foundation links in {filepath}:")
    for link in unique_links:
        print(f" - {link}")

    for link in unique_links:
        print(f"\nStarting scrape for: {link}")
        # Call the existing scraper script
        # We use subprocess to run it as a separate process
        try:
            subprocess.run(["python3", "scripts/scrape_nom_foundation.py", "--url", link], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error scraping {link}: {e}")

if __name__ == "__main__":
    target_file = "scripts/Untitled-1"
    if os.path.exists(target_file):
        scrape_from_file(target_file)
    else:
        print(f"File not found: {target_file}")
