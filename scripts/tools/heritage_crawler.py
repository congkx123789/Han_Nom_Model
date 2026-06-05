import requests
from bs4 import BeautifulSoup
import time
import json
import os
import concurrent.futures
import sqlite3
from typing import List, Dict
import logging
from pymilvus import MilvusClient
from langchain_ollama import OllamaEmbeddings

# Cấu hình Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("HeritageTotalWar")

class HeritageDestroyer:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        self.ollama_url = "http://ollama:11434/api/generate" 
        self.db_path = "/app/data/total_war_state.db"
        self._init_state_db()
        self.temp_results = []
        
        try:
            self.milvus = MilvusClient(uri="http://milvus_standalone:19530")
            self.embedder = OllamaEmbeddings(model="nomic-embed-text", base_url="http://ollama:11434")
            logger.info("✅ Đã kết nối Milvus & Embedder.")
        except Exception as e:
            logger.error(f"❌ Lỗi kết nối Milvus: {e}")
            self.milvus = None

    def _init_state_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("CREATE TABLE IF NOT EXISTS crawled_urls (url TEXT PRIMARY KEY, title TEXT)")
        conn.close()

    def is_crawled(self, url):
        conn = sqlite3.connect(self.db_path)
        res = conn.execute("SELECT 1 FROM crawled_urls WHERE url=?", (url,)).fetchone()
        conn.close()
        return res is not None

    def mark_crawled(self, url, title):
        conn = sqlite3.connect(self.db_path)
        conn.execute("INSERT OR REPLACE INTO crawled_urls (url, title) VALUES (?, ?)", (url, title))
        conn.commit()
        conn.close()

    def crawl_cohoc_total(self, max_pages=875):
        """Quét sạch 875 trang của Cổ Học (tuvi.cohoc.net)"""
        logger.info(f"🔥 BẮT ĐẦU QUÉT SẠCH COHOC.NET (875 TRANG)...")
        for p in range(1, max_pages + 1):
            url = f"https://tuvi.cohoc.net/hoc-tu-vi-{p}.html"
            try:
                resp = requests.get(url, headers=self.headers, timeout=15)
                soup = BeautifulSoup(resp.content, 'html.parser')
                links = soup.select('a[href*="-nid-"]')
                if not links: break
                
                for l in links:
                    full_url = "https://tuvi.cohoc.net/" + l['href'] if not l['href'].startswith('http') else l['href']
                    if not self.is_crawled(full_url):
                        self.process_article(full_url, "Cổ Học")
                
                logger.info(f"✅ [Cổ Học] Đã xong trang {p}/{max_pages}. Tổng mục: {len(self.results)}")
            except: continue

    def crawl_hannom_rcv_wiki_total(self):
        """Quét toàn bộ Special:AllPages của SRAHN Wiki"""
        logger.info("🔥 BẮT ĐẦU QUÉT SẠCH HANNOM-RCV.ORG WIKI...")
        url = "https://www.hannom-rcv.org/wi/index.php?title=Special:AllPages&uiLang=vi"
        while url:
            try:
                resp = requests.get(url, headers=self.headers, timeout=20)
                soup = BeautifulSoup(resp.content, 'html.parser')
                links = soup.select('ul.mw-allpages-chunk a')
                for l in links:
                    full_url = "https://www.hannom-rcv.org" + l['href']
                    if not self.is_crawled(full_url):
                        self.process_article(full_url, "SRAHN Wiki")
                
                next_link = soup.select_one('a.mw-allpages-nav-next')
                url = "https://www.hannom-rcv.org" + next_link['href'] if next_link else None
                logger.info(f"📄 Chuyển sang trang Wiki tiếp theo...")
            except: break

    def crawl_vass_total(self, start=1, end=70000):
        """Quét 70,000 đầu mục Viện Hán Nôm"""
        logger.info(f"🔥 BẮT ĐẦU QUÉT SẠCH VIỆN HÁN NÔM ({start} -> {end})...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            for i in range(start, end):
                url = f"https://hannom.vass.gov.vn/?pageid=46955&sachid={i}"
                if not self.is_crawled(url):
                    executor.submit(self.process_article, url, "Viện Hán Nôm")

    def process_article(self, url, source):
        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(resp.content, 'html.parser')
            
            title = "N/A"
            if "cohoc.net" in url:
                title = soup.find('h1').text.strip() if soup.find('h1') else "Bài viết"
            elif "vass" in url:
                b = soup.select_one('#lnkCatName')
                title = b.next_sibling.strip() if b and b.next_sibling else "Di sản"
            elif "chunom.org" in url:
                title = soup.select_one('div.shelf-title').text.strip() if soup.select_one('div.shelf-title') else "Tác phẩm Nôm"
            else:
                title = soup.find('h1').text.strip() if soup.find('h1') else "Wiki Page"

            if "chunom.org" in url:
                content = soup.select_one('div.work-content').get_text(separator='\n').strip() if soup.select_one('div.work-content') else ""
            else:
                content = soup.get_text(separator=' ').strip()
            
            # CHẾ ĐỘ SIÊU TỐC: Chỉ cào và lưu file JSON, không tốn GPU cho Embedding/AI
            self.mark_crawled(url, title)
            self.temp_results.append({
                "title": title, 
                "source": source, 
                "url": url,
                "content": content
            })
            
            if len(self.temp_results) >= 50: # Tăng lô lưu file lên 50 để nhanh hơn
                self.save_temp_file()
        except: pass

    def save_temp_file(self):
        path = "/app/data/total_war_harvest.json"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        existing = []
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                try: existing = json.load(f)
                except: pass
        existing.extend(self.temp_results)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(existing, f, ensure_ascii=False, indent=4)
        self.temp_results = []
        logger.info(f"💾 Đã lưu đợt dữ liệu mới vào {path}")

    def crawl_chunom_org(self, max_works=2000): # Tăng lên 2000 mục
        """Cào từ chunom.org - Dữ liệu Unicode cực quý giá"""
        logger.info(f"🚀 Bắt đầu cào CHUNOM.ORG (Mở rộng 2000 tác phẩm)...")
        for i in range(1, max_works + 1):
            url = f"https://chunom.org/shelf/work/{i}/"
            if not self.is_crawled(url):
                self.process_article(url, "Chữ Nôm Unicode")
                logger.info(f"✅ [Chữ Nôm] Đã hốt mục ID: {i}")
                time.sleep(0.5)

    def crawl_vanban_hannom_com(self):
        """Cào từ vanbanhan-nom.com - Văn bản hành chính & Sắc phong"""
        logger.info(f"🚀 Bắt đầu cào VANBANHAN-NOM.COM (Tư liệu hành chính cổ)...")
        categories = ['sach-han-nom', 'van-ban-han-nom', 'tu-lieu-han-nom']
        for cat in categories:
            for p in range(1, 50): # Quét 50 trang mỗi chuyên mục
                url = f"http://vanbanhan-nom.com/chuyen-muc/{cat}/page/{p}"
                try:
                    resp = requests.get(url, headers=self.headers, timeout=15)
                    soup = BeautifulSoup(resp.content, 'html.parser')
                    links = soup.select('h2.entry-title a')
                    if not links: break
                    for l in links:
                        full_url = l['href']
                        if not self.is_crawled(full_url):
                            self.process_article(full_url, "Văn Bản Hán Nôm")
                    logger.info(f"✅ [Văn Bản HN] Xong trang {p} mục {cat}")
                except: break

    def start_total_war(self):
        self.crawl_vanban_hannom_com() # Mục tiêu mới 1
        self.crawl_chunom_org() # Mục tiêu mở rộng
        self.crawl_cohoc_total()
        self.crawl_hannom_rcv_wiki_total()
        self.crawl_vass_total(start=1, end=75000)
        logger.info("🚩 CHIẾN DỊCH QUÉT SẠCH HOÀN TẤT!")

if __name__ == "__main__":
    destroyer = HeritageDestroyer()
    destroyer.start_total_war()
