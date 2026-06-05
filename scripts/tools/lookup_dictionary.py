#!/usr/bin/env python3
import os
import sys
import pandas as pd
import re

# File Paths
DATA_DIR = "/home/alida/Documents/Cursor/Han_Nom_Model/data"
CVDICT_PATH = os.path.join(DATA_DIR, "CVDICT_Trung_Viet.csv")
THIEUCHUU_PATH = os.path.join(DATA_DIR, "Thieu_Chuu_Dictionary.csv")
UNIHAN_PATH = os.path.join(DATA_DIR, "Unihan_Vietnamese.csv")

# Terminal Color Codes
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
WHITE = "\033[97m"
BOLD = "\033[1m"
RESET = "\033[0m"

class HanNomDictionary:
    def __init__(self):
        print(f"{CYAN}⏳ Đang tải cơ sở dữ liệu từ điển...{RESET}", end="", flush=True)
        t0 = pd.Timestamp.now()
        
        self.df_cvdict = None
        self.df_thieuchuu = None
        self.df_unihan = None

        if os.path.exists(CVDICT_PATH):
            self.df_cvdict = pd.read_csv(CVDICT_PATH)
        if os.path.exists(THIEUCHUU_PATH):
            self.df_thieuchuu = pd.read_csv(THIEUCHUU_PATH)
        if os.path.exists(UNIHAN_PATH):
            self.df_unihan = pd.read_csv(UNIHAN_PATH)

        duration = (pd.Timestamp.now() - t0).total_seconds()
        print(f"\r{GREEN}✅ Đã tải xong từ điển trong {duration:.2f} giây!{RESET}")
        
        # Print summary
        cvdict_len = len(self.df_cvdict) if self.df_cvdict is not None else 0
        thieuchuu_len = len(self.df_thieuchuu) if self.df_thieuchuu is not None else 0
        unihan_len = len(self.df_unihan) if self.df_unihan is not None else 0
        
        print(f"   • {BOLD}CVDICT (Trung-Việt):{RESET} {cvdict_len:,} mục từ")
        print(f"   • {BOLD}Thiều Chửu (Hán-Việt):{RESET} {thieuchuu_len:,} mục từ")
        print(f"   • {BOLD}Unihan (Âm Hán-Việt):{RESET} {unihan_len:,} chữ")
        print("-" * 60)

    def is_chinese_char(self, char):
        # Checks if char is a Chinese character (CJK Unified Ideographs)
        return any('\u4e00' <= c <= '\u9fff' or '\u3400' <= c <= '\u4dbf' or '\U00020000' <= c <= '\U0002a6df' for c in char)

    def search(self, query):
        query = query.strip()
        if not query:
            return

        is_cjk = self.is_chinese_char(query)
        print(f"\n{BOLD}{YELLOW}🔎 Kết quả tìm kiếm cho: \"{query}\"{RESET}")
        print("=" * 60)

        found_any = False

        # --- 1. SEARCH THIEU CHUU DICTIONARY ---
        if self.df_thieuchuu is not None:
            tc_results = []
            if is_cjk and len(query) == 1:
                tc_results = self.df_thieuchuu[self.df_thieuchuu['char'] == query]
            elif not is_cjk:
                # Search inside definitions or pronunciations
                tc_results = self.df_thieuchuu[
                    self.df_thieuchuu['definition'].str.contains(query, case=False, na=False) |
                    self.df_thieuchuu['pronunciation'].str.contains(query, case=False, na=False)
                ].head(10) # limit to top 10 for readability

            if len(tc_results) > 0:
                found_any = True
                print(f"\n{BOLD}{RED}📖 TỪ ĐIỂN THIỀU CHỬU HÁN-VIỆT ({len(tc_results)} kết quả):{RESET}")
                for idx, row in tc_results.iterrows():
                    print(f"  • {BOLD}{GREEN}{row['char']}{RESET} [{row['pronunciation'].upper()}]")
                    # Clean definitions spacing
                    definition = str(row['definition']).replace(" . ", ". ").replace(" , ", ", ")
                    print(f"    {WHITE}{definition}{RESET}")
                print("-" * 40)

        # --- 2. SEARCH UNIHAN VIETNAMESE ---
        if self.df_unihan is not None:
            unihan_results = []
            if is_cjk and len(query) == 1:
                unihan_results = self.df_unihan[self.df_unihan['char'] == query]
            elif not is_cjk:
                unihan_results = self.df_unihan[
                    self.df_unihan['kVietnamese'].str.contains(query, case=False, na=False) |
                    self.df_unihan['kDefinition'].str.contains(query, case=False, na=False)
                ].head(10)

            if len(unihan_results) > 0:
                found_any = True
                print(f"\n{BOLD}{BLUE}🌐 DỮ LIỆU UNIHAN VIỆT NAM / QUỐC TẾ ({len(unihan_results)} kết quả):{RESET}")
                for idx, row in unihan_results.iterrows():
                    viet_pron = row['kVietnamese'] if pd.notna(row['kVietnamese']) else "N/A"
                    eng_def = row['kDefinition'] if pd.notna(row['kDefinition']) else "N/A"
                    print(f"  • {BOLD}{GREEN}{row['char']}{RESET} (Unicode: {row['hex']})")
                    print(f"    - Phát âm Việt: {CYAN}{viet_pron}{RESET}")
                    print(f"    - Định nghĩa Anh: {WHITE}{eng_def}{RESET}")
                print("-" * 40)

        # --- 3. SEARCH CVDICT (TRUNG-VIỆT) ---
        if self.df_cvdict is not None:
            cv_results = []
            if is_cjk:
                # Search by exact match of characters
                cv_results = self.df_cvdict[
                    (self.df_cvdict['Phồn_thể'] == query) | 
                    (self.df_cvdict['Giản_thể'] == query)
                ]
                # If no exact match, search substring
                if len(cv_results) == 0:
                    cv_results = self.df_cvdict[
                        self.df_cvdict['Phồn_thể'].str.contains(query, na=False) | 
                        self.df_cvdict['Giản_thể'].str.contains(query, na=False)
                    ].head(10)
            else:
                # Search in meaning, pinyin or pronunciation
                cv_results = self.df_cvdict[
                    self.df_cvdict['Nghĩa_Việt'].str.contains(query, case=False, na=False) |
                    self.df_cvdict['Pinyin'].str.contains(query, case=False, na=False)
                ].head(10)

            if len(cv_results) > 0:
                found_any = True
                print(f"\n{BOLD}{MAGENTA}📚 TỪ ĐIỂN TRUNG-VIỆT (CVDICT) ({len(cv_results)} kết quả):{RESET}")
                for idx, row in cv_results.iterrows():
                    trad = row['Phồn_thể']
                    simp = row['Giản_thể']
                    char_display = f"{trad}" if trad == simp else f"{trad} ({simp})"
                    print(f"  • {BOLD}{GREEN}{char_display}{RESET} | Pinyin: {CYAN}{row['Pinyin']}{RESET}")
                    print(f"    - Nghĩa: {WHITE}{row['Nghĩa_Việt']}{RESET}")
                print("-" * 40)

        if not found_any:
            print(f"\n{RED}❌ Không tìm thấy kết quả nào trùng khớp cho từ khóa \"{query}\".{RESET}")
        print("=" * 60)

def main():
    dict_helper = HanNomDictionary()
    
    # If arguments are provided, query directly
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        dict_helper.search(query)
    else:
        # Interactive mode
        print(f"\n{BOLD}{CYAN}🏮 HỆ THỐNG TRA CỨU TỪ ĐIỂN HÁN NÔM ĐA NĂNG 🏮{RESET}")
        print("Nhập từ khóa cần tra cứu (chữ Hán, Nôm, Pinyin, hoặc nghĩa Tiếng Việt).")
        print(f"Nhập '{BOLD}exit{RESET}' hoặc '{BOLD}quit{RESET}' để thoát.")
        print("=" * 60)
        
        try:
            while True:
                query = input(f"\n{BOLD}Từ điển Hán Nôm > {RESET}").strip()
                if query.lower() in ['exit', 'quit']:
                    print(f"{GREEN}Cảm ơn bạn đã sử dụng hệ thống! Tạm biệt.{RESET}")
                    break
                if not query:
                    continue
                dict_helper.search(query)
        except (KeyboardInterrupt, EOFError):
            print(f"\n{GREEN}Tạm biệt.{RESET}")

if __name__ == "__main__":
    main()
