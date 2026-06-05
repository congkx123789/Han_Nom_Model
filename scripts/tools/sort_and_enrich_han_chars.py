#!/usr/bin/env python3
"""
sort_and_enrich_han_chars.py
============================
Làm sạch và tái cấu trúc file han_characters_only.csv:
  1. Tải bảng tần suất chữ Hán (Junda corpus - chuẩn thực tế nhất)
  2. Sắp xếp từ phổ biến → hiếm (chữ không có trong bảng → cuối)
  3. Làm sạch cột Phiên_âm_Hán_Việt (bỏ ['...'] rác)
  4. Thêm cột Tần_suất_rank và Số_âm_HV
"""

import os
import re
import csv
import urllib.request

DATA_DIR = "/home/alida/Documents/Cursor/Han_Nom_Model/data"
INPUT_CSV  = os.path.join(DATA_DIR, "han_characters_only.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "han_characters_only.csv")   # overwrite in-place
FREQ_CACHE = os.path.join(DATA_DIR, "raw", "junda_char_freq.txt")

# ---------------------------------------------------------------------------
# 1. TẢI BẢNG TẦN SUẤT JUNDA
# ---------------------------------------------------------------------------
JUNDA_URL = "https://raw.githubusercontent.com/fxsjy/jieba/master/extra_dict/dict.txt.small"
# Dự phòng – dùng unicode block order nếu không tải được

def load_junda_freq() -> dict[str, int]:
    """
    Tải Junda character frequency list.
    File format (jieba small dict):  word  freq  pos
    Lọc ra các entry có len(word)==1 để lấy tần suất đơn ký tự.
    Returns dict {char: rank} (rank 1 = phổ biến nhất).
    """
    os.makedirs(os.path.join(DATA_DIR, "raw"), exist_ok=True)

    if not os.path.exists(FREQ_CACHE):
        print(f"⬇️  Đang tải bảng tần suất từ Jieba corpus...")
        try:
            urllib.request.urlretrieve(JUNDA_URL, FREQ_CACHE)
            print(f"   ✅ Đã lưu tại {FREQ_CACHE}")
        except Exception as e:
            print(f"   ⚠️  Không tải được ({e}). Dùng thứ tự Unicode.")
            return {}
    else:
        print(f"   📂 Dùng cache: {FREQ_CACHE}")

    freq_map: dict[str, int] = {}
    with open(FREQ_CACHE, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                word = parts[0]
                try:
                    freq = int(parts[1])
                except ValueError:
                    continue
                if len(word) == 1:
                    char = word
                    if char not in freq_map or freq > freq_map[char]:
                        freq_map[char] = freq

    # Chuyển sang rank (1 = phổ biến nhất)
    sorted_chars = sorted(freq_map.items(), key=lambda x: -x[1])
    rank_map = {char: rank + 1 for rank, (char, _) in enumerate(sorted_chars)}
    print(f"   📊 Bảng tần suất: {len(rank_map):,} ký tự đơn")
    return rank_map


# ---------------------------------------------------------------------------
# 2. LÀM SẠCH PHIÊN ÂM HÁN VIỆT
# ---------------------------------------------------------------------------

def clean_hanviet(raw: str) -> str:
    """
    Trích xuất các âm Hán-Việt sạch từ chuỗi hỗn hợp như:
      "tuấn / ['tuấn']"          → "tuấn"
      "phỏng / ['phỏng' / 'phảng']"  → "phỏng / phảng"
      "['đồng'] / đông"          → "đồng / đông"
      ""                          → ""
    """
    if not raw or raw.strip() in ("", "nan", "N/A"):
        return ""

    # Tách theo dấu /
    parts = [p.strip() for p in raw.split("/")]
    cleaned = []
    seen = set()
    for p in parts:
        # Lấy text trong dấu nháy bên trong [...] hoặc text thuần
        inner = re.findall(r"'([^']+)'", p)
        if inner:
            for w in inner:
                w = w.strip()
                if w and w not in seen:
                    cleaned.append(w)
                    seen.add(w)
        else:
            # Bỏ dấu ngoặc vuông và ngoặc đơn thừa
            p_clean = re.sub(r"[\[\]']", "", p).strip()
            if p_clean and p_clean not in seen:
                cleaned.append(p_clean)
                seen.add(p_clean)

    return " / ".join(cleaned)


def count_hanviet_readings(clean_pron: str) -> int:
    """Đếm số âm đọc Hán-Việt (đếm số phần tách bởi /)."""
    if not clean_pron:
        return 0
    return len([p for p in clean_pron.split("/") if p.strip()])


# ---------------------------------------------------------------------------
# 3. XỬ LÝ CHÍNH
# ---------------------------------------------------------------------------

def process():
    print(f"\n📖 Đang đọc: {INPUT_CSV}")
    rows = []
    with open(INPUT_CSV, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        original_fields = reader.fieldnames or []
        for row in reader:
            rows.append(row)
    print(f"   Tổng số ký tự: {len(rows):,}")

    # Tải bảng tần suất
    rank_map = load_junda_freq()

    # Làm sạch và bổ sung cột
    enriched = []
    no_rank_count = 0
    for row in rows:
        trad  = row.get("Phồn_thể", "").strip()
        simp  = row.get("Giản_thể", "").strip()
        raw_pron = row.get("Phiên_âm_Hán_Việt", "")

        # Tần suất rank: ưu tiên phồn thể, fallback giản thể
        rank = rank_map.get(trad) or rank_map.get(simp)
        if rank is None:
            no_rank_count += 1

        # Làm sạch phiên âm
        pron_clean = clean_hanviet(raw_pron)
        num_readings = count_hanviet_readings(pron_clean)

        enriched.append({
            "Phồn_thể":            trad,
            "Giản_thể":            simp,
            "Pinyin":              row.get("Pinyin", "").strip(),
            "Phiên_âm_Hán_Việt":  pron_clean,
            "Số_âm_HV":           num_readings,
            "Tần_suất_rank":       rank if rank is not None else "",
        })

    # Sắp xếp: có rank → theo rank tăng dần; không có rank → cuối, theo unicode
    def sort_key(r):
        rk = r["Tần_suất_rank"]
        if rk == "":
            return (1, 999_999_999, r["Phồn_thể"])
        return (0, int(rk), "")

    enriched.sort(key=sort_key)

    # Ghi file
    fieldnames = [
        "Phồn_thể",
        "Giản_thể",
        "Pinyin",
        "Phiên_âm_Hán_Việt",
        "Số_âm_HV",
        "Tần_suất_rank",
    ]
    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched)

    # Thống kê
    has_rank = sum(1 for r in enriched if r["Tần_suất_rank"] != "")
    print(f"\n✅ Hoàn thành!")
    print(f"   Tổng ký tự      : {len(enriched):,}")
    print(f"   Có tần suất rank: {has_rank:,}  ({has_rank/len(enriched)*100:.1f}%)")
    print(f"   Không có rank   : {no_rank_count:,}  (đặt cuối file)")
    print(f"   Đã lưu tại      : {OUTPUT_CSV}")

    # Preview 10 dòng đầu
    print("\n📋 Preview 10 ký tự phổ biến nhất:")
    print(f"{'Phồn':5} {'Giản':5} {'Pinyin':12} {'Phiên_âm_HV':25} {'#âm':4} {'Rank':6}")
    print("-" * 60)
    for r in enriched[:10]:
        print(f"{r['Phồn_thể']:5} {r['Giản_thể']:5} {r['Pinyin']:12} {r['Phiên_âm_Hán_Việt']:25} {r['Số_âm_HV']:4} {r['Tần_suất_rank']:6}")


if __name__ == "__main__":
    process()
