#!/usr/bin/env python3
"""
merge_new_han_chars.py
======================
Merge 1,493 ký tự Hán-Việt mới từ file root vào file data/ đã enrich:
  - File nguồn mới : han_characters_only.csv (root, 4 cột)
  - File đích enrich: data/han_characters_only.csv (6 cột)

Các ký tự mới sẽ được:
  1. Lấy Pinyin + Phiên_âm_Hán_Việt từ file nguồn
  2. Tính Số_âm_HV (đếm âm đọc)
  3. Tra Tần_suất_rank từ bảng Jieba
  4. Merge + re-sort theo tần suất
  5. Ghi lại vào data/han_characters_only.csv
"""

import os
import csv
import re

ROOT      = "/home/alida/Documents/Cursor/Han_Nom_Model"
NEW_FILE  = os.path.join(ROOT, "han_characters_only.csv")
DATA_FILE = os.path.join(ROOT, "data", "han_characters_only.csv")
FREQ_CACHE= os.path.join(ROOT, "data", "raw", "junda_char_freq.txt")

FIELDNAMES = [
    "Phồn_thể",
    "Giản_thể",
    "Pinyin",
    "Phiên_âm_Hán_Việt",
    "Số_âm_HV",
    "Tần_suất_rank",
]

# -------------------------------------------------------------------------
# 1. Load bảng tần suất Jieba
# -------------------------------------------------------------------------

def load_rank_map() -> dict[str, int]:
    if not os.path.exists(FREQ_CACHE):
        print("⚠️  Không tìm thấy bảng tần suất. Chạy sort_and_enrich_han_chars.py trước.")
        return {}

    freq_map: dict[str, int] = {}
    with open(FREQ_CACHE, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and len(parts[0]) == 1:
                try:
                    freq_map[parts[0]] = int(parts[1])
                except ValueError:
                    pass

    sorted_chars = sorted(freq_map.items(), key=lambda x: -x[1])
    rank_map = {ch: rk + 1 for rk, (ch, _) in enumerate(sorted_chars)}
    print(f"📊 Bảng tần suất: {len(rank_map):,} ký tự")
    return rank_map


# -------------------------------------------------------------------------
# 2. Đếm số âm HV
# -------------------------------------------------------------------------

def count_hanviet_readings(pron: str) -> int:
    if not pron or not pron.strip():
        return 0
    return len([p for p in pron.split("/") if p.strip()])


# -------------------------------------------------------------------------
# 3. Đọc cả 2 file
# -------------------------------------------------------------------------

def read_csv(path: str, encoding="utf-8-sig") -> tuple[list, list[str]]:
    rows = []
    with open(path, encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for r in reader:
            rows.append(dict(r))
    return rows, fieldnames


# -------------------------------------------------------------------------
# 4. Merge & enrich
# -------------------------------------------------------------------------

def merge():
    print(f"\n📖 Đọc file đã enrich: {DATA_FILE}")
    old_rows, _ = read_csv(DATA_FILE)
    print(f"   → {len(old_rows):,} ký tự")

    print(f"📖 Đọc file mới: {NEW_FILE}")
    new_rows, _ = read_csv(NEW_FILE)
    print(f"   → {len(new_rows):,} ký tự")

    # Index file cũ bằng Phồn_thể
    old_index: dict[str, dict] = {r["Phồn_thể"]: r for r in old_rows}

    # Load bảng tần suất
    rank_map = load_rank_map()

    # Tìm ký tự mới (chưa có trong data/)
    new_entries = []
    updated_count = 0
    for r in new_rows:
        trad = r["Phồn_thể"].strip()
        simp = r["Giản_thể"].strip()
        pinyin = r["Pinyin"].strip()
        pron   = r["Phiên_âm_Hán_Việt"].strip()

        if trad not in old_index:
            # Ký tự hoàn toàn mới
            rank = rank_map.get(trad) or rank_map.get(simp)
            new_entries.append({
                "Phồn_thể":           trad,
                "Giản_thể":           simp,
                "Pinyin":             pinyin,
                "Phiên_âm_Hán_Việt":  pron,
                "Số_âm_HV":          count_hanviet_readings(pron),
                "Tần_suất_rank":      rank if rank is not None else "",
            })
        else:
            # Ký tự đã có — nếu file mới có Pinyin/phiên âm tốt hơn (không rỗng) thì update
            existing = old_index[trad]
            changed = False
            if not existing.get("Pinyin") and pinyin:
                existing["Pinyin"] = pinyin
                changed = True
            if not existing.get("Phiên_âm_Hán_Việt") and pron:
                existing["Phiên_âm_Hán_Việt"] = pron
                existing["Số_âm_HV"] = count_hanviet_readings(pron)
                changed = True
            if changed:
                updated_count += 1

    print(f"\n🔎 Kết quả đối chiếu:")
    print(f"   Ký tự MỚI cần thêm : {len(new_entries):,}")
    print(f"   Ký tự cũ được cập nhật: {updated_count:,}")

    # Ghép 2 danh sách
    all_rows = list(old_index.values()) + new_entries

    # Normalize: đảm bảo tất cả có đủ 6 cột
    for r in all_rows:
        if "Số_âm_HV" not in r or r["Số_âm_HV"] == "":
            r["Số_âm_HV"] = count_hanviet_readings(r.get("Phiên_âm_Hán_Việt", ""))
        if "Tần_suất_rank" not in r:
            trad = r["Phồn_thể"]
            simp = r.get("Giản_thể", trad)
            rk = rank_map.get(trad) or rank_map.get(simp)
            r["Tần_suất_rank"] = rk if rk is not None else ""

    # Sort theo tần suất
    def sort_key(r):
        rk = r.get("Tần_suất_rank", "")
        if rk == "" or rk is None:
            return (1, 999_999_999, r["Phồn_thể"])
        try:
            return (0, int(rk), "")
        except (ValueError, TypeError):
            return (1, 999_999_999, r["Phồn_thể"])

    all_rows.sort(key=sort_key)

    # Ghi file
    with open(DATA_FILE, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    # Thống kê
    has_rank    = sum(1 for r in all_rows if r.get("Tần_suất_rank") not in ("", None))
    has_pron    = sum(1 for r in all_rows if r.get("Phiên_âm_Hán_Việt"))
    has_pinyin  = sum(1 for r in all_rows if r.get("Pinyin"))

    print(f"\n✅ ĐÃ GHI: {DATA_FILE}")
    print(f"   Tổng ký tự      : {len(all_rows):,}")
    print(f"   Có Pinyin       : {has_pinyin:,}  ({has_pinyin/len(all_rows)*100:.1f}%)")
    print(f"   Có Phiên âm HV  : {has_pron:,}  ({has_pron/len(all_rows)*100:.1f}%)")
    print(f"   Có tần suất rank: {has_rank:,}  ({has_rank/len(all_rows)*100:.1f}%)")

    print(f"\n📋 Preview 15 ký tự phổ biến nhất sau merge:")
    print(f"{'Rank':>6}  {'Phồn':5} {'Giản':5} {'Pinyin':12} {'Phiên_âm_HV':30} {'#âm':4}")
    print("-" * 70)
    for r in all_rows[:15]:
        print(f"{str(r.get('Tần_suất_rank','')):>6}  {r['Phồn_thể']:5} {r['Giản_thể']:5} "
              f"{r['Pinyin']:12} {r['Phiên_âm_Hán_Việt']:30} {r['Số_âm_HV']:4}")


if __name__ == "__main__":
    merge()
