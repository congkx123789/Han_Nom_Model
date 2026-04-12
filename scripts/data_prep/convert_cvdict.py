import csv

input_file = '/tmp/CVDICT/CVDICT.u8'
output_file = '/home/alida/Documents/Cursor/Han_Nom_Model/data/CVDICT_Trung_Viet.csv'

with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8', newline='') as f_out:
    writer = csv.writer(f_out)
    writer.writerow(['Phồn_thể', 'Giản_thể', 'Pinyin', 'Nghĩa_Việt'])
    
    for line in f_in:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('%'):
            continue
        
        if ' [' in line and '] /' in line:
            parts1 = line.split(' [', 1)
            han_chars = parts1[0].split(' ', 1)
            trad = han_chars[0]
            simp = han_chars[1] if len(han_chars) > 1 else trad
            
            parts2 = parts1[1].split('] /', 1)
            pinyin = parts2[0]
            meanings = parts2[1].strip('/')
            
            writer.writerow([trad, simp, pinyin, meanings])

print(f"Successfully converted {input_file} to {output_file}")
