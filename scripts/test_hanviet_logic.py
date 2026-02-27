from hanviet_utils import get_hanviet

def test():
    examples = [
        ('中國', ['zhong1', 'guo2'], 'trung quốc'),
        ('百發百中', ['bai3', 'fa1', 'bai3', 'zhong4'], 'bách phát bách trúng'),
        ('投降', ['tou2', 'xiang2'], 'đầu hàng'),
        ('降級', ['jiang4', 'ji2'], 'giáng cấp'),
        ('和尚', ['he2', 'shang5'], 'hoà thượng'),
        ('X光', ['X', 'guang1'], 'X quang'),
        ('斷', ['duan4'], 'đoạn, đoán')
    ]

    for text, pinyins, expected in examples:
        result = get_hanviet(text, pinyins)
        print(f"Text: {text}, Pinyins: {pinyins}")
        print(f"Result:   {result}")
        print(f"Expected: {expected}")
        print("-" * 20)

if __name__ == "__main__":
    test()
