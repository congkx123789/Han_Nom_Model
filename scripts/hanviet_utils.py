import pandas as pd
import os
import ast

class HanVietConverter:
    def __init__(self, csv_path):
        self.data = {}
        self.load_data(csv_path)

    def load_data(self, csv_path):
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found.")
            return

        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            char = row['char']
            hanviet = row['hanviet']
            pinyin = str(row['pinyin'])

            # Parse hanviet string list to actual list
            try:
                if isinstance(hanviet, str):
                    hanviet_list = ast.literal_eval(hanviet)
                else:
                    hanviet_list = []
            except (ValueError, SyntaxError):
                hanviet_list = []

            if char not in self.data:
                self.data[char] = {}
            
            self.data[char][pinyin] = hanviet_list

    def get_hanviet(self, text, pinyins):
        if len(text) != len(pinyins):
            raise ValueError("Text and pinyins must have the same length.")

        result = []
        for char, pinyin in zip(text, pinyins):
            # Preserve alphanumeric characters
            if char.isalnum() and (ord(char) < 128):
                result.append(char)
                continue

            if char in self.data:
                char_data = self.data[char]
                # Check for specific pinyin
                if pinyin in char_data:
                    result.append(", ".join(char_data[pinyin]))
                # Check for wildcard
                elif '*' in char_data:
                    result.append(", ".join(char_data['*']))
                else:
                    result.append("_")
            else:
                result.append("_")

        return " ".join(result)

# Singleton instance
_converter = None

def get_hanviet(text, pinyins):
    global _converter
    if _converter is None:
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'hanviet.csv')
        _converter = HanVietConverter(csv_path)
    return _converter.get_hanviet(text, pinyins)
