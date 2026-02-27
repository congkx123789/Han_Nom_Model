from PIL import Image, ImageDraw, ImageFont
import os

def test_font(font_path, chars, output_path):
    if not os.path.exists(font_path):
        print(f"Font not found: {font_path}")
        return

    image = Image.new("L", (200, 100), color=255)
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype(font_path, 40)
        draw.text((10, 10), chars, font=font, fill=0)
        image.save(output_path)
        print(f"Saved test image to {output_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    font_path = "/home/alida/Documents/Cursor/Han_Nom_Model/fonts/NomNaTong-Regular.ttf"
    # Test with:
    # 1. Basic ASCII (A)
    # 2. Common Han (One - 一)
    # 3. Common Nom (Nam - 𡨸 - wait, Nam is 南, 𡨸 is Chu Nom for 'Chu')
    # Let's try 'Nam' (South) - 南 which is common.
    test_chars = "A 一 南" 
    test_font(font_path, test_chars, "/home/alida/Documents/Cursor/Han_Nom_Model/test_font_render.png")
