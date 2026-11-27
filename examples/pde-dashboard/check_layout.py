"""把 21 张半分辨率关键帧拼成检查图（7 列 x 3 行）。"""
import os
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "preview_grid")
OUT = os.path.join(HERE, "preview_contact3.jpg")
TW, TH = 270, 338
files = sorted(f for f in os.listdir(SRC) if f.endswith(".png"))
canvas = Image.new("RGB", (TW * 7, (TH + 20) * 3), (24, 24, 26))
d = ImageDraw.Draw(canvas)
for i, fn in enumerate(files):
    im = Image.open(os.path.join(SRC, fn)).convert("RGB").resize((TW, TH), Image.LANCZOS)
    x, y = (i % 7) * TW, (i // 7) * (TH + 20)
    canvas.paste(im, (x, y + 20))
    d.text((x + 5, y + 3), fn, fill=(240, 240, 240))
canvas.save(OUT, quality=90)
print(OUT)
