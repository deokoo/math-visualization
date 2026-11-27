# -*- coding: utf-8 -*-
"""Stitch 21 preview frames into a contact sheet for layout inspection."""
from PIL import Image, ImageDraw
import os

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preview_grid")
files = sorted(f for f in os.listdir(D) if f.startswith("k") and f.endswith(".png"))
ims = [Image.open(os.path.join(D, f)).convert("RGB") for f in files]
w, h = ims[0].size
cols, rows = 5, 5
cw, ch = w, h
canvas = Image.new("RGB", (cols*cw, rows*(ch+18)), (18, 18, 20))
dr = ImageDraw.Draw(canvas)
for i, im in enumerate(ims):
    r, c = divmod(i, cols)
    x, y = c*cw, r*(ch+18)
    canvas.paste(im, (x, y))
    dr.text((x+5, y+ch+2), files[i], fill=(240, 240, 240))
out = os.path.join(os.path.dirname(D), "preview_contact.jpg")
canvas.save(out, quality=90)
print(out, canvas.size)
