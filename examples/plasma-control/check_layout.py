# -*- coding: utf-8 -*-
from PIL import Image, ImageDraw
import os
D = "C:/Users/admin/WorkBuddy/automation-2026-11-18-11-29-46/manim_plasma_control/preview_grid"
ks = ["k%02d.png" % i for i in range(21)]
ims = [Image.open(os.path.join(D, k)).convert("RGB") for k in ks]
w, h = ims[0].size
scale = 0.30
tw, th = int(w*scale), int(h*scale)
cols = 5
rows = 5
canvas = Image.new("RGB", (cols*tw, rows*(th+18)), (18, 18, 18))
dr = ImageDraw.Draw(canvas)
for i, (im, k) in enumerate(zip(ims, ks)):
    r, c = divmod(i, cols)
    x, y = c*tw, r*(th+18)
    canvas.paste(im.resize((tw, th), Image.LANCZOS), (x, y))
    dr.text((x+5, y+th+2), "t=%ds" % (i*70//20), fill=(255, 255, 255))
p = "C:/Users/admin/WorkBuddy/automation-2026-11-18-11-29-46/manim_plasma_control/preview_contact.jpg"
canvas.save(p, quality=92)
print(p, canvas.size)
