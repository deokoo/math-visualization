from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
FR = ROOT / "original_frames"
OUT = ROOT / "frame_analysis"
OUT.mkdir(exist_ok=True)

frames = sorted(FR.glob("frame_*.png"))
print("FRAME_COUNT", len(frames))


def make_sheet(indices, name, cols=6, cell=(228, 510), thumb=(216, 480)):
    rows = max(1, (len(indices) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * cell[0], rows * cell[1]), (18, 18, 22))
    draw = ImageDraw.Draw(sheet)
    for k, idx in enumerate(indices):
        if idx >= len(frames):
            continue
        im = Image.open(frames[idx]).convert("RGB")
        im.thumbnail(thumb)
        x0 = (k % cols) * cell[0]
        y0 = (k // cols) * cell[1]
        sheet.paste(im, (x0 + (cell[0] - im.width) // 2, y0 + 24))
        draw.text((x0 + 6, y0 + 5), f"#{idx + 1:04d}  {idx / 30.34:05.2f}s", fill=(255, 220, 120))
    path = OUT / name
    sheet.save(path, quality=92)
    print("SAVED", path, sheet.size)


make_sheet(list(range(0, len(frames), 15)), "sheet_overview.jpg", cols=6, cell=(228, 510))
make_sheet(list(range(0, min(120, len(frames)), 5)), "sheet_intro.jpg", cols=6, cell=(228, 510))
make_sheet(list(range(max(0, len(frames) - 120), len(frames), 5)), "sheet_outro.jpg", cols=6, cell=(228, 510))

prev = None
scores = []
for i, path in enumerate(frames):
    arr = np.asarray(Image.open(path).convert("L").resize((144, 320)), dtype=np.float32)
    if prev is not None:
        scores.append((float(np.mean(np.abs(arr - prev))), i))
    prev = arr

print("TOP_MOTION")
for score, i in sorted(scores, reverse=True)[:30]:
    print(f"frame={i + 1:04d} t={i / 30.34:05.2f}s diff={score:6.2f}")
