"""从预览成片抽取关键帧并拼接成检查图。"""
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw
import imageio_ffmpeg

HERE = Path(__file__).resolve().parent
FFMPEG = Path(imageio_ffmpeg.get_ffmpeg_exe())
cands = sorted(HERE.glob("media/videos/preview_scene/*/DivergenceTheorem.mp4"))
if not cands:
    raise SystemExit("找不到预览视频")
SRC = cands[-1]
OUT = HERE / "preview_check.jpg"
TIMES = [0.8, 2.4, 4.6, 6.2, 8.4, 10.4, 12.2, 14.4, 16.6, 18.4, 20.0, 21.4]

frames = []
for i, t in enumerate(TIMES):
    p = HERE / f"_check_{i:02d}.png"
    subprocess.run([str(FFMPEG), "-y", "-loglevel", "error", "-ss", str(t),
                    "-i", str(SRC), "-frames:v", "1", str(p)], check=True)
    frames.append((t, Image.open(p).convert("RGB")))

tw, th = 270, 338
canvas = Image.new("RGB", (tw * 4, (th + 22) * 3), (24, 24, 26))
draw = ImageDraw.Draw(canvas)
for i, (t, im) in enumerate(frames):
    x, y = (i % 4) * tw, (i // 4) * (th + 22)
    canvas.paste(im.resize((tw, th)), (x, y + 22))
    draw.text((x + 6, y + 4), f"t={t:.1f}s", fill=(240, 240, 240))
canvas.save(OUT, quality=90)
for p in HERE.glob("_check_*.png"):
    p.unlink()
print(OUT)
