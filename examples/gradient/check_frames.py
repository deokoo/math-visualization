"""从预览成片抽取关键帧并拼接成检查图。"""
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
try:
    import imageio_ffmpeg
    FFMPEG = Path(imageio_ffmpeg.get_ffmpeg_exe())
except ImportError:
    FFMPEG = Path("ffmpeg")
SRC = HERE / "media/videos/preview_scene/676p30/Gradient.mp4"
OUT = HERE / "preview_check.jpg"
TIMES = [0.8, 2.6, 4.2, 6.2, 8.4, 10.2, 12.2, 14.2, 16.0, 17.7, 18.8, 20.8]

frames = []
for i, t in enumerate(TIMES):
    p = HERE / f"_check_{i:02d}.png"
    subprocess.run(
        [str(FFMPEG), "-y", "-loglevel", "error", "-ss", str(t),
         "-i", str(SRC), "-frames:v", "1", str(p)],
        check=True,
    )
    frames.append((t, Image.open(p).convert("RGB")))

tw, th = 270, 338
canvas = Image.new("RGB", (tw * 4, (th + 22) * 3), (24, 24, 26))
draw = ImageDraw.Draw(canvas)
for i, (t, im) in enumerate(frames):
    x, y = (i % 4) * tw, (i // 4) * (th + 22)
    canvas.paste(im.resize((tw, th)), (x, y + 22))
    draw.text((x + 6, y + 4), f"t={t:.1f}s", fill=(240, 240, 240))
canvas.save(OUT, quality=92)
for p in HERE.glob("_check_*.png"):
    p.unlink()
print(OUT)
