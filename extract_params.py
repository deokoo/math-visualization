from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
FR = ROOT / "original_frames"
OUT = ROOT / "frame_analysis" / "params.csv"
frames = sorted(FR.glob("frame_*.png"))

CX, CY = 110, 580
BAR_X0, BAR_XPI, BAR_Y = 204, 546, 620

rows = []
for idx, path in enumerate(frames):
    a = np.asarray(Image.open(path).convert("RGB")).astype(np.int16)

    # I^2 from red progress bar
    line = a[BAR_Y - 2 : BAR_Y + 3, 190:560]
    red_line = (line[:, :, 0] > 110) & (line[:, :, 1] < 110) & (line[:, :, 2] < 110)
    xs = np.where(red_line)[1] + 190
    i2 = 0.0 if len(xs) == 0 else float(np.clip((xs.max() - BAR_X0) / (BAR_XPI - BAR_X0), 0, 1) * np.pi)

    # polar region
    sub = a[440:720, 0:195]
    yy, xx = np.mgrid[440:720, 0:195]
    dist = np.hypot(xx - CX, yy - CY)
    red = (sub[:, :, 0] > 120) & (sub[:, :, 1] < 100) & (sub[:, :, 2] < 100) & (dist < 150)
    if red.sum() > 0:
        d = dist[red]
        r_mean, r_max, r_min = float(d.mean()), float(d.max()), float(d.min())
    else:
        r_mean = r_max = r_min = 0.0

    bright = (sub[:, :, 0] > 185) & (sub[:, :, 1] > 185) & (sub[:, :, 2] > 185) & (dist > 18) & (dist < 145)
    if bright.sum() > 0:
        ang = np.arctan2(-(yy[bright] - CY), xx[bright] - CX)
        hist, _ = np.histogram(ang, bins=360, range=(-np.pi, np.pi))
        kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        pad = np.r_[hist[-2:], hist, hist[:2]]
        sm = np.convolve(pad, kernel / kernel.sum(), mode="same")[2:-2]
        theta = float((int(np.argmax(sm)) - 180) * np.pi / 180.0)
    else:
        theta = float("nan")

    rows.append((idx + 1, idx / 30.34, i2, r_mean, r_max, r_min, theta, int(red.sum())))

with OUT.open("w", encoding="utf-8") as f:
    f.write("frame,t,i2,r_mean,r_max,r_min,theta,nred\n")
    for row in rows:
        f.write(",".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in row) + "\n")

print("PARAMS_SAVED", OUT, "rows", len(rows))
for target in (1, 100, 300, 491, 643):
    row = rows[target - 1]
    print(f"frame={row[0]:04d} t={row[1]:5.2f} i2={row[2]:.4f} rmean={row[3]:.1f} rmax={row[4]:.1f} theta_deg={np.degrees(row[6]):7.2f} nred={row[7]}")
