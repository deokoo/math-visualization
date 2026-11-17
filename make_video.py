#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
复刻视频号「架构师研究会」高斯积分科普短片（纯 Python 生成，含程序化背景音乐）

主题：
    I = ∫ e^(-x²) dx = √π —— 把积分平方升维成三维钟形曲面，
    切换极坐标（雅可比行列式 J = r），换元 u = r²，最终得到 π。

本版重点：
    1. 公式 mathtext 逐个「弹出」（缩放回弹 + 淡入），弹出后位置永久固定
    2. 3D 曲面补齐 x / y / z 三轴箭头与标注，墙上公式 e^(-x²) / e^(-y²)
    3. 逐笔描绘式动画：网格绘制、曲线追踪、光点扫掠、数值递增、数轴游标

运行：
    python make_video.py
依赖：
    numpy, pillow, matplotlib, imageio-ffmpeg
输出：
    gaussian_integral_replica.mp4 / bgm.wav
"""

from __future__ import annotations

import math
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

# ----------------------------------------------------------------------------
# 全局配置
# ----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
VIDEO_PATH = HERE / "gaussian_integral_replica.mp4"
BGM_PATH = HERE / "bgm.wav"

W, H = 1080, 1364
FPS = 30
DURATION = 58.0
TOTAL_FRAMES = int(FPS * DURATION)
SAMPLE_RATE = 44100

BG = (19, 20, 22)
WHITE = (240, 241, 238)
DIM = (150, 156, 162)
GRAY = (105, 112, 120)
CYAN = (75, 214, 200)
RED = (232, 96, 82)
ORANGE = (255, 168, 72)
PINK = (232, 150, 240)

CJK_FONT = "C:/Windows/Fonts/msyh.ttc"


# ----------------------------------------------------------------------------
# 通用工具
# ----------------------------------------------------------------------------
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def smooth(x: float) -> float:
    x = clamp01(x)
    return x * x * (3.0 - 2.0 * x)


def ramp(t: float, a: float, b: float) -> float:
    if b <= a:
        return 1.0 if t >= b else 0.0
    return smooth((t - a) / (b - a))


def rgba(c, a: float = 1.0):
    return (int(c[0]), int(c[1]), int(c[2]), int(255 * clamp01(a)))


def ease_out_back(u: float) -> float:
    """0→1 回弹曲线，末端轻微过冲，用于公式弹出。"""
    u = clamp01(u)
    c1, c3 = 1.70158, 2.70158
    v = 1.0 + c3 * (u - 1) ** 3 + c1 * (u - 1) ** 2
    return max(0.0, v)


def pop_state(t: float, start: float, dur: float = 0.75):
    """返回 (可见?, 缩放, 透明度)：start 时刻开始弹出，dur 秒内完成。"""
    if t < start:
        return False, 0.0, 0.0
    u = clamp01((t - start) / dur)
    return True, ease_out_back(u), min(1.0, u * 3.0)


# ---- mathtext 渲染 ---------------------------------------------------------
_MATH_CACHE: dict[tuple, Image.Image] = {}


def math_img(expr: str, size: int = 34, color=WHITE) -> Image.Image:
    key = (expr, size, color)
    if key in _MATH_CACHE:
        return _MATH_CACHE[key]
    fig = Figure(figsize=(18.0, 2.2), dpi=150)
    fig.patch.set_alpha(0.0)
    canvas = FigureCanvasAgg(fig)
    fig.text(0.01, 0.5, expr, fontsize=size,
             color="#%02x%02x%02x" % tuple(color), ha="left", va="center")
    canvas.draw()
    im = Image.fromarray(np.asarray(canvas.buffer_rgba()).copy(), mode="RGBA")
    bbox = im.getchannel("A").getbbox()
    im = im.crop(bbox) if bbox else Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    _MATH_CACHE[key] = im
    return im


def place(base: Image.Image, sprite: Image.Image, x: float, y: float,
          anchor: str = "mm", alpha: float = 1.0, scale: float = 1.0,
          glow: float = 0.0) -> None:
    """贴图。anchor: 第一个字符 l/m/r，第二个字符 t/m/b。"""
    if alpha <= 0 or sprite.width < 2:
        return
    if abs(scale - 1.0) > 1e-3:
        nw = max(1, int(sprite.width * scale))
        nh = max(1, int(sprite.height * scale))
        sprite = sprite.resize((nw, nh), Image.LANCZOS)
    w, h = sprite.size
    px, py = x, y
    if anchor[0] == "m":
        px -= w / 2
    elif anchor[0] == "r":
        px -= w
    if anchor[1] == "m":
        py -= h / 2
    elif anchor[1] == "b":
        py -= h
    px, py = int(round(px)), int(round(py))

    if glow > 0:
        halo = sprite.copy()
        a = halo.getchannel("A").point(lambda v: int(v * glow))
        halo.putalpha(a)
        halo = halo.filter(ImageFilter.GaussianBlur(9))
        base.alpha_composite(halo, (px, py))
    if alpha < 1.0:
        sprite = sprite.copy()
        sprite.putalpha(sprite.getchannel("A").point(lambda v: int(v * alpha)))
    base.alpha_composite(sprite, (px, py))


def put_math(base: Image.Image, expr: str, x: float, y: float, size: int,
             color=WHITE, anchor: str = "mm", alpha: float = 1.0,
             scale: float = 1.0, glow: float = 0.0) -> None:
    place(base, math_img(expr, size, color), x, y, anchor, alpha, scale, glow)


def math_pop(base: Image.Image, expr: str, x: float, y: float, size: int,
             t: float, start: float, color=WHITE, anchor: str = "mm",
             dur: float = 0.75, glow: float = 0.0) -> None:
    """公式弹出：缩放回弹 + 淡入，之后位置固定。"""
    vis, sc, al = pop_state(t, start, dur)
    if vis:
        put_math(base, expr, x, y, size, color, anchor, al, sc, glow * al)


def glow_line(ov: Image.Image, pts, color, width: int = 3,
              alpha: float = 1.0, frac: float = 1.0) -> None:
    """逐笔绘制的发光曲线。frac<1 时只画前 frac 部分。"""
    if len(pts) < 2 or alpha <= 0 or frac <= 0:
        return
    k = max(2, int(len(pts) * clamp01(frac)))
    p = [(int(round(x)), int(round(y))) for x, y in pts[:k]]
    d = ImageDraw.Draw(ov)
    for w_, a_ in ((width + 9, 0.09), (width + 4, 0.20), (width, 0.95)):
        d.line(p, fill=rgba(color, a_ * alpha), width=w_, joint="curve")
    # 笔尖高光
    hx, hy = p[-1]
    d.ellipse([hx - 4, hy - 4, hx + 4, hy + 4],
              fill=rgba(WHITE, 0.85 * alpha * (1.0 if frac >= 1 else 1.0)))


DEJAVU = str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf")


def text_sprite(text: str, px_size: int, color=WHITE) -> Image.Image:
    f = ImageFont.truetype(DEJAVU, px_size)
    tmp = Image.new("RGBA", (400, 120), (0, 0, 0, 0))
    ImageDraw.Draw(tmp).text((10, 60), text, font=f, anchor="lm",
                             fill=rgba(color, 1.0))
    bbox = tmp.getchannel("A").getbbox()
    return tmp.crop(bbox) if bbox else Image.new("RGBA", (1, 1), (0, 0, 0, 0))


def dynamic_text(base: Image.Image, text: str, x: float, y: float,
                 size: int = 26, color=WHITE, anchor: str = "mm",
                 alpha: float = 1.0) -> None:
    place(base, text_sprite(text, size, color), x, y, anchor, alpha)


# ----------------------------------------------------------------------------
# 背景
# ----------------------------------------------------------------------------
def make_background() -> Image.Image:
    yy, xx = np.mgrid[0:H, 0:W]
    nx = (xx - W / 2) / (W / 2)
    ny = (yy - H / 2) / (H / 2)
    d = np.sqrt(nx * nx * 0.75 + ny * ny)
    v = np.clip(1.0 - 0.34 * d, 0.0, 1.0)
    arr = np.zeros((H, W, 4), dtype=np.uint8)
    arr[..., :3] = (np.array(BG, dtype=np.float32) * v[..., None]).astype(np.uint8)
    arr[..., 3] = 255
    return Image.fromarray(arr, mode="RGBA")


BG_IMG = make_background()


# ----------------------------------------------------------------------------
# 3D 高斯曲面（含 x / y / z 三轴与墙面公式）
# ----------------------------------------------------------------------------
SURF_CX, SURF_CY, SURF_SCALE = 528, 300, 66
LIM, AMP, DECAY = 3.0, 1.45, 0.95
WALL_H = 1.95


def proj(x, y, z, az, el):
    ca, sa = math.cos(az), math.sin(az)
    X = x * ca - y * sa
    Y = x * sa + y * ca
    return (SURF_CX + SURF_SCALE * X,
            SURF_CY + SURF_SCALE * Y * math.sin(el)
            - SURF_SCALE * z * math.cos(el))


def draw_surface(ov: Image.Image, t: float, alpha: float) -> dict:
    """绘制钟形曲面 + 双背板墙 + 三轴；返回各标注的屏幕坐标。"""
    labels: dict[str, tuple[float, float]] = {}
    if alpha <= 0:
        return labels
    d = ImageDraw.Draw(ov)
    az = 0.66 + 0.030 * math.sin(t * 0.19)
    el = 0.37 + 0.014 * math.sin(t * 0.13)

    def P(x, y, z):
        return proj(x, y, z, az, el)

    draw_p = ramp(t, 0.6, 3.2)          # 网格逐笔绘出
    surf_p = ramp(t, 1.6, 5.2)          # 曲面逐笔绘出
    curve_p = ramp(t, 3.0, 6.4)         # 边缘曲线追踪

    # ── 两块竖直背板墙 ────────────────────────────────────
    wa = alpha * 0.30
    wall_lines = []
    for xv in np.linspace(-LIM, LIM, 11):
        wall_lines.append([P(xv, -LIM, 0), P(xv, -LIM, WALL_H)])
    for zv in np.linspace(0, WALL_H, 7):
        wall_lines.append([P(-LIM, -LIM, zv), P(LIM, -LIM, zv)])
    for yv in np.linspace(-LIM, LIM, 11):
        wall_lines.append([P(LIM, yv, 0), P(LIM, yv, WALL_H)])
    for zv in np.linspace(0, WALL_H, 7):
        wall_lines.append([P(LIM, -LIM, zv), P(LIM, LIM, zv)])
    for i, seg in enumerate(wall_lines):
        frac = clamp01(draw_p * len(wall_lines) - i)
        if frac > 0:
            p0, p1 = seg
            end = (p0[0] + (p1[0] - p0[0]) * frac,
                   p0[1] + (p1[1] - p0[1]) * frac)
            d.line([p0, end], fill=rgba(GRAY, wa), width=1)

    # ── 地面网格 ──────────────────────────────────────────
    ga = alpha * 0.34
    floor = []
    for v in np.linspace(-LIM, LIM, 16):
        floor.append([P(-LIM, v, 0), P(LIM, v, 0)])
        floor.append([P(v, -LIM, 0), P(v, LIM, 0)])
    for i, seg in enumerate(floor):
        frac = clamp01(draw_p * len(floor) - i)
        if frac > 0:
            p0, p1 = seg
            end = (p0[0] + (p1[0] - p0[0]) * frac,
                   p0[1] + (p1[1] - p0[1]) * frac)
            d.line([p0, end], fill=rgba(GRAY, ga), width=1)

    # ── 钟形曲面细网格 ────────────────────────────────────
    sa = alpha * 0.78
    xs = np.linspace(-LIM, LIM, 40)
    ys = np.linspace(-LIM, LIM, 40)
    curves = []
    for xv in xs:
        curves.append([P(xv, yv, AMP * math.exp(-DECAY * (xv * xv + yv * yv)))
                       for yv in ys])
    for yv in ys:
        curves.append([P(xv, yv, AMP * math.exp(-DECAY * (xv * xv + yv * yv)))
                       for xv in xs])
    for i, pts in enumerate(curves):
        frac = clamp01(surf_p * len(curves) - i)
        if frac > 0:
            col = (146, 152, 160) if i < len(xs) else (124, 130, 138)
            glow_line(ov, pts, col, 1, sa * 0.85, frac)
            # glow_line 会画光点，曲面网格不需要：仅保留线条即可

    # ── 峰顶高光：弹出 ────────────────────────────────────
    vis, sc, pa = pop_state(t, 5.4, 0.6)
    if vis:
        px, py = P(0, 0, AMP)
        for rr, aa in ((17, 0.14), (9, 0.34), (3.6, 0.92)):
            d.ellipse([px - rr * sc, py - rr * sc, px + rr * sc, py + rr * sc],
                      fill=rgba(PINK, aa * pa * alpha))

    # ── 墙上边缘分布曲线 e^(-x²) / e^(-y²) ────────────────
    gx = np.linspace(-LIM, LIM, 170)
    cyan_pts = [P(xv, -LIM, AMP * math.exp(-DECAY * xv * xv)) for xv in gx]
    red_pts = [P(LIM, yv, AMP * math.exp(-DECAY * yv * yv)) for yv in gx]
    glow_line(ov, cyan_pts, CYAN, 3, alpha, curve_p)
    glow_line(ov, red_pts, RED, 3, alpha, curve_p)

    # ── 扫描光面（持续运动）───────────────────────────────
    if t > 6.0:
        sweep = -LIM + 2 * LIM * ((t * 0.11) % 1.0)
        sw = [P(sweep, yv, AMP * math.exp(-DECAY * (sweep * sweep + yv * yv)))
              for yv in ys]
        d.line([(int(x), int(y)) for x, y in sw],
               fill=rgba(CYAN, alpha * 0.22), width=2)

    # ── 三轴箭头与标注坐标 ────────────────────────────────
    axis_a = alpha * ramp(t, 5.0, 6.5)
    corner = P(-LIM, -LIM, 0)          # 后角：z 轴起点
    z_top = P(-LIM, -LIM, WALL_H + 0.55)
    d.line([corner, z_top], fill=rgba(WHITE, 0.55 * axis_a), width=2)
    # 箭头
    dx, dy = z_top[0] - corner[0], z_top[1] - corner[1]
    ln = max(1e-6, math.hypot(dx, dy))
    ux, uy = dx / ln, dy / ln
    d.polygon([z_top,
               (z_top[0] - 9 * ux + 5 * uy, z_top[1] - 9 * uy - 5 * ux),
               (z_top[0] - 9 * ux - 5 * uy, z_top[1] - 9 * uy + 5 * ux)],
              fill=rgba(WHITE, 0.55 * axis_a))

    labels["z"] = (z_top[0], z_top[1] - 16)
    labels["x"] = P(LIM + 0.42, -LIM, 0)
    labels["y"] = P(-LIM, LIM + 0.42, 0)
    labels["ex"] = P(-2.65, -LIM, WALL_H * 0.92)
    labels["ey"] = P(LIM, 1.75, WALL_H * 0.92)
    return labels


# ----------------------------------------------------------------------------
# 极坐标网
# ----------------------------------------------------------------------------
PC = (166, 548)
PR = 142


def draw_polar(ov: Image.Image, t: float, alpha: float) -> None:
    if alpha <= 0:
        return
    d = ImageDraw.Draw(ov)
    cx, cy = PC
    a = alpha
    p = ramp(t, 4.0, 7.2)

    # 同心圆逐圈绘出
    for i in range(1, 8):
        r = PR * i / 7
        if p * 7 >= i:
            d.ellipse([cx - r, cy - r, cx + r, cy + r],
                      outline=rgba(GRAY, 0.48 * a), width=1)
    # 辐射线逐条绘出
    for i, deg in enumerate(range(0, 360, 30)):
        if p * 12 < i:
            continue
        ang = math.radians(deg)
        d.line([(cx, cy), (cx + PR * math.cos(ang), cy + PR * math.sin(ang))],
               fill=rgba(GRAY, 0.44 * a), width=1)
    # 横竖主轴
    d.line([(cx - PR - 14, cy), (cx + PR + 14, cy)],
           fill=rgba(GRAY, 0.70 * a), width=2)
    d.line([(cx, cy - PR - 14), (cx, cy + PR + 14)],
           fill=rgba(GRAY, 0.70 * a), width=2)

    # 高亮微元：dr × r dθ，持续绕圈
    th = (t * 0.50) % (2 * math.pi)
    dth, r0, r1 = 0.30, 0.30 * PR, 0.58 * PR
    ths = np.linspace(th, th + dth, 22)
    poly = ([(cx + r1 * math.cos(v), cy + r1 * math.sin(v)) for v in ths] +
            [(cx + r0 * math.cos(v), cy + r0 * math.sin(v)) for v in ths[::-1]])
    pulse = 0.70 + 0.30 * math.sin(t * 2.0)
    d.polygon(poly, fill=rgba(ORANGE, 0.22 * a * pulse))
    d.line(poly + [poly[0]], fill=rgba(ORANGE, 0.88 * a), width=3)


# ----------------------------------------------------------------------------
# 行列式矩阵 / 数轴
# ----------------------------------------------------------------------------
def draw_matrix(base: Image.Image, ov: Image.Image, entries, x, y,
                size=30, color=WHITE, alpha=1.0) -> float:
    if alpha <= 0:
        return 0.0
    sprites = [[math_img(e, size, color) if e.startswith("$")
                else text_sprite(e, int(size * 1.5), color)
                for e in row] for row in entries]
    cw = max(s.width for row in sprites for s in row) + 26
    ch = max(s.height for row in sprites for s in row) + 18
    d = ImageDraw.Draw(ov)
    bx, by = int(x), int(y - ch)
    d.line([(bx + 9, by), (bx, by), (bx, by + 2 * ch)],
           fill=rgba(color, 0.9 * alpha), width=3)
    rx = bx + 2 * cw + 18
    d.line([(rx - 9, by), (rx, by), (rx, by + 2 * ch)],
           fill=rgba(color, 0.9 * alpha), width=3)
    for i, row in enumerate(sprites):
        for j, s in enumerate(row):
            place(base, s, bx + 9 + j * cw + (cw - 18 - s.width) / 2,
                  by + i * ch + (ch - s.height) / 2, "lt", alpha)
    return 2 * cw + 28


def draw_numberline(ov: Image.Image, alpha: float, progress: float) -> None:
    if alpha <= 0:
        return
    d = ImageDraw.Draw(ov)
    x0, x1, y = 322, 800, 582
    d.line([(x0, y), (x1, y)], fill=rgba(DIM, 0.70 * alpha), width=2)
    d.line([(x1, y - 7), (x1, y + 16)], fill=rgba(DIM, 0.9 * alpha), width=2)
    f = ImageFont.truetype(DEJAVU, 22)
    d.text((x1, y + 22), "π", font=f, anchor="mt", fill=rgba(DIM, 0.85 * alpha))
    dot = x0 + (x1 - x0) * clamp01(progress)
    d.ellipse([dot - 8, y - 8, dot + 8, y + 8],
              fill=rgba(RED, 0.95 * alpha))


# ----------------------------------------------------------------------------
# 背景音乐：程序化 ambient + 琶音（无版权依赖）
# ----------------------------------------------------------------------------
def make_bgm(path: Path) -> None:
    dur = DURATION + 1.0
    n = int(SAMPLE_RATE * dur)
    audio = np.zeros(n, dtype=np.float64)
    prog = [(220.00, "min"), (174.61, "maj"), (261.63, "maj"), (196.00, "maj")]
    bar = 8.0
    for cycle in range(2):
        for i, (root, kind) in enumerate(prog):
            start = cycle * 4 * bar + i * bar
            if start >= DURATION:
                break
            end = min(start + bar, DURATION)
            i0, i1 = int(start * SAMPLE_RATE), int(end * SAMPLE_RATE)
            tt = np.arange(i1 - i0) / SAMPLE_RATE
            intervals = [0, 3, 7] if kind == "min" else [0, 4, 7]
            for semi in intervals:
                f = root * 2 ** (semi / 12)
                for det in (-2.0, 2.0):
                    audio[i0:i1] += 0.042 * np.sin(
                        2 * np.pi * (f + det) * tt + 0.4 * i)
            audio[i0:i1] += 0.085 * np.sin(2 * np.pi * (root / 2) * tt)
            arp = [root * 2 ** (s / 12) * 2 for s in intervals]
            k = 0
            while k * 0.5 < end - start:
                ts = k * 0.5
                s0 = int((start + ts) * SAMPLE_RATE)
                s1 = min(int((start + ts + 0.44) * SAMPLE_RATE), n)
                if s1 <= s0:
                    break
                env = np.exp(-np.arange(s1 - s0) / SAMPLE_RATE * 7.5)
                f = arp[k % 3]
                local = np.arange(s1 - s0) / SAMPLE_RATE
                audio[s0:s1] += 0.085 * (
                    np.sin(2 * np.pi * f * local) +
                    0.30 * np.sin(4 * np.pi * f * local)) * env
                k += 1
    rng = np.random.default_rng(11)
    audio += 0.0045 * rng.normal(size=n)
    fi, fo = int(2.0 * SAMPLE_RATE), int(3.0 * SAMPLE_RATE)
    audio[:fi] *= np.linspace(0, 1, fi)
    audio[-fo:] *= np.linspace(1, 0, fo)
    audio /= max(1e-9, float(np.max(np.abs(audio))))
    audio *= 0.82
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((audio * 32767).astype("<i2").tobytes())


# ----------------------------------------------------------------------------
# 单帧渲染
# ----------------------------------------------------------------------------
def render_frame(t: float) -> Image.Image:
    img = BG_IMG.copy()
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))

    # 顶部呼吸光标
    cursor = (1.0 - ramp(t, 0.4, 2.0)) + 0.22 * (1.0 - ramp(t, 2.0, 3.4))
    if cursor > 0:
        ImageDraw.Draw(ov).line([(W / 2, 40), (W / 2, 74)],
                                fill=rgba(WHITE, cursor), width=5)

    # 3D 曲面 / 极坐标网
    labels = draw_surface(ov, t, 1.0)
    draw_polar(ov, t, 1.0)

    # 参数动画：初始值与封面一致（r=0.35, θ=0.25），随后递增/旋转
    theta = 0.25 + t * 0.34
    r = 0.35 + 0.95 * ramp(t, 2.0, 46.0) + 0.025 * math.sin(t * 1.1)
    r = max(0.30, r)
    est = math.pi * ramp(t, 3.0, 50.0) + \
          0.012 * math.sin(t * 2.3) * math.exp(-t / 30.0)
    draw_numberline(ov, 1.0, est / math.pi)
    img = Image.alpha_composite(img, ov)

    # ── 三轴与墙面公式标注（弹出）────────────────────────
    for k, (expr, col, sz) in {
        "x": (r"$x$", GRAY, 22),
        "y": (r"$y$", GRAY, 22),
        "z": (r"$z$", GRAY, 22),
        "ex": (r"$e^{-x^2}$", CYAN, 21),
        "ey": (r"$e^{-y^2}$", RED, 21),
    }.items():
        if k in labels:
            math_pop(img, expr, labels[k][0], labels[k][1], sz,
                     t, 5.6 + {"x": 0, "y": 0.1, "z": 0.2,
                               "ex": 0.35, "ey": 0.5}[k], col, "mm", 0.55)

    # 极坐标微元的 dr / dθ 标注
    math_pop(img, r"$dr$", PC[0] + 60, PC[1] + 54, 14, t, 7.4, ORANGE)
    math_pop(img, r"$d\theta$", PC[0] + 100, PC[1] + 24, 14, t, 7.6, ORANGE)

    # ── 弹出公式层：弹出后位置永久固定 ───────────────────
    # A. 数值雅可比矩阵 = 红字 det
    vis_a, sc_a, al_a = pop_state(t, 0.9, 0.8)
    if vis_a:
        put_math(img, r"$J=\mathrm{det}$", 326, 468, 24, WHITE, "lm", al_a, sc_a)
        vals = [[math.cos(theta), -r * math.sin(theta)],
                [math.sin(theta), r * math.cos(theta)]]
        ov2 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw_matrix(img, ov2, [[f"{v:+.2f}" for v in row] for row in vals],
                    516, 474, 18, WHITE, al_a)
        img = Image.alpha_composite(img, ov2)
        put_math(img, r"$=$", 842, 474, 24, WHITE, "lm", al_a)
        dynamic_text(img, f"{r:.2f}", 890, 474, 24, RED, "lm", al_a)

    # B. I² ≈ 估计值
    math_pop(img, r"$I^2\approx$", 326, 548, 27, t, 3.4, WHITE, "lm")
    dynamic_text(img, f"{est:.4f}", 452, 548, 27, RED, "lm",
                 ramp(t, 3.4, 4.2))

    # C. 双积分
    math_pop(img,
             r"$I^2=\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}"
             r"e^{-(x^2+y^2)}\,dx\,dy$",
             W / 2, 688, 27, t, 8.0, WHITE, "mm", 0.8, 0.18)

    # D. 雅可比矩阵推导（两组矩阵）
    vis_d, sc_d, al_d = pop_state(t, 13.0, 0.85)
    if vis_d:
        put_math(img, r"$J=\mathrm{det}$", 50, 800, 22, WHITE, "lm", al_d, sc_d)
        ov2 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        mw1 = draw_matrix(img, ov2,
                          [[r"$\frac{\partial x}{\partial r}$",
                            r"$\frac{\partial x}{\partial\theta}$"],
                           [r"$\frac{\partial y}{\partial r}$",
                            r"$\frac{\partial y}{\partial\theta}$"]],
                          205, 800, 17, WHITE, al_d)
        img = Image.alpha_composite(img, ov2)
        put_math(img, r"$=\mathrm{det}$", 205 + mw1 + 18, 800, 22,
                 WHITE, "lm", al_d)
        ov2 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        mw2 = draw_matrix(img, ov2,
                          [[r"$\cos\theta$", r"$-r\sin\theta$"],
                           [r"$\sin\theta$", r"$r\cos\theta$"]],
                          205 + mw1 + 166, 800, 15, WHITE, al_d)
        img = Image.alpha_composite(img, ov2)
        put_math(img, r"$=r$", 205 + mw1 + 166 + mw2 + 16, 800, 25,
                 WHITE, "lm", al_d)

    # E. 极坐标积分
    math_pop(img,
             r"$I^2=\int_0^{2\pi}\int_0^{\infty}e^{-r^2}r\,dr\,d\theta"
             r"=\int_0^{2\pi}\frac{1}{2}\,d\theta=\pi$",
             W / 2, 896, 23, t, 19.0, WHITE, "mm", 0.8, 0.18)

    # F. 最终答案
    math_pop(img, r"$I=\int_{-\infty}^{\infty}e^{-x^2}dx=\sqrt{\pi}$",
             W / 2, 984, 30, t, 25.0, WHITE, "mm", 0.9, 0.22)

    # 收尾淡出
    if t > DURATION - 1.5:
        fade = 1.0 - (t - (DURATION - 1.5)) / 1.5
        img = Image.alpha_composite(
            Image.new("RGBA", (W, H), (0, 0, 0, 255)),
            img)
        img.putalpha(img.getchannel("A").point(lambda v: int(v * fade)))
    return img.convert("RGB")


# ----------------------------------------------------------------------------
# 编码输出
# ----------------------------------------------------------------------------
def get_ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        p = shutil.which("ffmpeg")
        if not p:
            raise RuntimeError("未找到 ffmpeg，请 pip install imageio-ffmpeg")
        return p


def encode() -> None:
    make_bgm(BGM_PATH)
    cmd = [
        get_ffmpeg(), "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-i", str(BGM_PATH),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", "192k", "-shortest",
        str(VIDEO_PATH),
    ]
    with (HERE / "ffmpeg.log").open("wb") as log:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=log)
        assert proc.stdin is not None
        for i in range(TOTAL_FRAMES):
            proc.stdin.write(render_frame(i / FPS).tobytes())
            if i % 150 == 0:
                print(f"[{i:4d}/{TOTAL_FRAMES}] {i / FPS:5.1f}s", flush=True)
        proc.stdin.close()
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg 编码失败（退出码 {code}）")
    print(f"完成：{VIDEO_PATH}  ({VIDEO_PATH.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    encode()
