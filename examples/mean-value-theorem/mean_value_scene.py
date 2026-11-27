#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manim 数学动画：拉格朗日中值定理 —— 平均变化率一定由某点的瞬时变化率实现。

与系列前作同风格：
  - 1080x1350 竖屏、30fps、约 22.3s
  - 公式全程固定，只动参数 / 点 / 切线 / 数值
  - 无 LaTeX：公式用 matplotlib mathtext 预渲染为透明 PNG
  - 暗色底 #101113、中灰清晰线条、青/红边缘曲线、纯内存动态数字
  - 顶部中文主题 + 英文标题，整体淡入后常亮，无打字机竖线

画面：
  左栏 = f(x)=sin x 在 [a,b] 上的曲线、端点割线、升/宽直角三角形、可滑动切线；
  右栏 = 导函数 f'(x)=cos x 的曲线与水平线 k，交点标出 ξ；
  中下 = 割线斜率 k、切线斜率 f'(c)、两者差值三个动态读数；
  底部 = 五行固定公式链 + 随阶段切换的中文讲述行。

运行：
    python precheck.py
    manim -pql preview_scene.py MeanValueTheorem
    python check_frames.py
    manim -pqh mean_value_scene.py MeanValueTheorem
    python add_bgm.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
from PIL import Image, ImageDraw, ImageFont

from manim import *

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
FORMULA_DIR = HERE / "formulas"
FORMULA_DIR.mkdir(exist_ok=True)

config.pixel_width = 1080
config.pixel_height = 1350
config.frame_width = 8
config.frame_height = 10
config.background_color = "#101113"
config.frame_rate = 30

DURATION = 21.26

C_TEXT = (236, 232, 220)
C_FORMULA = (236, 232, 220)
C_GRID = (104, 110, 114)
C_GRID_DIM = (66, 70, 74)
C_CYAN = (74, 214, 200)
C_RED = (232, 96, 82)
C_ORANGE = (238, 148, 72)

M_TEXT = "#ECE8DC"
M_CYAN = "#4AD6C8"
M_RED = "#E86052"
M_ORANGE = "#EE9448"

_SERIF = str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSerif.ttf")

# ---------------------------------------------------------------------------
# 无 LaTeX 公式预渲染 + 纯内存动态数字
# ---------------------------------------------------------------------------
_MATH_CACHE: dict[tuple, Image.Image] = {}


def _math_img(expr: str, size: int, color) -> Image.Image:
    if isinstance(color, str):
        color = tuple(int(c) for c in ManimColor(color).to_rgb() * 255)
    key = (expr, size, color)
    if key in _MATH_CACHE:
        return _MATH_CACHE[key]
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(16.0, 2.4), dpi=200)
    fig.patch.set_alpha(0.0)
    canvas = FigureCanvasAgg(fig)
    fig.text(0.01, 0.5, expr, fontsize=size,
             color="#%02x%02x%02x" % color, ha="left", va="center")
    canvas.draw()
    arr = np.asarray(canvas.buffer_rgba()).copy()
    im = Image.fromarray(arr, mode="RGBA")
    bbox = im.getchannel("A").getbbox()
    im = im.crop(bbox) if bbox else Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    _MATH_CACHE[key] = im
    return im


def mi(expr: str, name: str, color=C_FORMULA, height=0.30, size=27) -> ImageMobject:
    path = FORMULA_DIR / f"mvt_{name}.png"
    if not path.exists():
        _math_img(expr, size, color).save(path)
    m = ImageMobject(str(path))
    if m.height > 0:
        m.scale(height / m.height)
    return m


def text_img(text: str, px: int, color=C_TEXT, height=0.3) -> ImageMobject:
    img = Image.new("RGBA", (1600, 200), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(_SERIF, px)
    d.text((20, 20), text, font=font, fill=(*color, 255))
    bbox = img.getchannel("A").getbbox()
    img = img.crop(bbox) if bbox else img
    m = ImageMobject(np.asarray(img))
    if m.height > 0:
        m.scale(height / m.height)
    return m


def cn_text(text: str, px: int, color=C_TEXT, height=0.3) -> ImageMobject:
    img = Image.new("RGBA", (1800, 220), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", px)
    d.text((20, 16), text, font=font, fill=(*color, 255))
    bbox = img.getchannel("A").getbbox()
    img = img.crop(bbox) if bbox else img
    m = ImageMobject(np.asarray(img))
    if m.height > 0:
        m.scale(height / m.height)
    return m


class PixelNumber(ImageMobject):
    _FONT = str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf")

    def __init__(self, value=0.0, places: int = 3, color=M_TEXT, px: int = 40,
                 nchars: int | None = None, **kwargs):
        self.places = places
        self.value = value
        self._color = tuple(int(c) for c in ManimColor(color).to_rgb() * 255) \
            if not isinstance(color, tuple) else color
        self._font = ImageFont.truetype(self._FONT, px)
        n = nchars if nchars is not None else (1 + 1 + places + (1 if places else 0))
        self._cw, self._ch = int(px * (n * 0.62 + 0.4)), int(px * 1.6)
        super().__init__(np.zeros((self._ch, self._cw, 4), dtype=np.uint8), **kwargs)
        self.set_value(value)

    def set_value(self, value):
        self.value = value
        txt = value if isinstance(value, str) else f"{value:.{self.places}f}"
        img = Image.new("RGBA", (self._cw, self._ch), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        box = d.textbbox((0, 0), txt, font=self._font)
        tw, th = box[2] - box[0], box[3] - box[1]
        d.text(((self._cw - tw) / 2 - box[0], (self._ch - th) / 2 - box[1]),
               txt, font=self._font, fill=(*self._color, 255))
        self.pixel_array[:] = np.asarray(img)


# ---------------------------------------------------------------------------
# 数学设置
#   f(x)=sin x, a=0.28, b=1.46
#   k = (sin b - sin a)/(b-a), f'(x)=cos x
#   xi = arccos(k)，数值落在 (a,b) 内
# ---------------------------------------------------------------------------
A, B = 0.28, 1.46
K_MEAN = (math.sin(B) - math.sin(A)) / (B - A)
XI = math.acos(K_MEAN)

RES_W, RES_H = 1000, 560
BG_RGB = (16, 17, 19)
PIX_FONT = ImageFont.truetype(_SERIF, 17)
PIX_FONT_B = ImageFont.truetype(_SERIF, 20)

LX0, LX1, LY0, LY1 = 56.0, 630.0, 40.0, 505.0
XMIN, XMAX = 0.0, 1.75
YMIN, YMAX = -0.05, 1.12

RX0, RX1, RY0, RY1 = 700.0, 970.0, 70.0, 480.0
RYMIN, RYMAX = -0.10, 1.05

PHASES = [
    ("intro",    0.00,  1.55),
    ("secant",   1.55,  5.20),
    ("search",   5.20, 11.40),
    ("approach",11.40, 16.20),
    ("equal",   16.20, 19.50),
    ("final",   19.50, DURATION),
]


def phase_info(t: float):
    for name, t0, t1 in PHASES:
        if t < t1:
            return name, (0.0 if name == "intro"
                          else float(np.clip((t - t0) / (t1 - t0), 0.0, 1.0)))
    return "final", 1.0


def smooth(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def fval(x: float) -> float:
    return math.sin(x)


def fpval(x: float) -> float:
    return math.cos(x)


def left_px(x: float, y: float):
    return (LX0 + (x - XMIN) / (XMAX - XMIN) * (LX1 - LX0),
            LY1 - (y - YMIN) / (YMAX - YMIN) * (LY1 - LY0))


def right_px(x: float, y: float):
    return (RX0 + (x - A) / (B - A) * (RX1 - RX0),
            RY1 - (y - RYMIN) / (RYMAX - RYMIN) * (RY1 - RY0))


def current_c(t: float) -> float:
    phase, prog = phase_info(t)
    if phase in ("intro", "secant"):
        return A
    if phase == "search":
        return A + (B - A) * prog
    if phase == "approach":
        return B + (XI - B) * smooth(prog)
    return XI


def reveal_state(t: float):
    """返回 (secant_alpha, triangle_alpha, deriv_alpha, tangent_alpha, match_alpha)。"""
    phase, prog = phase_info(t)
    sec = smooth((t - 0.45) / 1.00)
    tri = smooth((t - 2.05) / 1.30) if phase != "intro" else 0.0
    der = smooth((t - 3.10) / 1.50) if phase != "intro" else 0.0
    tan = 0.0
    if phase == "search":
        tan = smooth(prog / 0.16)
    elif phase in ("approach", "equal", "final"):
        tan = 1.0
    match = 0.0
    if phase == "approach":
        match = smooth((prog - 0.72) / 0.28)
    elif phase in ("equal", "final"):
        match = 1.0
    return sec, tri, der, tan, match


def make_base() -> Image.Image:
    img = Image.new("RGBA", (RES_W, RES_H), (*BG_RGB, 255))
    d = ImageDraw.Draw(img, "RGBA")
    # 分栏
    d.line([(664, 22), (664, RES_H - 18)], fill=(*C_GRID_DIM, 150), width=1)
    d.text((30, 16), "GRAPH OF  f(x) = sin x", font=PIX_FONT, fill=(*C_GRID, 255))
    d.text((694, 16), "DERIVATIVE  f'(x) = cos x", font=PIX_FONT, fill=(*C_GRID, 255))

    # 左栏网格
    for i in range(0, 8):
        x = XMAX * i / 7
        p0 = left_px(x, YMIN)
        p1 = left_px(x, YMAX)
        d.line([p0, p1], fill=(*C_GRID_DIM, 190 if i % 2 == 0 else 120), width=1)
    for j in range(0, 6):
        y = -0.05 + 0.23 * j
        if y > YMAX:
            break
        p0 = left_px(XMIN, y)
        p1 = left_px(XMAX, y)
        d.line([p0, p1], fill=(*C_GRID_DIM, 190 if j % 2 == 0 else 120), width=1)
    d.line([left_px(XMIN, 0), left_px(XMAX, 0)], fill=(*C_GRID, 255), width=1)
    d.line([left_px(A, YMIN), left_px(A, YMAX)], fill=(*C_GRID_DIM, 190), width=1)
    d.line([left_px(B, YMIN), left_px(B, YMAX)], fill=(*C_GRID_DIM, 190), width=1)
    d.text((left_px(A, 0)[0] - 5, LY1 + 7), "a", font=PIX_FONT_B, fill=(*C_GRID, 230))
    d.text((left_px(B, 0)[0] - 5, LY1 + 7), "b", font=PIX_FONT_B, fill=(*C_GRID, 230))

    # 右栏网格
    for i in range(0, 5):
        x = A + (B - A) * i / 4
        d.line([right_px(x, RYMIN), right_px(x, RYMAX)],
               fill=(*C_GRID_DIM, 190 if i in (0, 4) else 120), width=1)
    for j in range(0, 6):
        y = -0.1 + 0.23 * j
        if y > RYMAX:
            break
        d.line([right_px(A, y), right_px(B, y)],
               fill=(*C_GRID_DIM, 190 if j == 4 else 120), width=1)
    d.line([right_px(A, 0), right_px(B, 0)], fill=(*C_GRID, 255), width=1)
    d.text((RX0 - 2, RY1 + 7), "a", font=PIX_FONT_B, fill=(*C_GRID, 230))
    d.text((RX1 - 10, RY1 + 7), "b", font=PIX_FONT_B, fill=(*C_GRID, 230))
    return img


BASE_IMG = make_base()


def draw_dashed(d, p0, p1, color, width=1, dash=7.0, gap=5.0, alpha=255):
    """沿线段相对距离画虚线（避免绝对坐标点积造成的死循环）。"""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return
    ux, uy = dx / L, dy / L
    dist = 0.0
    while dist < L:
        e = min(L, dist + dash)
        d.line([(p0[0] + dist * ux, p0[1] + dist * uy),
                (p0[0] + e * ux, p0[1] + e * uy)],
               fill=(*color, alpha), width=width)
        dist = e + gap


def draw_parallel_mark(d, p, color, alpha):
    """两条平行线上的粉色标记（双箭头）。"""
    a = int(alpha)
    for shift in (-5, 5):
        x = p[0] + shift
        d.line([(x - 5, p[1] + 5), (x, p[1] - 5), (x + 5, p[1] + 5)],
               fill=(*color, a), width=2)


def tangent_points(c: float):
    pts = []
    for i in range(121):
        x = c - 0.52 + 1.04 * i / 120
        if not (XMIN <= x <= XMAX):
            continue
        y = fval(c) + fpval(c) * (x - c)
        if not (YMIN <= y <= YMAX):
            continue
        pts.append(left_px(x, y))
    return pts


class MeanValuePlane(ImageMobject):
    def __init__(self, disp_height=3.70):
        super().__init__(np.asarray(BASE_IMG).copy())
        self.scale(disp_height / self.height)

    def redraw(self, t: float):
        arr = self.pixel_array
        arr[:] = np.asarray(BASE_IMG)
        img = Image.fromarray(arr, mode="RGBA")
        d = ImageDraw.Draw(img, "RGBA")
        sec, tri, der, tan, match = reveal_state(t)
        c = current_c(t)

        # 左栏：固定函数曲线
        curve = [left_px(XMAX * i / 240, fval(XMAX * i / 240)) for i in range(241)]
        d.line(curve, fill=(*C_GRID, 255), width=2)

        # 端点 A、B
        pa, pb = left_px(A, fval(A)), left_px(B, fval(B))
        for p in (pa, pb):
            d.ellipse([p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5],
                      fill=(*C_RED, int(230 * sec)), outline=(*C_RED, int(255 * sec)), width=2)
        d.text((pa[0] - 20, pa[1] - 24), "A", font=PIX_FONT_B, fill=(*C_RED, int(255 * sec)))
        d.text((pb[0] + 10, pb[1] - 24), "B", font=PIX_FONT_B, fill=(*C_RED, int(255 * sec)))

        # 割线（红）与升/宽三角形
        if sec > 0.001:
            xa0, xb0 = max(XMIN, A - 0.16), min(XMAX, B + 0.16)
            s0 = left_px(xa0, fval(A) + (xa0 - A) * K_MEAN)
            s1 = left_px(xb0, fval(A) + (xb0 - A) * K_MEAN)
            d.line([s0, s1], fill=(*C_RED, int(245 * sec)), width=2)
        if tri > 0.001:
            corner = left_px(B, fval(A))
            draw_dashed(d, pa, corner, C_RED, width=1, alpha=int(200 * tri))
            draw_dashed(d, corner, pb, C_RED, width=1, alpha=int(200 * tri))
            d.text(((pa[0] + corner[0]) / 2 - 12, corner[1] + 8), "dx",
                   font=PIX_FONT, fill=(*C_RED, int(235 * tri)))
            d.text((corner[0] + 8, (corner[1] + pb[1]) / 2 - 6), "df",
                   font=PIX_FONT, fill=(*C_RED, int(235 * tri)))

        # 当前点 c 与切线（青）
        if tan > 0.001:
            pc = left_px(c, fval(c))
            tpts = tangent_points(c)
            if len(tpts) > 1:
                d.line(tpts, fill=(*C_CYAN, int(250 * tan)), width=3)
            draw_dashed(d, pc, left_px(c, 0), C_CYAN, width=1, alpha=int(170 * tan))
            d.ellipse([pc[0] - 6, pc[1] - 6, pc[0] + 6, pc[1] + 6],
                      fill=(*C_CYAN, int(250 * tan)), outline=(*C_TEXT, int(220 * tan)), width=1)
            d.text((pc[0] + 10, pc[1] - 26), "c", font=PIX_FONT_B,
                   fill=(*C_CYAN, int(245 * tan)))

        # 平行标记：切线与割线各一个
        if match > 0.001:
            tm = left_px(c + 0.24, fval(c) + fpval(c) * 0.24)
            sm = left_px(A + 0.58, fval(A) + K_MEAN * 0.58)
            draw_parallel_mark(d, tm, C_ORANGE, int(235 * match))
            draw_parallel_mark(d, sm, C_ORANGE, int(235 * match))
            if match > 0.6:
                d.text((LX0 + 12, LY0 + 12), "PARALLEL", font=PIX_FONT_B,
                       fill=(*C_ORANGE, int(235 * match)))

        # 右栏：导函数曲线 + 水平线 k
        dcurve = [right_px(A + (B - A) * i / 160, fpval(A + (B - A) * i / 160))
                  for i in range(161)]
        if der > 0.001:
            d.line(dcurve, fill=(*C_GRID, int(255 * der)), width=2)
            pk0, pk1 = right_px(A, K_MEAN), right_px(B, K_MEAN)
            draw_dashed(d, pk0, pk1, C_RED, width=2, dash=10, gap=6, alpha=int(235 * der))
            d.text((pk1[0] + 6, pk1[1] - 9), "k", font=PIX_FONT_B,
                   fill=(*C_RED, int(245 * der)))
            # 交点 xi（红色空心圆）
            pxi = right_px(XI, K_MEAN)
            d.ellipse([pxi[0] - 6, pxi[1] - 6, pxi[0] + 6, pxi[1] + 6],
                      outline=(*C_RED, int(245 * der)), width=2)

        # 右栏游标：当前 c 对应的 f'(c)
        if tan > 0.001:
            q = right_px(c, fpval(c))
            draw_dashed(d, q, right_px(c, 0), C_CYAN, width=1, alpha=int(170 * tan))
            d.ellipse([q[0] - 5, q[1] - 5, q[0] + 5, q[1] + 5],
                      fill=(*C_CYAN, int(250 * tan)), outline=(*C_TEXT, int(220 * tan)), width=1)

        self.pixel_array[:] = np.asarray(img)


class MeanValueTheorem(Scene):
    def construct(self):
        # 标题：双语、整体淡入后常亮、无打字机竖线
        cn_title = cn_text("拉格朗日中值定理", 50, C_TEXT, 0.30).move_to([0, 4.68, 0])
        en_title = text_img("L A G R A N G E   M E A N   V A L U E   T H E O R E M",
                            18, C_CYAN, 0.10).move_to([0, 4.37, 0])
        watermark = text_img("AIMaker-Lab", 17, C_GRID, 0.09).move_to([3.05, 4.68, 0])
        title = Group(cn_title, en_title, watermark)

        leg = Group(
            Group(Line(LEFT * 0.13, RIGHT * 0.13, color=M_RED, stroke_width=3),
                  text_img("SECANT  A-B", 23, C_RED, 0.14)).arrange(RIGHT, buff=0.06),
            Group(Line(LEFT * 0.13, RIGHT * 0.13, color=M_CYAN, stroke_width=3),
                  text_img("TANGENT  AT  c", 23, C_CYAN, 0.14)).arrange(RIGHT, buff=0.06),
            Group(Line(LEFT * 0.13, RIGHT * 0.13, color="#686E72",
                       stroke_width=3),
                  text_img("f'(x)  PANEL", 23, C_GRID, 0.14)).arrange(RIGHT, buff=0.06),
        ).arrange(RIGHT, buff=0.14).move_to([0, 4.02, 0])

        plane = MeanValuePlane(disp_height=3.70).move_to([0, 2.02, 0])

        lbl_k = cn_text("割线斜率 k", 27, C_RED, 0.22).move_to([-2.30, -0.12, 0])
        lbl_fp = cn_text("切线斜率 f'(c)", 27, C_CYAN, 0.22).move_to([0, -0.12, 0])
        lbl_d = cn_text("斜率差 |f'(c)-k|", 27, C_ORANGE, 0.22).move_to([2.35, -0.12, 0])
        num_k = PixelNumber("—", 3, M_RED, 40, nchars=7).set(height=0.30).move_to([-2.30, -0.50, 0])
        num_fp = PixelNumber("—", 3, M_CYAN, 40, nchars=7).set(height=0.30).move_to([0, -0.50, 0])
        num_d = PixelNumber("—", 3, M_ORANGE, 40, nchars=7).set(height=0.30).move_to([2.35, -0.50, 0])

        NARR = [
            ("intro",    "端点 A、B 连成割线：它给出整段区间的平均变化率", C_TEXT),
            ("secant",   "割线斜率是直角三角形的高比宽：k = (f(b)-f(a))/(b-a)", C_RED),
            ("search",   "让切线沿曲线滑动：瞬时斜率 f'(c) 随 c 连续变化", C_CYAN),
            ("approach", "寻找切线与割线平行的那一刻：斜率差正在缩小", C_ORANGE),
            ("equal",    "存在 ξ∈(a,b)，使 f'(ξ) 恰好等于割线斜率", C_CYAN),
            ("final",    "一段的平均变化率，必由区间内某一点的瞬时变化率实现", C_TEXT),
        ]
        narr_pos = [0, -0.95, 0]
        narr_lbls = {}
        for key, s, col in NARR:
            lb = cn_text(s, 26, col, 0.22).move_to(narr_pos)
            lb.move_to([9, 9, 0])
            narr_lbls[key] = lb

        f1 = mi(r"$k=\frac{f(b)-f(a)}{b-a}$", "f1", C_FORMULA, 0.40, 27).move_to([0, -1.48, 0])
        f2 = mi(r"$f(x)=\sin x,\qquad a=0.28,\quad b=1.46$", "f2",
                C_FORMULA, 0.28, 27).move_to([0, -2.08, 0])
        f3 = mi(r"$f'(x)=\cos x$", "f3", C_FORMULA, 0.30, 27).move_to([0, -2.68, 0])
        f4 = mi(r"$f'(\xi)=k\quad\Longrightarrow\quad"
                r"\cos\xi=\frac{\sin b-\sin a}{b-a}$", "f4",
                C_FORMULA, 0.43, 27).move_to([0, -3.32, 0])
        f5 = mi(r"$\exists\,\xi\in(a,b):\quad f(b)-f(a)=f'(\xi)(b-a)$", "f5",
                C_FORMULA, 0.33, 27).move_to([0, -3.96, 0])
        FORMULA_LIST = [f1, f2, f3, f4, f5]
        phase_formula = {"intro": 0, "secant": 0, "search": 2,
                         "approach": 3, "equal": 3, "final": 4}

        self.add(leg, lbl_k, lbl_fp, lbl_d)
        dynamic = Group(title, plane, num_k, num_fp, num_d,
                        *narr_lbls.values(), Group(*FORMULA_LIST))
        self.add(dynamic)

        def apply(t):
            phase, prog = phase_info(t)
            sec, tri, der, tan, match = reveal_state(t)
            c = current_c(t)
            num_k.set_value(f"{K_MEAN:.3f}" if sec > 0.75 else "—")
            num_fp.set_value(f"{fpval(c):.3f}" if tan > 0.2 else "—")
            num_d.set_value(f"{abs(fpval(c) - K_MEAN):.3f}" if tan > 0.2 else "—")
            if phase == "equal" and prog > 0.55:
                num_fp.set_value(f"{K_MEAN:.3f}")
                num_d.set_value("0.000")
            plane.redraw(t)
            for key, lb in narr_lbls.items():
                lb.move_to(narr_pos if key == phase else [9, 9, 0])
            hi = phase_formula[phase]
            for i, fm in enumerate(FORMULA_LIST):
                fm.set_opacity(1.0 if i == hi else 0.48)

        apply(0)
        self.play(FadeIn(title), run_time=0.7)
        self.play(FadeIn(Group(*FORMULA_LIST)), run_time=0.3)

        class Timeline(Animation):
            def __init__(self, mobject, **kw):
                super().__init__(mobject, run_time=DURATION, **kw)

            def interpolate_mobject(self, alpha):
                apply(alpha * DURATION)

        self.play(Timeline(dynamic), rate_func=linear)
        apply(DURATION - 1 / config.frame_rate)
        self.wait(0.08)
