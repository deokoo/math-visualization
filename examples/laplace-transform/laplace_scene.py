#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manim 数学动画：拉普拉斯变换 —— 把时间函数变成复频域函数

视觉规范与前作一致，但**不显示标题打字机竖线**：
  - 1080x1350 竖屏、30fps、约 22.3s
  - 公式全程固定，只动参数 / 面积 / 动画
  - 无 LaTeX：公式用 matplotlib mathtext 预渲染为透明 PNG
  - 暗色底 #101113、中灰清晰线条、青/红边缘曲线、纯内存动态数字

画面：
  左侧 = 时间域：f(t)=e^{-at}（红）、核 e^{-st}（青）、乘积下的面积（橙）
         ① 乘上衰减核，η 从 0 增到 1（核从 1 平滑变成 e^{-st}）
         ② 积分上限 T 从 0 扫到 6，橙色面积累积，F(T) 逼近 1/(s+a)
  右侧 = s 平面：极点 × 在 s=-a，ROC 阴影 s>-a，当前 s 橙点
         ③ s 穿过极点：收敛 / 发散切换
  底部 = 5 行固定公式链；中文讲述行随阶段切换

运行：
    manim -pql preview_scene.py LaplaceTransform   # 半分辨率预览
    manim -pqh laplace_scene.py LaplaceTransform   # 1080×1350 成片
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
    path = FORMULA_DIR / f"lp_{name}.png"
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
        txt = value if isinstance(value, str) else f"{abs(value):.{self.places}f}"
        img = Image.new("RGBA", (self._cw, self._ch), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        box = d.textbbox((0, 0), txt, font=self._font)
        tw, th = box[2] - box[0], box[3] - box[1]
        d.text(((self._cw - tw) / 2 - box[0], (self._ch - th) / 2 - box[1]),
               txt, font=self._font, fill=(*self._color, 255))
        self.pixel_array[:] = np.asarray(img)


# ---------------------------------------------------------------------------
# 数学设置
#   f(t) = e^{-a t},  a = 0.8
#   核   = e^{-s t}  (s 取实数 σ)
#   乘积 = e^{-(s+a)t}
#   F_T  = ∫_0^T e^{-(s+a)t} dt = (1 - e^{-(s+a)T}) / (s+a)
#   s+a>0 时 T→∞ 得 F(s)=1/(s+a)，ROC: Re(s)>-a，极点 s=-a
# ---------------------------------------------------------------------------
A = 0.8
T_MAX = 6.0
SIGMA_HI = 1.2
SIGMA_LO = -1.45

RES_W, RES_H = 900, 430
BG_RGB = (16, 17, 19)
PIX_FONT = ImageFont.truetype(_SERIF, 18)

# 左时间域面板 / 右 s 平面面板（像素坐标）
TL, TR, TB, TT = 48, 520, 382, 52
SL, SR, SCY = 566, 872, 232
S_MIN, S_MAX = -2.4, 2.4


def tx(t: float) -> float:
    return TL + t / T_MAX * (TR - TL)


def ty(y: float) -> float:
    return TB - y / 1.15 * (TB - TT)


def sx(s: float) -> float:
    return SL + (s - S_MIN) / (S_MAX - S_MIN) * (SR - SL)


def fval(t: float) -> float:
    return math.exp(-A * t)


def partial(sigma: float, T: float) -> float:
    q = sigma + A
    if q <= 1e-9:
        return math.inf
    return (1.0 - math.exp(-q * T)) / q


def sigma_at(t: float) -> float:
    phase, prog = phase_info(t)
    if phase in ("intro", "kernel", "integrate"):
        return SIGMA_HI
    if phase == "roc":
        # 穿过极点后在发散区停留一段时间，再回到收敛区
        if prog < 0.35:
            return SIGMA_HI + (SIGMA_LO - SIGMA_HI) * (prog / 0.35)
        if prog < 0.65:
            return SIGMA_LO
        return SIGMA_LO + (SIGMA_HI - SIGMA_LO) * ((prog - 0.65) / 0.35)
    return SIGMA_HI


PHASES = [
    ("intro",     0.00,  1.30),
    ("kernel",    1.30,  5.20),
    ("integrate", 5.20, 10.60),
    ("roc",      10.60, 16.20),
    ("pairs",    16.20, 18.60),
    ("final",    18.60, 19.90),
    ("replay",   19.90, DURATION),
]


def phase_info(t: float):
    for name, t0, t1 in PHASES:
        if t < t1:
            return name, (0.0 if name == "intro"
                          else float(np.clip((t - t0) / (t1 - t0), 0.0, 1.0)))
    return "replay", 1.0


def make_base() -> Image.Image:
    img = Image.new("RGBA", (RES_W, RES_H), (*BG_RGB, 255))
    d = ImageDraw.Draw(img, "RGBA")
    # 左：时间域网格
    for t in range(0, 7):
        d.line([(tx(t), TT), (tx(t), TB)], fill=(*C_GRID_DIM, 150), width=1)
        d.text((tx(t) - 3, TB + 4), str(t), font=PIX_FONT, fill=(*C_GRID_DIM, 255))
    for y in (0.0, 0.5, 1.0):
        d.line([(TL, ty(y)), (TR, ty(y))], fill=(*C_GRID_DIM, 150), width=1)
    d.line([(TL, TB), (TR, TB)], fill=(*C_GRID, 220), width=1)
    d.line([(TL, TT), (TL, TB)], fill=(*C_GRID, 220), width=1)
    d.text((TL + 4, TT + 4), "TIME DOMAIN  f(t), e^(-st), product",
           font=PIX_FONT, fill=(*C_GRID, 255))
    # 右：s 平面（实轴）
    d.line([(SL, SCY), (SR, SCY)], fill=(*C_GRID, 220), width=1)
    d.line([(sx(0), SCY - 56), (sx(0), SCY + 56)], fill=(*C_GRID_DIM, 180), width=1)
    d.text((SL + 4, SCY - 96), "s-PLANE  /  CONVERGENCE",
           font=PIX_FONT, fill=(*C_GRID, 255))
    for s in (-2, -1, 0, 1, 2):
        d.line([(sx(s), SCY - 4), (sx(s), SCY + 4)], fill=(*C_GRID_DIM, 220), width=1)
        d.text((sx(s) - 4, SCY + 8), str(s), font=PIX_FONT, fill=(*C_GRID_DIM, 255))
    # 分栏线
    d.line([(542, TT), (542, TB)], fill=(*C_GRID_DIM, 120), width=1)
    return img


BASE_IMG = make_base()


class LaplaceField(ImageMobject):
    def __init__(self, disp_height=3.50):
        super().__init__(np.asarray(BASE_IMG).copy())
        self.scale(disp_height / self.height)

    def redraw(self, t: float):
        phase, prog = phase_info(t)
        arr = self.pixel_array
        arr[:] = np.asarray(BASE_IMG)
        img = Image.fromarray(arr, mode="RGBA")
        d = ImageDraw.Draw(img, "RGBA")
        sigma = sigma_at(t)
        q = sigma + A
        eta = 0.0
        if phase == "kernel":
            eta = float(np.clip(prog * 1.25, 0, 1))
        elif phase in ("integrate", "roc", "pairs", "final", "replay"):
            eta = 1.0

        # 左：f(t) 红、核青（由 1 平滑过渡到 e^{-st}）、乘积橙
        pts_f = [(tx(x), ty(fval(x))) for x in np.linspace(0, T_MAX, 240)]
        d.line(pts_f, fill=(*C_RED, 245), width=3, joint="curve")
        kern = [1.0 + (math.exp(-sigma * x) - 1.0) * eta
                for x in np.linspace(0, T_MAX, 240)]
        pts_k = [(tx(x), ty(min(1.15, max(0, k))))
                 for x, k in zip(np.linspace(0, T_MAX, 240), kern)]
        d.line(pts_k, fill=(*C_CYAN, 235), width=3, joint="curve")

        Tscan = T_MAX
        if phase == "integrate":
            Tscan = 0.35 + (T_MAX - 0.35) * prog
        gx = np.linspace(0, Tscan, max(2, int(240 * Tscan / T_MAX)))
        gv = [min(1.15, fval(x) * k) for x, k in
              zip(gx, [1.0 + (math.exp(-sigma * x) - 1.0) * eta for x in gx])]
        if phase in ("integrate", "roc", "pairs", "final", "replay"):
            poly = [(tx(0), ty(0))] + [(tx(x), ty(y)) for x, y in zip(gx, gv)] \
                   + [(tx(Tscan), ty(0))]
            d.polygon(poly, fill=(*C_ORANGE, 145))
            d.line([(tx(x), ty(y)) for x, y in zip(gx, gv)],
                   fill=(*C_ORANGE, 255), width=3, joint="curve")
        if phase == "integrate":
            d.line([(tx(Tscan), TB), (tx(Tscan), ty(gv[-1]))],
                   fill=(*C_ORANGE, 240), width=2)
            d.ellipse([tx(Tscan) - 4, ty(gv[-1]) - 4,
                       tx(Tscan) + 4, ty(gv[-1]) + 4], fill=(*C_ORANGE, 255))

        # 右：ROC 阴影 + 极点 + 当前 s
        pole = -A
        rect = [sx(pole), SCY - 40, SR, SCY + 40]
        d.rectangle(rect, fill=(*C_CYAN, 42))
        d.line([(sx(pole), SCY - 44), (sx(pole), SCY + 44)],
               fill=(*C_CYAN, 190), width=2)
        px_, py_ = sx(pole), SCY
        d.line([(px_ - 7, py_ - 7), (px_ + 7, py_ + 7)], fill=(*C_RED, 255), width=3)
        d.line([(px_ - 7, py_ + 7), (px_ + 7, py_ - 7)], fill=(*C_RED, 255), width=3)
        d.text((px_ - 14, SCY + 50), "s = -a", font=PIX_FONT, fill=(*C_RED, 255))
        d.text((sx(0.15), SCY - 36), "ROC: Re(s) > -a",
               font=PIX_FONT, fill=(*C_CYAN, 255))
        cs = sx(float(np.clip(sigma, S_MIN, S_MAX)))
        dot_col = C_ORANGE if q > 0 else C_RED
        d.ellipse([cs - 6, SCY - 6, cs + 6, SCY + 6], fill=(*dot_col, 255))
        d.line([(cs, SCY - 52), (cs, SCY - 12)], fill=(*dot_col, 220), width=2)
        d.text((cs - 12, SCY - 76), f"s={sigma:5.2f}", font=PIX_FONT,
               fill=(*dot_col, 255))
        if q <= 0 and phase == "roc":
            d.text((SL + 8, SCY + 72), "DIVERGES: s <= -a",
                   font=PIX_FONT, fill=(*C_RED, 255))

        # 图例文字
        if phase in ("pairs", "final", "replay"):
            lx = TL + 236  # 放到时间域右上空白区，避开曲线
            d.text((lx, TT + 26), "e^(-at)  <->  1/(s+a)",
                   font=PIX_FONT, fill=(*C_ORANGE, 245))
            d.text((lx, TT + 50), "1       <->  1/s",
                   font=PIX_FONT, fill=(*C_CYAN, 220))
            d.text((lx, TT + 74), "sin(wt)  <->  w/(s^2+w^2)",
                   font=PIX_FONT, fill=(*C_GRID, 235))

        self.pixel_array[:] = np.asarray(img)


class LaplaceTransform(Scene):
    def construct(self):
        # 标题：无打字机竖线，直接整体淡入
        cn_title = cn_text("拉普拉斯变换", 50, C_TEXT, 0.30).move_to([0, 4.68, 0])
        en_title = text_img("L A P L A C E   T R A N S F O R M   /   T I M E   T O   F R E Q U E N C Y",
                            19, C_CYAN, 0.11).move_to([0, 4.37, 0])
        watermark = text_img("AIMaker-Lab", 17, C_GRID, 0.09).move_to([3.05, 4.68, 0])
        title = Group(cn_title, en_title, watermark)

        leg = Group(
            Group(Line(LEFT * 0.13, RIGHT * 0.13, color=M_RED, stroke_width=3),
                  text_img("f(t)=e^(-at)", 24, C_RED, 0.14)).arrange(RIGHT, buff=0.06),
            Group(Line(LEFT * 0.13, RIGHT * 0.13, color=M_CYAN, stroke_width=3),
                  text_img("KERNEL e^(-st)", 24, C_CYAN, 0.14)).arrange(RIGHT, buff=0.06),
            Group(Line(LEFT * 0.13, RIGHT * 0.13, color=M_ORANGE, stroke_width=3),
                  text_img("ACCUMULATED AREA", 24, C_ORANGE, 0.14)).arrange(RIGHT, buff=0.06),
        ).arrange(RIGHT, buff=0.18).move_to([0, 4.02, 0])

        plane = LaplaceField(disp_height=3.50).move_to([0, 1.75, 0])

        lbl_v = cn_text("累积面积 F(T)", 30, C_ORANGE, 0.21).move_to([-2.30, -0.22, 0])
        lbl_t = cn_text("收敛目标 1/(s+a)", 30, C_CYAN, 0.21).move_to([2.30, -0.22, 0])
        num_v = PixelNumber(0.0, 3, M_ORANGE, 40, nchars=6).set(height=0.30).move_to([-2.30, -0.66, 0])
        num_t = PixelNumber(0.0, 3, M_CYAN, 40, nchars=6).set(height=0.30).move_to([2.30, -0.66, 0])

        NARR = [
            ("intro",     "时间域的函数，如何变成复频域的函数？", C_TEXT),
            ("kernel",    "先乘衰减核 e^(-st)：s 越大，衰减越快", C_CYAN),
            ("integrate", "从 0 到 T 累加面积：这就是拉普拉斯积分", C_ORANGE),
            ("roc",       "s 穿过极点：右边收敛，左边发散", C_RED),
            ("pairs",     "记住三个变换对：指数、常数与正弦", C_ORANGE),
            ("final",     "F(s)=1/(s+a)，只在收敛域内成立", C_CYAN),
            ("replay",    "乘核、积分、看收敛域：拉普拉斯变换三步", C_TEXT),
        ]
        narr_pos = [0, -1.30, 0]
        narr_lbls = {}
        for key, s, col in NARR:
            lb = cn_text(s, 27, col, 0.23).move_to(narr_pos)
            lb.move_to([9, 9, 0])
            narr_lbls[key] = lb

        f1 = mi(r"$F(s)=\mathcal{L}\{f(t)\}=\int_0^{\infty}f(t)e^{-st}\,dt$",
                "f1", C_FORMULA, 0.43, 27).move_to([0, -2.00, 0])
        f2 = mi(r"$f(t)=e^{-at}\ \Rightarrow\ f(t)e^{-st}=e^{-(s+a)t}$",
                "f2", C_FORMULA, 0.28, 27).move_to([0, -2.64, 0])
        f3 = mi(r"$\int_0^{T}e^{-(s+a)t}\,dt=\frac{1-e^{-(s+a)T}}{s+a}$",
                "f3", C_FORMULA, 0.44, 27).move_to([0, -3.28, 0])
        f4 = mi(r"$T\to\infty,\ s+a>0:\quad F(s)=\frac{1}{s+a}$",
                "f4", C_FORMULA, 0.30, 27).move_to([0, -3.92, 0])
        f5 = mi(r"$\mathrm{ROC}:\ \mathrm{Re}(s)>-a\quad(\mathrm{pole}\ s=-a)$",
                "f5", C_FORMULA, 0.25, 27).move_to([0, -4.52, 0])
        formulas = Group(f1, f2, f3, f4, f5)
        FORMULA_COLORS = [f1, f2, f3, f4, f5]
        phase_formula = {"intro": 0, "kernel": 1, "integrate": 2, "roc": 4,
                         "pairs": 3, "final": 3, "replay": 4}

        self.add(leg, lbl_v, lbl_t)
        dynamic = Group(title, plane, num_v, num_t, *narr_lbls.values(), formulas)
        self.add(dynamic)

        def apply(t):
            phase, prog = phase_info(t)
            plane.redraw(t)
            sigma = sigma_at(t)
            Tscan = 0.35 + (T_MAX - 0.35) * prog if phase == "integrate" else T_MAX
            if phase in ("intro", "kernel"):
                val = 0.0
                target = 1.0 / (sigma + A)
            elif sigma + A <= 0:
                val = "∞"
                target = "∞"
            else:
                val = partial(sigma, Tscan)
                target = 1.0 / (sigma + A)
            num_v.set_value(val)
            num_t.set_value(target)

            for k, lb in narr_lbls.items():
                lb.move_to(narr_pos if k == phase else [9, 9, 0])

            hi = phase_formula[phase]
            for i, fm in enumerate(FORMULA_COLORS):
                fm.set_opacity(1.0 if i == hi else 0.48)

        apply(0)
        self.play(FadeIn(title), run_time=0.7)
        self.play(FadeIn(formulas), run_time=0.3)

        class Timeline(Animation):
            def __init__(self, mobject, **kw):
                super().__init__(mobject, run_time=DURATION, **kw)

            def interpolate_mobject(self, alpha):
                apply(alpha * DURATION)

        self.play(Timeline(dynamic), rate_func=linear)
        apply(DURATION - 1 / config.frame_rate)
        self.wait(0.08)
