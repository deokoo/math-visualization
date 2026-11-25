#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manim 数学动画：曲率与曲率圆 —— 单位弧长的转角，二阶接触的圆

视觉规范与前作一致：
  - 1080x1350 竖屏、30fps、约 22.3s
  - 公式全程固定，只动参数 / 曲线 / 动画
  - 无 LaTeX：公式用 matplotlib mathtext 预渲染为透明 PNG
  - 暗色底 #101113、中灰清晰线条、青/红边缘曲线、纯内存动态数字

画面：
  中央 = y = 0.7 sin x 的图像
         ① 两点切线夹角 Δα / 弧长 Δs，Δs 收缩 -> 曲率定义
         ② 红点沿曲线滑行，切线方向随 y' 转动，曲线按 κ 大小染色
         ③ 橙色半径连出曲率圆的圆心 C(a,b)，圆随 κ 变大而收紧
         ④ 同时画出峰顶小圆与平缓处大圆，对比"越弯圆越小"
         ⑤ 最后一行公式点亮，二阶接触定格
  底部 = 5 行固定公式链；中文讲述行随阶段切换

运行：
    manim -pql preview_scene.py CurvatureCircle     # 半分辨率预览
    manim -pqh curvature_scene.py CurvatureCircle   # 1080×1350 成片
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
M_GRID = "#767C80"
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
    path = FORMULA_DIR / f"cv_{name}.png"
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
#   f(x)  = 1.2 sin x
#   f'(x) = 1.2 cos x,  f''(x) = -1.2 sin x
#   κ     = |f''| / (1 + f'^2)^(3/2)
#   R     = 1/κ
#   C     = (x - f'(1+f'^2)/f'',  f + (1+f'^2)/f'')
#   峰顶 x=π/2: κ=1.200, R=0.833；两翼 x=0.85: κ≈0.435, R≈2.30
# ---------------------------------------------------------------------------
def f(x: float) -> float:
    return 1.2 * math.sin(x)


def d1(x: float) -> float:
    return 1.2 * math.cos(x)


def d2(x: float) -> float:
    return -1.2 * math.sin(x)


def kappa(x: float) -> float:
    return abs(d2(x)) / (1.0 + d1(x) ** 2) ** 1.5


def radius(x: float) -> float:
    return 1.0 / kappa(x)


def center(x: float):
    q = 1.0 + d1(x) ** 2
    return (x - d1(x) * q / d2(x), f(x) + q / d2(x))


X_MIN, X_MAX = -3.05, 3.05
Y_MIN, Y_MAX = -3.35, 1.55
# 两轴像素比例保持一致（≈124.6 px / 单位），曲率圆不会被拉成椭圆
RES_W, RES_H = 760, 610
BG_RGB = (16, 17, 19)
PIX_FONT = ImageFont.truetype(_SERIF, 19)


def px(x: float) -> float:
    return (x - X_MIN) / (X_MAX - X_MIN) * RES_W


def py(y: float) -> float:
    return (Y_MAX - y) / (Y_MAX - Y_MIN) * RES_H


def pp(x: float, y: float):
    return (px(x), py(y))


def make_base() -> Image.Image:
    img = Image.new("RGBA", (RES_W, RES_H), (*BG_RGB, 255))
    d = ImageDraw.Draw(img, "RGBA")
    for gx in range(-3, 4):
        col = (*C_GRID_DIM, 150) if gx else (*C_GRID_DIM, 220)
        d.line([pp(gx, Y_MIN), pp(gx, Y_MAX)], fill=col, width=1)
    for gy in (-3, -2, -1, 0, 1):
        col = (*C_GRID_DIM, 150) if gy else (*C_GRID_DIM, 220)
        d.line([pp(X_MIN, gy), pp(X_MAX, gy)], fill=col, width=1)
    p = [pp(x, f(x)) for x in np.linspace(X_MIN, X_MAX, 421)]
    d.line(p, fill=(*C_GRID_DIM, 235), width=2, joint="curve")
    return img


BASE_IMG = make_base()


def arrow(d, p0, p1, color, width=2, head=7):
    d.line([p0, p1], fill=color, width=width)
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    for da in (math.pi - 0.42, math.pi + 0.42):
        d.line([p1, (p1[0] + head * math.cos(ang + da),
                     p1[1] + head * math.sin(ang + da))], fill=color, width=width)


PHASES = [
    ("intro",   0.00,  1.30),
    ("limit",   1.30,  5.20),
    ("sweep",   5.20, 10.00),
    ("circle", 10.00, 15.40),
    ("compare", 15.40, 18.60),
    ("final",   18.60, 19.90),
    ("replay",  19.90, DURATION),
]


def phase_info(t: float):
    for name, t0, t1 in PHASES:
        if t < t1:
            return name, (0.0 if name == "intro"
                          else float(np.clip((t - t0) / (t1 - t0), 0.0, 1.0)))
    return "replay", 1.0


def sweep_x(t: float) -> float:
    u = float(np.clip((t - 5.20) / (15.40 - 5.20), 0.0, 1.0))
    return 2.25 - 1.40 * u


class CurveField(ImageMobject):
    def __init__(self, disp_height=3.50):
        super().__init__(np.asarray(BASE_IMG).copy())
        self.scale(disp_height / self.height)

    def _circle(self, d, x, col, width=2, alpha=235, label=None):
        a, b = center(x)
        r = radius(x)
        box = [px(a - r), py(b + r), px(a + r), py(b - r)]
        d.ellipse(box, outline=(*col, alpha), width=width)
        d.line([pp(x, f(x)), pp(a, b)], fill=(*C_ORANGE, alpha), width=2)
        d.ellipse([px(a) - 4, py(b) - 4, px(a) + 4, py(b) + 4],
                  fill=(*col, alpha))
        if label:
            lx = px(a) - 70 if a > 1.6 else px(a + 0.10 * r)
            d.text((lx, py(b) - 9), label,
                   font=PIX_FONT, fill=(*col, alpha))

    def redraw(self, t: float):
        phase, prog = phase_info(t)
        arr = self.pixel_array
        arr[:] = np.asarray(BASE_IMG)
        img = Image.fromarray(arr, mode="RGBA")
        d = ImageDraw.Draw(img, "RGBA")

        # 按 κ 大小着色的曲线，随阶段逐渐向右扫出
        if phase == "intro":
            reveal = X_MIN + (X_MAX - X_MIN) * prog
        elif phase == "limit":
            reveal = 2.90
        elif phase in ("sweep", "circle"):
            reveal = 2.90
        else:
            reveal = X_MAX
        xs = np.linspace(X_MIN, min(reveal, X_MAX), 400)
        steps = max(1, len(xs) - 1)
        for i in range(steps):
            xa, xb = xs[i], xs[i + 1]
            k = float(np.clip(kappa((xa + xb) / 2) / 1.20, 0, 1))
            col = (int(C_GRID[0] + (C_CYAN[0] - C_GRID[0]) * k),
                   int(C_GRID[1] + (C_CYAN[1] - C_GRID[1]) * k),
                   int(C_GRID[2] + (C_CYAN[2] - C_GRID[2]) * k), 255)
            d.line([pp(xa, f(xa)), pp(xb, f(xb))], fill=col, width=3)

        if phase == "limit":
            # 两点切线：Δx 收缩，Δα / Δs 的可视化
            x0 = 2.05
            dx = 0.70 - 0.61 * (prog ** 0.85)
            x1 = x0 + dx
            for x, col in ((x0, C_RED), (x1, C_ORANGE)):
                sl = d1(x)
                p0 = pp(x - 0.62, f(x) - 0.62 * sl)
                p1 = pp(x + 0.62, f(x) + 0.62 * sl)
                arrow(d, p0, p1, (*col, 240), 2, 7)
                d.ellipse([px(x) - 5, py(f(x)) - 5, px(x) + 5, py(f(x)) + 5],
                          fill=(*col, 255))
            arc = [pp(x0 + 0.30 * math.cos(a), f(x0) + 0.30 * math.sin(a))
                   for a in np.linspace(math.atan(d1(x1)), math.atan(d1(x0)), 30)]
            d.line(arc, fill=(*C_ORANGE, 235), width=3)
            seg = [pp(x, f(x)) for x in np.linspace(x0, x1, 40)]
            d.line(seg, fill=(*C_ORANGE, 255), width=5, joint="curve")
            d.text((px(x1) + 10, py(f(x1)) - 34), "Δα", font=PIX_FONT,
                   fill=(*C_ORANGE, 255))
            d.text((px((x0 + x1) / 2) - 10, py(f((x0 + x1) / 2)) + 10),
                   "Δs", font=PIX_FONT, fill=(*C_RED, 255))

        elif phase in ("sweep", "circle"):
            x = sweep_x(t)
            sl = d1(x)
            p0 = pp(x - 0.72, f(x) - 0.72 * sl)
            p1 = pp(x + 0.72, f(x) + 0.72 * sl)
            arrow(d, p0, p1, (*C_RED, 245), 2, 7)
            if phase == "circle":
                a = float(np.clip((prog - 0.05) / 0.25, 0.0, 1.0))
                self._circle(d, x, C_CYAN, 2, int(235 * a))
                axx, byy = center(x)
                d.text((px(axx) + 10, py(byy) + 8), "C", font=PIX_FONT,
                       fill=(*C_CYAN, int(255 * a)))
            d.ellipse([px(x) - 6, py(f(x)) - 6, px(x) + 6, py(f(x)) + 6],
                      fill=(*C_RED, 255), outline=(*C_TEXT, 220), width=1)

        elif phase in ("compare", "final", "replay"):
            xa, xb = math.pi / 2, 0.85
            self._circle(d, xa, C_CYAN, 2, 245, f"R={radius(xa):.2f}")
            self._circle(d, xb, C_ORANGE, 2, 245, f"R={radius(xb):.2f}")
            for x, col in ((xa, C_CYAN), (xb, C_ORANGE)):
                d.ellipse([px(x) - 6, py(f(x)) - 6, px(x) + 6, py(f(x)) + 6],
                          fill=(*C_RED, 255), outline=(*col, 255), width=1)
            d.text((px(-2.85), py(-2.20)), "same tangent, higher order contact",
                   font=PIX_FONT, fill=(*C_GRID, 255))

        self.pixel_array[:] = np.asarray(img)


class CurvatureCircle(Scene):
    def construct(self):
        cn_title = cn_text("曲率与曲率圆", 50, C_TEXT, 0.30).move_to([0, 4.68, 0])
        en_title = text_img("C U R V A T U R E   /   O S C U L A T I N G   C I R C L E",
                            19, C_CYAN, 0.11).move_to([0, 4.37, 0])
        watermark = text_img("AIMaker-Lab", 17, C_GRID, 0.09).move_to([3.05, 4.68, 0])
        title = Group(cn_title, en_title, watermark)
        cursor = Line(UP * 0.16, DOWN * 0.16, color=M_TEXT, stroke_width=2.5).move_to([0, 4.68, 0])
        cursor.set_opacity(0)

        leg = Group(
            Group(Line(LEFT * 0.13, RIGHT * 0.13, color=M_CYAN, stroke_width=3),
                  text_img("CURVATURE κ", 24, C_CYAN, 0.14)).arrange(RIGHT, buff=0.06),
            Group(Line(LEFT * 0.13, RIGHT * 0.13, color=M_RED, stroke_width=3),
                  text_img("TANGENT / POINT", 24, C_RED, 0.14)).arrange(RIGHT, buff=0.06),
            Group(Line(LEFT * 0.13, RIGHT * 0.13, color=M_ORANGE, stroke_width=3),
                  text_img("RADIUS / ANGLE", 24, C_ORANGE, 0.14)).arrange(RIGHT, buff=0.06),
        ).arrange(RIGHT, buff=0.18).move_to([0, 4.02, 0])

        plane = CurveField(disp_height=3.50).move_to([0, 1.75, 0])

        lbl_k = cn_text("曲率 κ", 30, C_CYAN, 0.21).move_to([-2.30, -0.22, 0])
        lbl_r = cn_text("曲率半径 R", 30, C_ORANGE, 0.21).move_to([2.30, -0.22, 0])
        num_k = PixelNumber(0.0, 3, M_CYAN, 40, nchars=6).set(height=0.30).move_to([-2.30, -0.66, 0])
        num_r = PixelNumber(0.0, 3, M_ORANGE, 40, nchars=6).set(height=0.30).move_to([2.30, -0.66, 0])

        NARR = [
            ("intro",   "一条曲线到底弯得多快？曲率给出答案", C_TEXT),
            ("limit",   "两点切线夹角 Δα 除以弧长 Δs：令 Δs 趋零", C_ORANGE),
            ("sweep",   "沿曲线滑行：越弯的地方青色越亮，κ 越大", C_CYAN),
            ("circle",  "R = 1/κ：橙色半径锁定最贴合的曲率圆", C_ORANGE),
            ("compare", "峰顶圆紧、两翼圆松：同切线，还是二阶接触", C_TEXT),
            ("final",   "圆心由 y' 与 y'' 精确锁定：曲率圆诞生", C_CYAN),
            ("replay",  "局部看曲线，二阶接触的就是曲率圆", C_TEXT),
        ]
        narr_pos = [0, -1.30, 0]
        narr_lbls = {}
        for key, s, col in NARR:
            lb = cn_text(s, 27, col, 0.23).move_to(narr_pos)
            lb.move_to([9, 9, 0])
            narr_lbls[key] = lb

        f1 = mi(r"$\kappa=\lim_{\Delta s\to 0}\left|\frac{\Delta\alpha}{\Delta s}\right|=\left|\frac{d\alpha}{ds}\right|$",
                "f1", C_FORMULA, 0.44, 27).move_to([0, -2.00, 0])
        f2 = mi(r"$y=f(x):\quad\kappa=\frac{|y''|}{(1+y'^2)^{3/2}}$",
                "f2", C_FORMULA, 0.42, 27).move_to([0, -2.66, 0])
        f3 = mi(r"$R=\frac{1}{\kappa}=\frac{(1+y'^2)^{3/2}}{|y''|}$",
                "f3", C_FORMULA, 0.42, 27).move_to([0, -3.28, 0])
        f4 = mi(r"$C=(a,b)=\left(x-\frac{y'(1+y'^2)}{y''},\ y+\frac{1+y'^2}{y''}\right)$",
                "f4", C_FORMULA, 0.45, 26).move_to([0, -3.92, 0])
        f5 = mi(r"$(x-a)^2+(y-b)^2=R^2\quad\Rightarrow\quad\mathrm{second\ order\ contact}$",
                "f5", C_FORMULA, 0.31, 26).move_to([0, -4.52, 0])
        formulas = Group(f1, f2, f3, f4, f5)
        FORMULA_COLORS = [f1, f2, f3, f4, f5]
        phase_formula = {"intro": 0, "limit": 0, "sweep": 1, "circle": 2,
                         "compare": 3, "final": 4, "replay": 4}

        self.add(leg, lbl_k, lbl_r)
        dynamic = Group(title, cursor, plane, num_k, num_r, *narr_lbls.values(), formulas)
        self.add(dynamic)

        def apply(t):
            phase, prog = phase_info(t)
            plane.redraw(t)

            if phase == "limit":
                x = 2.05
            elif phase in ("sweep", "circle"):
                x = sweep_x(t)
            else:
                x = math.pi / 2
            kv, rv = kappa(x), radius(x)
            if phase == "intro":
                kv, rv = 0.0, 0.0
            num_k.set_value(kv)
            num_r.set_value(rv)

            for k, lb in narr_lbls.items():
                lb.move_to(narr_pos if k == phase else [9, 9, 0])

            hi = phase_formula[phase]
            for i, fm in enumerate(FORMULA_COLORS):
                fm.set_opacity(1.0 if i == hi else 0.48)

            blink = (t < 0.28) or (17.62 <= t <= 17.96)
            title.set_opacity(1)
            cursor.set_opacity(0)

        apply(0)
        self.play(FadeIn(formulas), run_time=0.7)
        self.play(FadeIn(title), run_time=0.3)

        class Timeline(Animation):
            def __init__(self, mobject, **kw):
                super().__init__(mobject, run_time=DURATION, **kw)

            def interpolate_mobject(self, alpha):
                apply(alpha * DURATION)

        self.play(Timeline(dynamic), rate_func=linear)
        apply(DURATION - 1 / config.frame_rate)
        self.wait(0.08)
