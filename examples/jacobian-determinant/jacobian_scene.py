#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manim 数学动画：雅可比行列式与换元 —— 一张网格被弯曲时，面积怎样缩放？

与前作同风格：
  - 1080x1350 竖屏、30fps、约 22.3s
  - 公式全程固定，只动参数 / 网格 / 微元 / 数值
  - 无 LaTeX：公式用 matplotlib mathtext 预渲染为透明 PNG
  - 暗色底 #101113、中灰清晰线条、青/红边缘曲线、纯内存动态数字
  - 标题双语，整体淡入后常亮，无打字机竖线

画面：
  左 = (u,v) 参数平面的矩形网格与高亮小矩形；
  右 = 同一个网格在 T(u,v)=(u cos v, u sin v) 下的像（极坐标网格），
       高亮小矩形映成曲边四边形，两条偏导边向量（青/红）张成线性近似平行四边形；
  下方 = 两个动态读数「像面积 / 原面积」「雅可比 |J| = r」始终相等；
  底部 = 五行固定公式链 + 随阶段切换的中文讲述行。

运行：
    python precheck.py
    manim -pql preview_scene.py JacobianChange
    python check_frames.py
    manim -pqh jacobian_scene.py JacobianChange
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
    path = FORMULA_DIR / f"jac_{name}.png"
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
#   T_s(u,v) = ((1-s)u + s u cos v,  s u sin v)
#   s=0 恒等映射，s=1 极坐标换元；J_s = s^2 u + s(1-s) u cos v
#   s=1 时 J = u = r，故像面积 = u0 * Δu * Δv（精确）
# ---------------------------------------------------------------------------
RES_W, RES_H = 1000, 560
BG_RGB = (16, 17, 19)
PIX_FONT = ImageFont.truetype(_SERIF, 17)
PIX_FONT_B = ImageFont.truetype(_SERIF, 19)

UL, UR = 0.0, 2.60
VMAX = 2 * math.pi
LEFT_X0, LEFT_XS = 58.0, 146.0
LEFT_Y0, LEFT_YS = 507.0, 72.4
RCX, RCY, RSCALE = 758.0, 283.0, 82.0

PHASES = [
    ("intro",  0.00,  1.50),
    ("morph",  1.50,  5.20),
    ("patch",  5.20, 10.00),
    ("jac",   10.00, 15.00),
    ("shrink", 15.00, 18.80),
    ("final", 18.80, 20.10),
    ("replay", 20.10, DURATION),
]


def phase_info(t: float):
    for name, t0, t1 in PHASES:
        if t < t1:
            return name, (0.0 if name == "intro"
                          else float(np.clip((t - t0) / (t1 - t0), 0.0, 1.0)))
    return "replay", 1.0


def smooth(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def morph_s(t: float) -> float:
    phase, prog = phase_info(t)
    if phase == "intro":
        return 0.0
    if phase == "morph":
        return smooth(prog)
    return 1.0


def patch_params(t: float):
    """返回 (u0, v0, du, dv, reveal)。"""
    phase, prog = phase_info(t)
    u0, v0, du, dv = 0.90, 0.82, 0.44, 0.62
    reveal = 0.0
    if phase == "patch":
        reveal = smooth((prog - 0.12) / 0.48)
    elif phase in ("jac", "shrink", "final", "replay"):
        reveal = 1.0
    if phase == "jac":
        u0 = 0.72 + 1.00 * smooth(prog)
    elif phase in ("shrink", "final", "replay"):
        u0 = 1.55
    if phase == "shrink":
        w = smooth(prog)
        du = 0.44 + (0.12 - 0.44) * w
        dv = 0.62 + (0.18 - 0.62) * w
        v0 = 0.82 + 1.10 * w
    return u0, v0, du, dv, reveal


def left_px(u: float, v: float):
    return (LEFT_X0 + LEFT_XS * u, LEFT_Y0 - LEFT_YS * v)


def map_uv(u: float, v: float, s: float):
    return ((1.0 - s) * u + s * u * math.cos(v), s * u * math.sin(v))


def right_px(u: float, v: float, s: float):
    x, y = map_uv(u, v, s)
    return (RCX + RSCALE * x, RCY - RSCALE * y)


def make_base() -> Image.Image:
    img = Image.new("RGBA", (RES_W, RES_H), (*BG_RGB, 255))
    d = ImageDraw.Draw(img, "RGBA")
    # 分栏
    d.line([(495, 22), (495, RES_H - 18)], fill=(*C_GRID_DIM, 150), width=1)
    d.text((28, 18), "PARAMETER PLANE  (u, v)", font=PIX_FONT, fill=(*C_GRID, 255))
    d.text((526, 18), "IMAGE PLANE  (x, y)", font=PIX_FONT, fill=(*C_GRID, 255))
    # 左：固定矩形网格
    for i in range(0, 13):
        u = UL + (UR - UL) * i / 12
        a, b = left_px(u, 0.0), left_px(u, VMAX)
        d.line([a, b], fill=(*C_GRID_DIM, 105), width=1)
    for j in range(0, 13):
        v = VMAX * j / 12
        a, b = left_px(UL, v), left_px(UR, v)
        d.line([a, b], fill=(*C_GRID_DIM, 105), width=1)
    d.text((30, RES_H - 34), "rectangle  du x dv", font=PIX_FONT, fill=(*C_GRID_DIM, 255))
    d.text((526, RES_H - 34), "T(u,v) = (u cos v, u sin v)",
           font=PIX_FONT, fill=(*C_GRID_DIM, 255))
    return img


BASE_IMG = make_base()


def draw_arrow(d, p0, p1, color, width=2, head=7.0):
    d.line([p0, p1], fill=(*color, 255), width=width)
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    n = math.hypot(dx, dy)
    if n < 1e-6:
        return
    ux, uy = dx / n, dy / n
    px, py = -uy, ux
    a = (p1[0] - head * ux + 0.42 * head * px, p1[1] - head * uy + 0.42 * head * py)
    b = (p1[0] - head * ux - 0.42 * head * px, p1[1] - head * uy - 0.42 * head * py)
    d.polygon([p1, a, b], fill=(*color, 255))


def draw_dashed(d, p0, p1, color, width=1, dash=7.0, gap=5.0):
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
               fill=(*color, 255), width=width)
        dist = e + gap


class JacobianPlane(ImageMobject):
    def __init__(self, disp_height=3.70):
        super().__init__(np.asarray(BASE_IMG).copy())
        self.scale(disp_height / self.height)

    def redraw(self, t: float):
        arr = self.pixel_array
        arr[:] = np.asarray(BASE_IMG)
        img = Image.fromarray(arr, mode="RGBA")
        d = ImageDraw.Draw(img, "RGBA")
        s = morph_s(t)
        u0, v0, du, dv, rev = patch_params(t)
        phase, prog = phase_info(t)

        # --- 右栏：随 s 形变的网格 ---
        for i in range(1, 12):
            u = UL + (UR - UL) * i / 12
            pts = [right_px(u, VMAX * k / 80, s) for k in range(81)]
            d.line(pts, fill=(*C_GRID_DIM, 148), width=1)
        for j in range(0, 12):
            v = VMAX * j / 12
            pts = [right_px((UR - UL) * k / 60, v, s) for k in range(61)]
            d.line(pts, fill=(*C_GRID_DIM, 148), width=1)
        if phase != "intro":
            d.text((RCX + RSCALE * 2.45, RCY - 14), "r", font=PIX_FONT_B,
                   fill=(*C_GRID, 220))
            d.text((RCX - 10, RCY - RSCALE * 2.35 - 18), "theta", font=PIX_FONT,
                   fill=(*C_GRID, 220))

        if rev <= 0.001:
            self.pixel_array[:] = np.asarray(img)
            return

        # --- 左栏：高亮小矩形 ---
        c0 = left_px(u0 - du / 2, v0 + dv / 2)
        c1 = left_px(u0 + du / 2, v0 - dv / 2)
        d.rectangle([c0[0], c0[1], c1[0], c1[1]], fill=(*C_CYAN, int(46 * rev)),
                    outline=(*C_CYAN, int(235 * rev)), width=2)
        # 边长标注
        d.text((c0[0] - 6, c1[1] + 6), "du", font=PIX_FONT,
               fill=(*C_CYAN, int(220 * rev)))
        d.text((c1[0] + 8, c0[1] - 8), "dv", font=PIX_FONT,
               fill=(*C_CYAN, int(220 * rev)))

        # --- 右栏：曲边四边形的像 ---
        corners = [
            right_px(u0 - du / 2, v0 - dv / 2, s),
            right_px(u0 + du / 2, v0 - dv / 2, s),
            right_px(u0 + du / 2, v0 + dv / 2, s),
            right_px(u0 - du / 2, v0 + dv / 2, s),
        ]
        d.polygon(corners, fill=(*C_CYAN, int(60 * rev)))
        d.line(corners + [corners[0]], fill=(*C_CYAN, int(245 * rev)), width=2)

        # --- 两条偏导边向量：青 = ∂T/∂u·du，红 = ∂T/∂v·dv ---
        p0 = right_px(u0 - du / 2, v0 - dv / 2, s)
        pu = right_px(u0 + du / 2, v0 - dv / 2, s)
        pv = right_px(u0 - du / 2, v0 + dv / 2, s)
        draw_arrow(d, p0, pu, C_CYAN, width=2, head=7.0)
        draw_arrow(d, p0, pv, C_RED, width=2, head=7.0)

        # --- 线性近似平行四边形（虚线）---
        q1 = (pu[0] + pv[0] - p0[0], pu[1] + pv[1] - p0[1])
        draw_dashed(d, pu, q1, C_ORANGE, width=1)
        draw_dashed(d, pv, q1, C_ORANGE, width=1)
        d.text((q1[0] + 8, q1[1] - 10), "linear", font=PIX_FONT,
               fill=(*C_ORANGE, int(225 * rev)))

        self.pixel_array[:] = np.asarray(img)


class JacobianChange(Scene):
    def construct(self):
        # 标题：双语、整体淡入后常亮、无打字机竖线
        cn_title = cn_text("雅可比行列式与换元", 50, C_TEXT, 0.30).move_to([0, 4.68, 0])
        en_title = text_img("J A C O B I A N   D E T E R M I N A N T   /   C H A N G E   O F   V A R I A B L E S",
                            18, C_CYAN, 0.10).move_to([0, 4.37, 0])
        watermark = text_img("AIMaker-Lab", 17, C_GRID, 0.09).move_to([3.05, 4.68, 0])
        title = Group(cn_title, en_title, watermark)

        leg = Group(
            Group(Line(LEFT * 0.13, RIGHT * 0.13, color=M_CYAN, stroke_width=3),
                  text_img("PARAM PATCH", 23, C_CYAN, 0.14)).arrange(RIGHT, buff=0.06),
            Group(Line(LEFT * 0.13, RIGHT * 0.13, color=M_CYAN, stroke_width=3),
                  text_img("dT/du * du", 23, C_CYAN, 0.14)).arrange(RIGHT, buff=0.06),
            Group(Line(LEFT * 0.13, RIGHT * 0.13, color=M_RED, stroke_width=3),
                  text_img("dT/dv * dv", 23, C_RED, 0.14)).arrange(RIGHT, buff=0.06),
        ).arrange(RIGHT, buff=0.14).move_to([0, 4.02, 0])

        plane = JacobianPlane(disp_height=3.70).move_to([0, 2.02, 0])

        lbl_a = cn_text("像面积 / 原面积", 27, C_CYAN, 0.22).move_to([-2.05, -0.12, 0])
        lbl_j = cn_text("雅可比行列式 |J| = r", 27, C_ORANGE, 0.22).move_to([2.05, -0.12, 0])
        num_a = PixelNumber(0.0, 3, M_CYAN, 40, nchars=7).set(height=0.30).move_to([-2.05, -0.50, 0])
        num_j = PixelNumber(0.0, 3, M_ORANGE, 40, nchars=7).set(height=0.30).move_to([2.05, -0.50, 0])

        NARR = [
            ("intro",  "换元时，一小块面积会被放大多少倍？", C_TEXT),
            ("morph",  "令 (u,v) 变成 (u cos v, u sin v)：矩形网格弯成极坐标网格", C_CYAN),
            ("patch",  "盯住参数平面里的小矩形：它的两条边分别映成青、红两个向量", C_CYAN),
            ("jac",    "两个向量张成的平行四边形面积，就是局部面积放大率", C_ORANGE),
            ("shrink", "缩小并重放微元：放大率只取决于 r，与 v 和微元大小无关", C_RED),
            ("final",  "这个放大率就是雅可比行列式：dx dy = |J| du dv = r dr dθ", C_CYAN),
            ("replay", "网格怎么弯不重要，重要的是每一点的局部面积怎么变", C_TEXT),
        ]
        narr_pos = [0, -0.93, 0]
        narr_lbls = {}
        for key, s, col in NARR:
            lb = cn_text(s, 26, col, 0.22).move_to(narr_pos)
            lb.move_to([9, 9, 0])
            narr_lbls[key] = lb

        f1 = mi(r"$\int\!\!\int_D f\,dx\,dy=\int\!\!\int_{D'} f\,|J|\,du\,dv$",
                "f1", C_FORMULA, 0.40, 27).move_to([0, -1.52, 0])
        f2 = mi(r"$x=u\cos v,\qquad y=u\sin v\qquad (u=r,\ v=\theta)$",
                "f2", C_FORMULA, 0.28, 27).move_to([0, -2.12, 0])
        f3 = mi(r"$J=\frac{\partial(x,y)}{\partial(u,v)}"
                r"=\frac{\partial x}{\partial u}\frac{\partial y}{\partial v}"
                r"-\frac{\partial x}{\partial v}\frac{\partial y}{\partial u}$",
                "f3", C_FORMULA, 0.43, 27).move_to([0, -2.72, 0])
        f4 = mi(r"$J=\cos v\cdot u\cos v-(-u\sin v)\cdot\sin v=u$",
                "f4", C_FORMULA, 0.28, 27).move_to([0, -3.32, 0])
        f5 = mi(r"$\Delta A_{\mathrm{image}}=u_0\,\Delta u\,\Delta v"
                r"\quad\Rightarrow\quad dx\,dy=r\,dr\,d\theta$",
                "f5", C_FORMULA, 0.28, 27).move_to([0, -3.92, 0])
        FORMULA_LIST = [f1, f2, f3, f4, f5]
        phase_formula = {"intro": 0, "morph": 1, "patch": 2, "jac": 3,
                         "shrink": 4, "final": 0, "replay": 0}

        self.add(leg, lbl_a, lbl_j)
        dynamic = Group(title, plane, num_a, num_j, *narr_lbls.values(), Group(*FORMULA_LIST))
        self.add(dynamic)

        def apply(t):
            phase, prog = phase_info(t)
            u0, v0, du, dv, rev = patch_params(t)
            s = morph_s(t)
            # s=1 时像面积 / 原面积精确等于 u0；形变阶段沿用同一解析式
            ratio = s * s * u0 + s * (1.0 - s) * u0 \
                * (2.0 * math.sin(dv / 2.0) / dv) * math.cos(v0)
            num_a.set_value("—" if rev <= 0.001 else ratio)
            num_j.set_value("—" if rev <= 0.001 else u0)
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
