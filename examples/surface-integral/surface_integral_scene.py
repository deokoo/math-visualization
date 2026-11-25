#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manim 数学动画：曲面积分 —— 向量场穿过曲面的通量

视觉规范与前作一致：
  - 1080x1350 竖屏、30fps、约 22.3s
  - 公式全程固定，只动参数 / 网格 / 动画
  - 无 LaTeX：公式用 matplotlib mathtext 预渲染为透明 PNG
  - 暗色底 #101113、中灰清晰线条、青/红边缘曲线、纯内存动态数字

画面：
  中央 = 参数曲面 Σ: r(u,v)=(u,v,f(u,v)) 的 3D 线框（青色边界）
         ① 底部参数区域 D 亮起，一块面元上长出 r_u、r_v 与法向 r_u×r_v
         ② 青色向量场箭头浮在曲面上；红色箭头表示穿过曲面的通量
         ③ 面元按 u+v 前沿逐块点亮，右侧「累计通量 Φ」0 → 11.278
         ④ 公式链最后一行点亮，定格
  底部 = 5 行固定公式链；中文讲述行随阶段切换

运行：
    manim -pql surface_integral_scene.py SurfaceIntegral
    manim -pqh surface_integral_scene.py SurfaceIntegral
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
M_FORMULA = "#ECE8DC"
M_GRID = "#767C80"
M_GRID_DIM = "#5C6165"
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
    fig = Figure(figsize=(16.0, 2.0), dpi=200)
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


def mi(expr: str, name: str, color=C_FORMULA, height=0.25, size=32) -> ImageMobject:
    path = FORMULA_DIR / f"si_{name}.png"
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

    def __init__(self, value=0.0, places: int = 2, color=M_TEXT, px: int = 40,
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
        if isinstance(value, str):
            txt = value
        else:
            txt = f"{abs(value):.{self.places}f}"
            if value < 0:
                txt = "-" + txt
        img = Image.new("RGBA", (self._cw, self._ch), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        box = d.textbbox((0, 0), txt, font=self._font)
        tw, th = box[2] - box[0], box[3] - box[1]
        d.text(((self._cw - tw) / 2 - box[0], (self._ch - th) / 2 - box[1]),
               txt, font=self._font, fill=(*self._color, 255))
        self.pixel_array[:] = np.asarray(img)


# ---------------------------------------------------------------------------
# 数学设置
#   Σ: r(u,v)=(u,v,f(u,v)),  D=[−1.6,1.6]²
#   f(u,v)=e^(−0.45(u²+v²))+0.25
#   r_u=(1,0,f_u), r_v=(0,1,f_v), r_u×r_v=(−f_u,−f_v,1)
#   F=(0.30v, 0.20u, 1.10+0.20u) → F·(r_u×r_v) ∈ [0.895, 1.535]
#   Φ=∬_Σ F·n dS ≈ 11.278093（1600² 中点法校验）
# ---------------------------------------------------------------------------
A = 1.6
RES = 520
BG_RGB = (16, 17, 19)
PHI_ROT = -0.55
COS_P, SIN_P = math.cos(PHI_ROT), math.sin(PHI_ROT)
CX, CY = RES / 2, 334
SCALE, TILT, ZS = 100.0, 0.42, 0.90


def f(u, v):
    return np.exp(-0.45 * (np.asarray(u) ** 2 + np.asarray(v) ** 2)) + 0.25


def fu(u, v):
    return -0.9 * np.asarray(u) * np.exp(-0.45 * (np.asarray(u) ** 2 + np.asarray(v) ** 2))


def fv(u, v):
    return -0.9 * np.asarray(v) * np.exp(-0.45 * (np.asarray(u) ** 2 + np.asarray(v) ** 2))


def dens(u, v):
    Fx, Fy, Fz = 0.30 * v, 0.20 * u, 1.10 + 0.20 * u
    return Fx * (-fu(u, v)) + Fy * (-fv(u, v)) + Fz


def proj(u, v, z):
    xr = u * COS_P - v * SIN_P
    yr = u * SIN_P + v * COS_P
    return (CX + SCALE * xr, CY + SCALE * TILT * yr - SCALE * ZS * z)


# 预计算：线框与区域网格
US = np.linspace(-A, A, 17)
VS = np.linspace(-A, A, 17)
US_FINE = np.linspace(-A, A, 61)
WIRE_U = [[proj(float(u), float(v), float(f(u, v))) for v in US_FINE] for u in US]
WIRE_V = [[proj(float(u), float(v), float(f(u, v))) for u in US_FINE] for v in VS]
BASE_U = [[proj(float(u), float(v), 0.0) for v in (-A, A)] for u in US[::2]]
BASE_V = [[proj(float(u), float(v), 0.0) for u in (-A, A)] for v in VS[::2]]
BOUNDARY = ([proj(float(u), -A, float(f(u, -A))) for u in US_FINE]
            + [proj(A, float(v), float(f(A, v))) for v in US_FINE]
            + [proj(float(u), A, float(f(u, A))) for u in US_FINE[::-1]]
            + [proj(-A, float(v), float(f(-A, v))) for v in US_FINE[::-1]])

# 面元（14×14），权重归一化到精确总通量
NU = NV = 14
H = 2 * A / NU
PATCHES = []
for i in range(NU):
    for j in range(NV):
        u, v = -A + (i + 0.5) * H, -A + (j + 0.5) * H
        PATCHES.append((u, v, float(dens(u, v))))
_raw = sum(w for _, _, w in PATCHES) * H * H
TOTAL = 11.2780932110086
# 每块权重 = 通量密度 × 面积，归一后逐块累加精确收敛到 TOTAL
PATCHES = [(u, v, w * H * H * TOTAL / _raw) for u, v, w in PATCHES]
PATCHES.sort(key=lambda e: e[0] + e[1])          # 沿 u+v 前沿逐块扫过
CUM = np.concatenate([[0.0], np.cumsum([w for _, _, w in PATCHES])])


def arrow(d, p0, p1, color, width=2, head=7):
    d.line([p0, p1], fill=color, width=width)
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    for da in (math.pi - 0.42, math.pi + 0.42):
        d.line([p1, (p1[0] + head * math.cos(ang + da),
                     p1[1] + head * math.sin(ang + da))], fill=color, width=width)


def make_base() -> Image.Image:
    img = Image.new("RGBA", (RES, RES), (*BG_RGB, 255))
    d = ImageDraw.Draw(img, "RGBA")
    # 参数域 D（z=0 平面）
    for seg in BASE_U + BASE_V:
        d.line(seg, fill=(*C_GRID_DIM, 170), width=1)
    # 曲面线框
    for seg in WIRE_U + WIRE_V:
        d.line(seg, fill=(92, 100, 104, 205), width=1)
    # 曲面边界（青）
    d.line(BOUNDARY + [BOUNDARY[0]], fill=(*C_CYAN, 255), width=3, joint="curve")
    return img


BASE_IMG = make_base()


class SurfaceField(ImageMobject):
    def __init__(self, disp_height=3.90):
        super().__init__(np.asarray(BASE_IMG).copy())
        self.scale(disp_height / self.height)

    def redraw(self, t: float):
        phase, prog = phase_info(t)
        arr = self.pixel_array
        arr[:] = np.asarray(BASE_IMG)
        img = Image.fromarray(arr, mode="RGBA")
        d = ImageDraw.Draw(img, "RGBA")

        # 阶段①：参数域 D 边框高亮
        if phase in ("param", "field", "integrate", "equal", "replay"):
            corners = [proj(-A, -A, 0), proj(A, -A, 0),
                       proj(A, A, 0), proj(-A, A, 0)]
            d.line(corners + [corners[0]], fill=(*C_ORANGE, 200), width=2)

        # 阶段①：一块面元 + r_u / r_v / 法向
        if phase == "param":
            u0, v0 = 0.48, -0.42
            z0 = float(f(u0, v0))
            quad = [proj(u0, v0, z0), proj(u0 + H, v0, float(f(u0 + H, v0))),
                    proj(u0 + H, v0 + H, float(f(u0 + H, v0 + H))),
                    proj(u0, v0 + H, float(f(u0, v0 + H)))]
            d.polygon(quad, fill=(238, 148, 72, 90), outline=(238, 148, 72, 230))
            p = proj(u0, v0, z0)
            arrow(d, p, proj(u0 + 0.55, v0, float(f(u0 + 0.55, v0))),
                  (255, 128, 112, 255), 3, 8)                     # r_u
            arrow(d, p, proj(u0, v0 + 0.55, float(f(u0, v0 + 0.55))),
                  (238, 190, 90, 255), 3, 8)                     # r_v
            nx, ny = -float(fu(u0, v0)), -float(fv(u0, v0))
            nn = math.hypot(nx, ny, 1.0)
            arrow(d, p, proj(u0 + 0.42 * nx / nn, v0 + 0.42 * ny / nn,
                             z0 + 0.42 / nn), (*C_CYAN, 255), 3, 9)   # r_u×r_v

        # 阶段②起：向量场箭头（青）与穿面通量箭头（红）
        if phase in ("field", "integrate", "equal", "replay"):
            q = np.arange(-1.35, 1.36, 0.675)
            for u in q:
                for v in q:
                    z = float(f(u, v))
                    Fx, Fy, Fz = 0.30 * v, 0.20 * u, 1.10 + 0.20 * u
                    nn = math.hypot(Fx, Fy, Fz)
                    s = 0.26 / nn
                    p0 = proj(float(u), float(v), z)
                    p1 = proj(float(u) + s * Fx, float(v) + s * Fy, z + s * Fz)
                    arrow(d, p0, p1, (74, 214, 200, 205), 2, 5)
            for u in (-0.95, 0.0, 0.95):
                for v in (-0.95, 0.0, 0.95):
                    z = float(f(u, v))
                    arrow(d, proj(u, v, z - 0.22), proj(u, v, z + 0.22),
                          (232, 96, 82, 220), 2, 6)

        # 阶段③起：面元按 u+v 前沿点亮
        reveal = 0
        if phase == "integrate":
            reveal = max(1, int(prog * len(PATCHES)))
        elif phase in ("equal", "replay"):
            reveal = len(PATCHES)
        for u, v, w in PATCHES[:reveal]:
            quad = [proj(u - H / 2, v - H / 2, float(f(u - H / 2, v - H / 2))),
                    proj(u + H / 2, v - H / 2, float(f(u + H / 2, v - H / 2))),
                    proj(u + H / 2, v + H / 2, float(f(u + H / 2, v + H / 2))),
                    proj(u - H / 2, v + H / 2, float(f(u - H / 2, v + H / 2)))]
            k = float(np.clip((w / H / H - 0.88) / 0.68, 0, 1))
            d.polygon(quad, fill=(int(150 + 70 * k), int(44 + 34 * k), int(38 + 26 * k), 175))

        self.pixel_array[:] = np.asarray(img)


PHASES = [
    ("intro",     0.00,  1.30),
    ("param",     1.30,  5.20),
    ("field",     5.20,  9.00),
    ("integrate", 9.00, 16.60),
    ("equal",    16.60, 18.40),
    ("replay",   18.40, DURATION),
]


def phase_info(t: float):
    for name, t0, t1 in PHASES:
        if t < t1:
            return name, (0.0 if name == "intro"
                          else float(np.clip((t - t0) / (t1 - t0), 0.0, 1.0)))
    return "replay", 1.0


class SurfaceIntegral(Scene):
    def construct(self):
        cn_title = cn_text("曲面积分", 50, C_TEXT, 0.30).move_to([0, 4.68, 0])
        en_title = text_img("S U R F A C E   I N T E G R A L   /   F L U X   T H R O U G H   A   S U R F A C E",
                            19, C_CYAN, 0.11).move_to([0, 4.37, 0])
        watermark = text_img("AIMaker-Lab", 17, C_GRID, 0.09).move_to([3.05, 4.68, 0])
        title = Group(cn_title, en_title, watermark)
        cursor = Line(UP * 0.16, DOWN * 0.16, color=M_TEXT, stroke_width=2.5).move_to([0, 4.68, 0])
        cursor.set_opacity(0)

        leg = Group(
            Group(Line(LEFT * 0.13, RIGHT * 0.13, color=M_CYAN, stroke_width=3),
                  text_img("VECTOR FIELD", 24, C_CYAN, 0.14)).arrange(RIGHT, buff=0.06),
            Group(Line(LEFT * 0.13, RIGHT * 0.13, color=M_RED, stroke_width=3),
                  text_img("FLUX", 24, C_RED, 0.14)).arrange(RIGHT, buff=0.06),
            Group(Line(LEFT * 0.13, RIGHT * 0.13, color=M_ORANGE, stroke_width=3),
                  text_img("SURFACE ELEMENT", 24, C_ORANGE, 0.14)).arrange(RIGHT, buff=0.06),
        ).arrange(RIGHT, buff=0.18).move_to([0, 4.02, 0])

        plane = SurfaceField(disp_height=3.90).move_to([0, 1.85, 0])

        lbl_d = cn_text("当前 F·n", 30, C_RED, 0.21).move_to([-2.30, -0.22, 0])
        lbl_phi = cn_text("累计通量 Φ", 30, C_ORANGE, 0.21).move_to([2.30, -0.22, 0])
        num_d = PixelNumber(0.0, 3, M_RED, 40, nchars=6).set(height=0.30).move_to([-2.30, -0.66, 0])
        num_phi = PixelNumber(0.0, 3, M_ORANGE, 40, nchars=7).set(height=0.30).move_to([2.30, -0.66, 0])

        NARR = [
            ("intro",     "曲面 Σ 与穿过它的向量场 F", C_TEXT),
            ("param",     "参数化：两条切向量叉出法向", C_ORANGE),
            ("field",     "通量密度 F·n：穿面流动的强度", C_CYAN),
            ("integrate", "每个面元贡献一点，逐块累加", C_RED),
            ("equal",     "曲面积分 = 参数区域上的二重积分", C_TEXT),
            ("replay",    "把曲面摊平成参数域来算", C_TEXT),
        ]
        narr_pos = [0, -1.30, 0]
        narr_lbls = {}
        for key, s, col in NARR:
            lb = cn_text(s, 27, col, 0.23).move_to(narr_pos)
            lb.move_to([9, 9, 0])
            narr_lbls[key] = lb

        f1 = mi(r"$D=[-a,a]^2,\quad \Sigma:\ \mathbf{r}(u,v)=(u,v,f(u,v))$",
                "f1", C_FORMULA, 0.29, 29).move_to([0, -1.98, 0])
        f2 = mi(r"$\mathbf{r}_u=(1,0,f_u),\quad \mathbf{r}_v=(0,1,f_v)$",
                "f2", C_FORMULA, 0.29, 29).move_to([0, -2.60, 0])
        f3 = mi(r"$\mathbf{r}_u\times\mathbf{r}_v=(-f_u,-f_v,1)$",
                "f3", C_FORMULA, 0.29, 29).move_to([0, -3.22, 0])
        f4 = mi(r"$\mathbf{F}\cdot\mathbf{n}\,dS=\mathbf{F}(\mathbf{r})\cdot(\mathbf{r}_u\times\mathbf{r}_v)\,du\,dv$",
                "f4", C_FORMULA, 0.34, 28).move_to([0, -3.84, 0])
        f5 = mi(r"$\Phi=\iint_\Sigma\mathbf{F}\cdot\mathbf{n}\,dS=\iint_D\mathbf{F}(\mathbf{r})\cdot(\mathbf{r}_u\times\mathbf{r}_v)\,du\,dv$",
                "f5", C_FORMULA, 0.34, 27).move_to([0, -4.48, 0])
        formulas = Group(f1, f2, f3, f4, f5)
        FORMULA_COLORS = [f1, f2, f3, f4, f5]
        phase_formula = {"intro": 0, "param": 1, "field": 3,
                         "integrate": 4, "equal": 4, "replay": 4}

        self.add(leg, lbl_d, lbl_phi)
        dynamic = Group(title, cursor, plane, num_d, num_phi, *narr_lbls.values(), formulas)
        self.add(dynamic)

        def apply(t):
            phase, prog = phase_info(t)
            plane.redraw(t)

            if phase in ("intro", "param", "field"):
                num_d.set_value(0.0)
                num_phi.set_value(0.0)
            elif phase == "integrate":
                idx = max(1, int(prog * len(PATCHES)))
                u, v, w = PATCHES[idx - 1]
                num_d.set_value(float(dens(u, v)))
                num_phi.set_value(float(CUM[idx]))
            else:
                num_d.set_value(float(dens(0.48, -0.42)))
                num_phi.set_value(TOTAL)

            for k, lb in narr_lbls.items():
                lb.move_to(narr_pos if k == phase else [9, 9, 0])

            hi = phase_formula[phase]
            for i, fm in enumerate(FORMULA_COLORS):
                fm.set_opacity(1.0 if i == hi else 0.48)

            blink = (t < 0.28) or (18.32 <= t <= 18.66)
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
