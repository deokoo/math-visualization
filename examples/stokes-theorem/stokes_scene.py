#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manim 数学动画：斯托克斯公式 —— 边界环流 = 曲面旋度通量

视觉规范与前作一致：
  - 1080x1350 竖屏、30fps、约 22.3s
  - 公式全程固定，只动参数 / 网格 / 动画
  - 无 LaTeX：公式用 matplotlib mathtext 预渲染为透明 PNG
  - 暗色底 #101113、中灰清晰线条、青/红边缘曲线、纯内存动态数字

画面：
  中央 = 以圆 C 为边界的曲面 Σ（先为抛物面帽，后形变为平圆盘）
         ① 橙色闭合边界 C 亮起，红色切向箭头绕行，左侧环流计数增长
         ② 青色旋度向量 ∇×F 浮上曲面
         ③ 曲面片按极角逐扇区点亮，右侧旋度通量追上环流
         ④ 曲面连续压平为共享同一边界的圆盘，两个数值都不变
         ⑤ 公式链最后一行点亮，定格
  底部 = 5 行固定公式链；中文讲述行随阶段切换

运行：
    manim -pql preview_scene.py StokesTheorem      # 半分辨率预览
    manim -pqh stokes_scene.py StokesTheorem       # 1080×1350 成片
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
    path = FORMULA_DIR / f"st_{name}.png"
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
#   C = ∂Σ : x²+y² = R², z = z0
#   Σ(h)  : z = z0 + h(1 − r²/R²),  h: H → 0（形变为平圆盘，边界不变）
#   F      = (−y, x, xy)  ⇒  ∇×F = (x, −y, 2)
#   ∮_C F·dr = ∫₀^{2π} R² dθ = 2πR²
#   (∇×F)·n dS = [2 + 2h(x²−y²)/R²] dx dy   ⇒  ∬_Σ = 2πR²（与 h 无关）
# ---------------------------------------------------------------------------
R = 1.30
Z0 = 0.35
H = 0.80
TOTAL = 2.0 * math.pi * R * R

RES = 520
BG_RGB = (16, 17, 19)
PHI_ROT = -0.55
COS_P, SIN_P = math.cos(PHI_ROT), math.sin(PHI_ROT)
CX, CY = RES / 2, 334
SCALE, TILT, ZS = 175.0, 0.42, 0.90


def surf(hm: float, r: float, th: float):
    x, y = r * math.cos(th), r * math.sin(th)
    return x, y, Z0 + hm * (1.0 - r * r / (R * R))


def proj(x: float, y: float, z: float):
    xr = x * COS_P - y * SIN_P
    yr = x * SIN_P + y * COS_P
    return (CX + SCALE * xr, CY + SCALE * TILT * yr - SCALE * ZS * z)


def flux_numeric(hm: float, n: int = 900) -> float:
    """高分辨率极坐标数值积分，校验旋度通量对任意 h 都等于 2πR²。"""
    rs = (np.arange(n) + 0.5) / n * R
    ths = (np.arange(2 * n) + 0.5) / (2 * n) * 2 * np.pi
    RR, TT = np.meshgrid(rs, ths, indexing="ij")
    x, y = RR * np.cos(TT), RR * np.sin(TT)
    dens = 2.0 + 2.0 * hm * (x * x - y * y) / (R * R)
    dA = (R / n) * (2 * np.pi / (2 * n)) * RR
    return float(np.sum(dens * dA))


# 预计算：静态底图（地面投影圆）
BASE_CIRCLE = [proj(R * math.cos(th), R * math.sin(th), 0.0)
               for th in np.linspace(0, 2 * np.pi, 181)]


def make_base() -> Image.Image:
    img = Image.new("RGBA", (RES, RES), (*BG_RGB, 255))
    d = ImageDraw.Draw(img, "RGBA")
    d.line(BASE_CIRCLE, fill=(*C_GRID_DIM, 180), width=1)
    return img


BASE_IMG = make_base()

# 面元：7 层 × 28 扇区，按极角排序形成绕边界一周的扫过前沿
NR, NT = 7, 28
DR, DT = R / NR, 2 * math.pi / NT
PATCHES = []
for ir in range(NR):
    r = (ir + 0.5) * DR
    for it in range(NT):
        th = (it + 0.5) * DT
        PATCHES.append((r, th, r * DR * DT))
RAW = sum(r * DR * DT for r, _, _ in PATCHES)
# 离散权重新：保证精确收敛到 2πR²（对任意形变 h 数学上都成立）
PATCHES = [(r, th, r * DR * DT * TOTAL / RAW) for r, th, _ in PATCHES]
PATCHES.sort(key=lambda e: (e[1] % (2 * np.pi), e[0]))
CUM = np.concatenate([[0.0], np.cumsum([w for _, _, w in PATCHES])])


def arrow(d, p0, p1, color, width=2, head=7):
    d.line([p0, p1], fill=color, width=width)
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    for da in (math.pi - 0.42, math.pi + 0.42):
        d.line([p1, (p1[0] + head * math.cos(ang + da),
                     p1[1] + head * math.sin(ang + da))], fill=color, width=width)


class StokesField(ImageMobject):
    def __init__(self, disp_height=3.90):
        super().__init__(np.asarray(BASE_IMG).copy())
        self.scale(disp_height / self.height)

    def redraw(self, t: float):
        phase, prog = phase_info(t)
        hm = H
        if phase == "swap":
            hm = H * (1.0 - prog * prog * (3.0 - 2.0 * prog))  # smoothstep 压平
        elif phase in ("equal", "replay"):
            hm = 0.0

        arr = self.pixel_array
        arr[:] = np.asarray(BASE_IMG)
        img = Image.fromarray(arr, mode="RGBA")
        d = ImageDraw.Draw(img, "RGBA")

        # 曲面线框：同心圆 + 径向辐条（随 hm 连续形变）
        for k in range(1, 7):
            r = R * k / 6
            seg = [proj(*surf(hm, r, th)) for th in np.linspace(0, 2 * np.pi, 97)]
            d.line(seg, fill=(96, 104, 108, 200), width=1)
        for k in range(12):
            th = 2 * np.pi * k / 12
            seg = [proj(*surf(hm, r, th)) for r in np.linspace(0, R, 33)]
            d.line(seg, fill=(88, 96, 100, 175), width=1)

        # 边界 C：橙色闭合曲线（形变时始终不动）
        boundary = [proj(R * math.cos(th), R * math.sin(th), Z0)
                    for th in np.linspace(0, 2 * np.pi, 181)]
        d.line(boundary + [boundary[0]], fill=(*C_ORANGE, 255), width=3, joint="curve")

        # 阶段①：红色切向箭头绕边界，已走过的一段加粗
        if phase in ("boundary", "curl", "flux", "swap", "equal", "replay"):
            for k in range(12):
                th = 2 * np.pi * k / 12
                p0 = proj(R * math.cos(th), R * math.sin(th), Z0)
                p1 = proj(R * math.cos(th) - 0.24 * math.sin(th),
                          R * math.sin(th) + 0.24 * math.cos(th), Z0)
                arrow(d, p0, p1, (*C_RED, 235), 2, 6)
        if phase == "boundary":
            stop = max(2, int(prog * 180))
            d.line(boundary[:stop], fill=(*C_RED, 255), width=5, joint="curve")
            d.ellipse([boundary[stop - 1][0] - 5, boundary[stop - 1][1] - 5,
                       boundary[stop - 1][0] + 5, boundary[stop - 1][1] + 5],
                      fill=(*C_RED, 255))

        # 阶段②起：青色旋度向量 ∇×F = (x, −y, 2)
        if phase in ("curl", "flux", "swap", "equal", "replay"):
            for r in (0.38, 0.78, 1.10):
                for k in range(12):
                    th = 2 * np.pi * k / 12 + 0.13
                    x, y, z = surf(hm, r, th)
                    cx, cy, cz = x, -y, 2.0
                    nn = math.hypot(cx, cy, cz)
                    s = 0.27 / nn
                    p0 = proj(x, y, z)
                    p1 = proj(x + s * cx, y + s * cy, z + s * cz)
                    arrow(d, p0, p1, (*C_CYAN, 220), 2, 5)

        # 阶段③起：面元按极角逐扇区点亮（旋度通量）
        reveal = 0
        if phase == "flux":
            reveal = max(1, int(prog * len(PATCHES)))
        elif phase in ("swap", "equal", "replay"):
            reveal = len(PATCHES)
        for r, th, w in PATCHES[:reveal]:
            quad = [proj(*surf(hm, r - DR / 2, th - DT / 2)),
                    proj(*surf(hm, r + DR / 2, th - DT / 2)),
                    proj(*surf(hm, r + DR / 2, th + DT / 2)),
                    proj(*surf(hm, r - DR / 2, th + DT / 2))]
            dens = 2.0 + 2.0 * hm * ((r * math.cos(th)) ** 2
                                     - (r * math.sin(th)) ** 2) / (R * R)
            k = float(np.clip((dens - 0.5) / 3.0, 0, 1))
            d.polygon(quad, fill=(int(150 + 70 * k), int(44 + 34 * k),
                                  int(38 + 26 * k), 172))

        # 曲面形变时，原抛物面帽留一层淡淡的" Ghost "轮廓
        if phase in ("swap", "equal", "replay"):
            for rr in (R / 2, R):
                seg = [proj(*surf(H, rr, th)) for th in np.linspace(0, 2 * np.pi, 97)]
                d.line(seg, fill=(104, 110, 114, 95), width=1)

        self.pixel_array[:] = np.asarray(img)


PHASES = [
    ("intro",     0.00,  1.30),
    ("boundary",  1.30,  5.20),
    ("curl",      5.20,  8.60),
    ("flux",      8.60, 13.60),
    ("swap",     13.60, 17.20),
    ("equal",    17.20, 19.60),
    ("replay",   19.60, DURATION),
]


def phase_info(t: float):
    for name, t0, t1 in PHASES:
        if t < t1:
            return name, (0.0 if name == "intro"
                          else float(np.clip((t - t0) / (t1 - t0), 0.0, 1.0)))
    return "replay", 1.0


class StokesTheorem(Scene):
    def construct(self):
        cn_title = cn_text("斯托克斯公式", 50, C_TEXT, 0.30).move_to([0, 4.68, 0])
        en_title = text_img("S T O K E S '   T H E O R E M   /   C U R L   A N D   C I R C U L A T I O N",
                            19, C_CYAN, 0.11).move_to([0, 4.37, 0])
        watermark = text_img("AIMaker-Lab", 17, C_GRID, 0.09).move_to([3.05, 4.68, 0])
        title = Group(cn_title, en_title, watermark)
        cursor = Line(UP * 0.16, DOWN * 0.16, color=M_TEXT, stroke_width=2.5).move_to([0, 4.68, 0])
        cursor.set_opacity(0)

        leg = Group(
            Group(Line(LEFT * 0.13, RIGHT * 0.13, color=M_ORANGE, stroke_width=3),
                  text_img("BOUNDARY C", 24, C_ORANGE, 0.14)).arrange(RIGHT, buff=0.06),
            Group(Line(LEFT * 0.13, RIGHT * 0.13, color=M_CYAN, stroke_width=3),
                  text_img("CURL FIELD", 24, C_CYAN, 0.14)).arrange(RIGHT, buff=0.06),
            Group(Line(LEFT * 0.13, RIGHT * 0.13, color=M_RED, stroke_width=3),
                  text_img("CIRCULATION", 24, C_RED, 0.14)).arrange(RIGHT, buff=0.06),
        ).arrange(RIGHT, buff=0.18).move_to([0, 4.02, 0])

        plane = StokesField(disp_height=3.90).move_to([0, 1.85, 0])

        lbl_circ = cn_text("边界环流", 30, C_RED, 0.21).move_to([-2.30, -0.22, 0])
        lbl_flux = cn_text("曲面旋度通量", 30, C_CYAN, 0.21).move_to([2.30, -0.22, 0])
        num_circ = PixelNumber(0.0, 3, M_RED, 40, nchars=7).set(height=0.30).move_to([-2.30, -0.66, 0])
        num_flux = PixelNumber(0.0, 3, M_CYAN, 40, nchars=7).set(height=0.30).move_to([2.30, -0.66, 0])

        NARR = [
            ("intro",     "一张曲面，一条边界，一个惊人的相等", C_TEXT),
            ("boundary",  "沿闭合边界走一圈：F·dr 的环流", C_ORANGE),
            ("curl",      "旋度向量场：曲面上的局部旋转强度", C_CYAN),
            ("flux",      "旋度通量逐块累加，追上边界环流", C_RED),
            ("swap",      "换一张共享同一边界的曲面，答案不变", C_ORANGE),
            ("equal",     "边界环流 = 曲面旋度通量：斯托克斯公式", C_TEXT),
            ("replay",    "积分只认边界，这就是斯托克斯的精髓", C_TEXT),
        ]
        narr_pos = [0, -1.30, 0]
        narr_lbls = {}
        for key, s, col in NARR:
            lb = cn_text(s, 27, col, 0.23).move_to(narr_pos)
            lb.move_to([9, 9, 0])
            narr_lbls[key] = lb

        f1 = mi(r"$C=\partial\Sigma:\ x^2+y^2=R^2,\quad z=z_0$",
                "f1", C_FORMULA, 0.29, 29).move_to([0, -1.98, 0])
        f2 = mi(r"$\mathbf{F}=(-y,\ x,\ xy),\quad \nabla\times\mathbf{F}=(x,\ -y,\ 2)$",
                "f2", C_FORMULA, 0.29, 27).move_to([0, -2.60, 0])
        f3 = mi(r"$\oint_C\mathbf{F}\cdot d\mathbf{r}=\int_0^{2\pi}R^2\,d\theta=2\pi R^2$",
                "f3", C_FORMULA, 0.50, 27).move_to([0, -3.22, 0])
        f4 = mi(r"$(\nabla\times\mathbf{F})\cdot\mathbf{n}\,dS=\left(2+\frac{2h(x^2-y^2)}{R^2}\right)dx\,dy$",
                "f4", C_FORMULA, 0.48, 25).move_to([0, -3.84, 0])
        f5 = mi(r"$\oint_{\partial\Sigma}\mathbf{F}\cdot d\mathbf{r}=\iint_\Sigma(\nabla\times\mathbf{F})\cdot\mathbf{n}\,dS$",
                "f5", C_FORMULA, 0.42, 26).move_to([0, -4.48, 0])
        formulas = Group(f1, f2, f3, f4, f5)
        FORMULA_COLORS = [f1, f2, f3, f4, f5]
        phase_formula = {"intro": 0, "boundary": 2, "curl": 1, "flux": 3,
                         "swap": 3, "equal": 4, "replay": 4}

        self.add(leg, lbl_circ, lbl_flux)
        dynamic = Group(title, cursor, plane, num_circ, num_flux, *narr_lbls.values(), formulas)
        self.add(dynamic)

        def apply(t):
            phase, prog = phase_info(t)
            plane.redraw(t)

            if phase == "boundary":
                num_circ.set_value(TOTAL * prog)
                num_flux.set_value(0.0)
            elif phase == "curl":
                num_circ.set_value(TOTAL)
                num_flux.set_value(0.0)
            elif phase == "flux":
                num_circ.set_value(TOTAL)
                num_flux.set_value(TOTAL * prog)
            else:
                num_circ.set_value(TOTAL)
                num_flux.set_value(TOTAL)

            for k, lb in narr_lbls.items():
                lb.move_to(narr_pos if k == phase else [9, 9, 0])

            hi = phase_formula[phase]
            for i, fm in enumerate(FORMULA_COLORS):
                fm.set_opacity(1.0 if i == hi else 0.48)

            blink = (t < 0.28) or (17.22 <= t <= 17.56)
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
