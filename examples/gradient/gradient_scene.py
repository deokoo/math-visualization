#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manim 数学动画：梯度与方向导数 —— 沿等高线走变化为零，沿梯度走变化最快

视觉规范与前作一致：
  - 1080x1350 竖屏、30fps、约 22.3s
  - 公式全程固定，只动参数 / 箭头 / 网格 / 动画
  - 无 LaTeX：公式用 matplotlib mathtext 预渲染为透明 PNG
  - 暗色底 #101113、中灰清晰线条、青/红边缘箭头、纯内存动态数字

画面：
  中央 = 标量场 f(x,y) 的等高线地形（灰度等高线 + 暗青热图）
         ① 青色箭头 ∇f 在定点 P 出现，直角标记显示它垂直于等高线
         ② 红色单位方向 u 绕 P 旋转，橙色角弧标出与梯度的夹角 θ
         ③ 左侧「梯度模长 |∇f|」固定，右侧「方向导数 D_u f」随 cosθ 实时变化
         ④ θ=90° 时数值归零（沿等高线不动），θ=0° 时数值升到 |∇f|（最大值）
         ⑤ 底部公式为完整推导链，中文讲述行随阶段切换
  顶部 = 中文主题名「梯度与方向导数」+ 英文数学标题
  底部 = 5 行固定公式链

数学设置：
  f(x,y) = 1.10 exp(-((x-0.35)^2+(y-0.25)^2)/0.90)
         + 0.55 exp(-((x+0.75)^2+(y+0.65)^2)/0.70)
         - 0.35 exp(-((x-0.60)^2+(y+0.80)^2)/0.50)
  P = (0.8, -0.2)，∇f(P) ≈ (-0.667, 0.527)，|∇f| ≈ 0.850
  D_u f = ∇f · u = |∇f| cos θ  （u 为单位向量）

运行：
    manim -pql gradient_scene.py Gradient
    manim -pqh gradient_scene.py Gradient
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
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
_SANS = str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf")

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
    path = FORMULA_DIR / f"gr_{name}.png"
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
    """中文标题/讲述行：微软雅黑渲染（衬线字体缺中文字形）。"""
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
    """PIL 渲染到固定 RGBA 数组，set_value 原地更新，零文件 IO。"""

    def __init__(self, value=0.0, places: int = 2, color=M_TEXT, px: int = 40,
                 nchars: int | None = None, **kwargs):
        self.places = places
        self.value = value
        self._color = tuple(int(c) for c in ManimColor(color).to_rgb() * 255) \
            if not isinstance(color, tuple) else color
        self._font = ImageFont.truetype(_SANS, px)
        n = nchars if nchars is not None else (1 + 1 + places + (1 if places else 0))
        self._cw, self._ch = int(px * (n * 0.62 + 0.4)), int(px * 1.6)
        canvas = np.zeros((self._ch, self._cw, 4), dtype=np.uint8)
        super().__init__(canvas, **kwargs)
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
# 数学对象：标量场 f、梯度 ∇f、定点 P
# ---------------------------------------------------------------------------
def fval(x, y):
    e1 = np.exp(-((np.asarray(x) - 0.35) ** 2 + (np.asarray(y) - 0.25) ** 2) / 0.90)
    e2 = np.exp(-((np.asarray(x) + 0.75) ** 2 + (np.asarray(y) + 0.65) ** 2) / 0.70)
    e3 = np.exp(-((np.asarray(x) - 0.60) ** 2 + (np.asarray(y) + 0.80) ** 2) / 0.50)
    return 1.10 * e1 + 0.55 * e2 - 0.35 * e3


def grad_f(x, y):
    x, y = float(x), float(y)
    e1 = math.exp(-((x - 0.35) ** 2 + (y - 0.25) ** 2) / 0.90)
    e2 = math.exp(-((x + 0.75) ** 2 + (y + 0.65) ** 2) / 0.70)
    e3 = math.exp(-((x - 0.60) ** 2 + (y + 0.80) ** 2) / 0.50)
    fx = (1.10 * (-2 * (x - 0.35) / 0.90) * e1
          + 0.55 * (-2 * (x + 0.75) / 0.70) * e2
          - 0.35 * (-2 * (x - 0.60) / 0.50) * e3)
    fy = (1.10 * (-2 * (y - 0.25) / 0.90) * e1
          + 0.55 * (-2 * (y + 0.65) / 0.70) * e2
          - 0.35 * (-2 * (y + 0.80) / 0.50) * e3)
    return fx, fy


PX, PY = 0.8, -0.2
GX, GY = grad_f(PX, PY)
GMAG = math.hypot(GX, GY)
GHX, GHY = GX / GMAG, GY / GMAG          # 梯度单位方向
THX, THY = -GHY, GHX                     # 等高线切向（垂直梯度）
PHI_G = math.atan2(GHY, GHX)             # 梯度的世界角

# 有限差分交叉验证梯度公式（防止解析导数写错）
_h = 1e-5
NUM_GX = (float(fval(PX + _h, PY)) - float(fval(PX - _h, PY))) / (2 * _h)
NUM_GY = (float(fval(PX, PY + _h)) - float(fval(PX, PY - _h))) / (2 * _h)
assert abs(NUM_GX - GX) < 1e-5 and abs(NUM_GY - GY) < 1e-5, "梯度解析式错误"

# 等高线线段（matplotlib 提取，映射到世界坐标；逐帧只查表）
W = 1.70
_xg = np.linspace(-W, W, 420)
_yg = np.linspace(-W, W, 420)
_Xg, _Yg = np.meshgrid(_xg, _yg)
_fig = Figure(figsize=(2, 2), dpi=100)
_ax = _fig.add_subplot(111)
_CS = _ax.contour(_Xg, _Yg, fval(_Xg, _Yg),
                  levels=np.linspace(-0.28, 1.12, 16), colors="k")
CONTOUR_SEGS = [[(float(a), float(b)) for a, b in seg]
                for level in _CS.allsegs for seg in level if len(seg) >= 2]

# ---------------------------------------------------------------------------
# 地形位图：静态底图（热图 + 等高线 + 网格 + P 点）；每帧叠加动态箭头层
# ---------------------------------------------------------------------------
RES = 480
BG_RGB = (16, 17, 19)


def w2p(x, y):
    return ((x + W) / (2 * W) * RES,
            (W - y) / (2 * W) * RES)


def make_base() -> Image.Image:
    # 暗青热图：低处近背景，高处偏青，保持暗调让箭头突出
    xs = np.linspace(-W, W, RES)
    ys = np.linspace(W, -W, RES)          # 图像第 0 行 = 世界 y=+W
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    F = fval(X, Y)
    u = np.clip((F + 0.28) / 1.40, 0, 1)
    arr = np.zeros((RES, RES, 4), dtype=np.uint8)
    arr[..., 0] = (12 + 16 * u).astype(np.uint8)
    arr[..., 1] = (17 + 50 * u).astype(np.uint8)
    arr[..., 2] = (19 + 52 * u).astype(np.uint8)
    arr[..., 3] = 255
    # 边缘压暗（暗角）
    yy, xx = np.mgrid[0:RES, 0:RES]
    rad = np.sqrt(((xx - RES / 2) / (RES / 2)) ** 2
                  + ((yy - RES / 2) / (RES / 2)) ** 2)
    vig = np.clip(1.0 - 0.32 * np.clip(rad - 0.45, 0, 1), 0, 1)[..., None]
    arr[..., :3] = (arr[..., :3] * vig).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGBA")
    d = ImageDraw.Draw(img, "RGBA")

    # 淡网格
    g = -1.6
    while g <= 1.6001:
        gx, _ = w2p(g, 0)
        _, gy = w2p(0, g)
        d.line([(gx, 0), (gx, RES)], fill=(*C_GRID_DIM, 110), width=1)
        d.line([(0, gy), (RES, gy)], fill=(*C_GRID_DIM, 110), width=1)
        g += 0.4

    # 等高线（中灰清晰）
    for seg in CONTOUR_SEGS:
        pts = [w2p(x, y) for x, y in seg]
        d.line(pts, fill=(*C_GRID, 165), width=2, joint="curve")

    # 坐标轴（细、暗）
    cx, cy = w2p(0, 0)
    d.line([(cx, 0), (cx, RES)], fill=(*C_GRID_DIM, 170), width=1)
    d.line([(0, cy), (RES, cy)], fill=(*C_GRID_DIM, 170), width=1)

    # 定点 P：白色空心环 + 实心心
    px, py = w2p(PX, PY)
    d.ellipse([px - 7, py - 7, px + 7, py + 7],
              outline=(*C_TEXT, 255), width=2)
    d.ellipse([px - 2.5, py - 2.5, px + 2.5, py + 2.5], fill=(*C_TEXT, 255))
    return img


BASE_IMG = make_base()


def draw_vec(d, x0, y0, dx, dy, color, width=4, head=0.095, alpha=255):
    """世界坐标画向量箭头（w2p 负责 y 翻转）。"""
    p0 = w2p(x0, y0)
    p1 = w2p(x0 + dx, y0 + dy)
    d.line([p0, p1], fill=(*color, alpha), width=width)
    ang = math.atan2(dy, dx)
    for da in (math.radians(148), math.radians(-148)):
        bx = x0 + dx + head * math.cos(ang + da)
        by = y0 + dy + head * math.sin(ang + da)
        d.line([p1, w2p(bx, by)], fill=(*color, alpha), width=width)


def draw_dashed(d, x0, y0, x1, y1, color, dash=0.07, gap=0.05, width=2, alpha=190):
    dist = math.hypot(x1 - x0, y1 - y0)
    if dist < 1e-6:
        return
    ux, uy = (x1 - x0) / dist, (y1 - y0) / dist
    s = 0.0
    while s < dist:
        e = min(s + dash, dist)
        d.line([w2p(x0 + ux * s, y0 + uy * s),
                w2p(x0 + ux * e, y0 + uy * e)],
               fill=(*color, alpha), width=width)
        s = e + gap


class TerrainField(ImageMobject):
    """numpy/PIL 整块位图：静态底图 + 逐帧动态层，原位更新 pixel_array。"""

    def __init__(self, disp_height=3.90):
        super().__init__(np.asarray(BASE_IMG).copy())
        self.scale(disp_height / self.height)

    def redraw(self, t: float):
        arr = self.pixel_array
        arr[:] = np.asarray(BASE_IMG)        # 每帧先清空，防止动态层累积
        img = Image.fromarray(arr, mode="RGBA")
        d = ImageDraw.Draw(img, "RGBA")
        phase, prog = phase_info(t)
        theta = theta_of(t)
        ux, uy = math.cos(PHI_G + theta), math.sin(PHI_G + theta)

        show_grad = phase in ("grad", "dir", "cos", "max", "replay")
        show_dir = phase in ("dir", "cos", "max", "replay")

        if show_grad:
            # 等高线切向虚线：让「垂直」关系一眼可见
            draw_dashed(d, PX - 0.46 * THX, PY - 0.46 * THY,
                        PX + 0.46 * THX, PY + 0.46 * THY,
                        C_GRID, width=2, alpha=205)
            # 梯度箭头（青）
            draw_vec(d, PX, PY, 0.62 * GHX, 0.62 * GHY, C_CYAN, width=4)
            # 直角标记：梯度与切向之间
            s = 0.125
            corner1 = (PX + s * GHX, PY + s * GHY)
            corner2 = (PX + s * GHX + s * THX, PY + s * GHY + s * THY)
            corner3 = (PX + s * THX, PY + s * THY)
            d.line([w2p(*corner1), w2p(*corner2), w2p(*corner3)],
                   fill=(*C_ORANGE, 225), width=2)

        if show_dir:
            # 方向向量 u（红）
            draw_vec(d, PX, PY, 0.62 * ux, 0.62 * uy, C_RED, width=4)
            # 夹角 θ 的世界角弧 + 数值标签
            a0, a1 = PHI_G, PHI_G + theta
            steps = max(6, int(abs(theta) / 0.06))
            arc = [w2p(PX + 0.24 * math.cos(a), PY + 0.24 * math.sin(a))
                   for a in np.linspace(a0, a1, steps)]
            d.line(arc, fill=(*C_ORANGE, 235), width=3)
            am = (a0 + a1) / 2
            lx, ly = w2p(PX + 0.36 * math.cos(am), PY + 0.36 * math.sin(am))
            deg = (theta * 180 / math.pi) % 360
            lbl = "θ=%d°" % int(round(deg))
            font = ImageFont.truetype(_SANS, 17)
            box = d.textbbox((0, 0), lbl, font=font)
            d.text((lx - (box[2] - box[0]) / 2 - box[0],
                    ly - (box[3] - box[1]) / 2 - box[1]),
                   lbl, font=font, fill=(*C_ORANGE, 255))

        # 右下角余弦表：条长 ∝ D_u f / |∇f|，正向右（青）/负向左（红）
        if show_dir:
            gx, gy = RES - 92, RES - 30
            d.line([(gx - 46, gy), (gx + 46, gy)], fill=(*C_GRID_DIM, 210), width=2)
            d.line([(gx, gy - 7), (gx, gy + 7)], fill=(*C_GRID, 255), width=2)
            dv = float(np.clip(math.cos(theta), -1, 1))
            barcol = C_CYAN if dv >= 0 else C_RED
            d.line([(gx, gy), (gx + 44 * dv, gy)], fill=(*barcol, 255), width=7)
            font = ImageFont.truetype(_SANS, 15)
            d.text((gx - 22, gy - 24), "cos θ", font=font,
                   fill=(*C_GRID, 255))

        self.pixel_array[:] = np.asarray(img)


# ---------------------------------------------------------------------------
# 阶段表
# ---------------------------------------------------------------------------
PHASES = [
    ("intro",  0.00,  1.30),
    ("grad",   1.30,  5.60),
    ("dir",    5.60,  9.80),
    ("cos",    9.80, 14.60),
    ("max",   14.60, 17.80),
    ("replay",17.80, DURATION),
]


def phase_info(t: float):
    for name, t0, t1 in PHASES:
        if t < t1:
            if name == "intro":
                return name, 0.0
            return name, float(np.clip((t - t0) / (t1 - t0), 0.0, 1.0))
    return "replay", 1.0


def smoothstep(v):
    v = float(np.clip(v, 0.0, 1.0))
    return v * v * (3 - 2 * v)


def theta_of(t: float) -> float:
    """红向量 u 相对梯度方向的夹角（弧度，逆时针为正）。"""
    phase, prog = phase_info(t)
    if phase in ("intro", "grad"):
        return 0.0
    if phase == "dir":
        return math.radians(55 + 35 * prog)            # 55° → 90°（归零）
    if phase == "cos":
        return math.radians(90 + 90 * prog)            # 90° → 180°（变负）
    if phase == "max":
        return math.radians(180 + 180 * smoothstep(prog))  # 180° → 360°=0°（升到最大）
    return math.radians(360 * prog)                    # 回放：完整扫一圈


# ---------------------------------------------------------------------------
# 场景
# ---------------------------------------------------------------------------
class Gradient(Scene):
    def construct(self):
        # 双语标题（新约定）
        cn_title = cn_text("梯度与方向导数", 50, C_TEXT, 0.30).move_to([0, 4.68, 0])
        en_title = text_img("G R A D I E N T   /   D I R E C T I O N A L   D E R I V A T I V E",
                            18, C_CYAN, 0.11).move_to([0, 4.37, 0])
        watermark = text_img("AIMaker-Lab", 17, C_GRID, 0.09).move_to([3.05, 4.68, 0])
        title = Group(cn_title, en_title, watermark)
        cursor = Line(UP * 0.16, DOWN * 0.16, color=M_TEXT, stroke_width=2.5).move_to([0, 4.68, 0])
        cursor.set_opacity(0)

        # 图例
        leg = Group(
            Group(Line(LEFT * 0.13, RIGHT * 0.13, color=M_CYAN, stroke_width=3),
                  text_img("GRADIENT", 24, C_CYAN, 0.14)).arrange(RIGHT, buff=0.06),
            Group(Line(LEFT * 0.13, RIGHT * 0.13, color=M_RED, stroke_width=3),
                  text_img("DIRECTION U", 24, C_RED, 0.14)).arrange(RIGHT, buff=0.06),
            Group(Line(LEFT * 0.13, RIGHT * 0.13, color=M_ORANGE, stroke_width=3),
                  text_img("ANGLE THETA", 24, C_ORANGE, 0.14)).arrange(RIGHT, buff=0.06),
        ).arrange(RIGHT, buff=0.18).move_to([0, 4.02, 0])

        # 中央等高线地形
        plane = TerrainField(disp_height=3.90).move_to([0, 1.85, 0])

        # 左：梯度模长（固定） × cosθ = 右：方向导数（随角度变）
        # 微软雅黑无 ∇ / ᵤ 字形：中文与数学符号分开渲染后拼接
        lbl_grad = Group(
            cn_text("梯度模长 ", 28, C_CYAN, 0.21),
            mi(r"$|\nabla f|$", "lblgrad", C_CYAN, 0.23, 30),
        ).arrange(RIGHT, buff=0.04).move_to([-2.35, -0.22, 0])
        lbl_dir = Group(
            cn_text("方向导数 ", 28, C_ORANGE, 0.21),
            mi(r"$D_{\mathbf{u}}f$", "lbldir", C_ORANGE, 0.26, 30),
        ).arrange(RIGHT, buff=0.04).move_to([2.35, -0.22, 0])
        num_grad = PixelNumber(GMAG, 3, M_CYAN, 40, nchars=6).set(height=0.30).move_to([-2.35, -0.66, 0])
        num_dir = PixelNumber(0.0, 3, M_ORANGE, 40, nchars=6).set(height=0.30).move_to([2.35, -0.66, 0])
        mul_cos = mi(r"$\times\ \cos\theta\ =$", "mulcos", C_TEXT, 0.27, 31).move_to([0, -0.66, 0])

        # 中文讲述行（预建，靠 move_to 移出画面切换）
        NARR = [
            ("intro",  "等高线地形：每一点都有一个高度 f(x,y)", C_TEXT),
            ("grad",   "梯度向量指向上升最快的方向，并垂直于等高线", C_CYAN),
            ("dir",    "任选单位方向 u，沿它走，高度变化率就是方向导数", C_RED),
            ("cos",    "方向导数 = 梯度模长 × cosθ：反向走，它会变成负值", C_ORANGE),
            ("max",    "把 u 转回梯度方向，变化率升到最大值", C_CYAN),
            ("replay", "梯度给方向，方向导数给速率", C_TEXT),
        ]
        narr_pos = [0, -1.30, 0]
        narr_lbls = {}
        for key, s, col in NARR:
            lb = cn_text(s, 26, col, 0.23).move_to(narr_pos)
            lb.move_to([9, 9, 0])
            narr_lbls[key] = lb

        # 底部固定公式链
        f1 = mi(r"$f(x,y)\quad(\mathrm{height\ on\ the\ map})$",
                "f1", C_FORMULA, 0.27, 30).move_to([0, -1.98, 0])
        f2 = mi(r"$\nabla f=\left(\frac{\partial f}{\partial x},\ \frac{\partial f}{\partial y}\right)$",
                "f2", C_FORMULA, 0.40, 30).move_to([0, -2.60, 0])
        f3 = mi(r"$D_{\mathbf{u}}f=\nabla f\cdot\mathbf{u}\quad(|\mathbf{u}|=1)$",
                "f3", C_FORMULA, 0.36, 30).move_to([0, -3.22, 0])
        f4 = mi(r"$D_{\mathbf{u}}f=|\nabla f|\cos\theta$",
                "f4", C_FORMULA, 0.34, 30).move_to([0, -3.84, 0])
        f5 = mi(r"$\theta=0:\max\quad \theta=\frac{\pi}{2}:0\quad \theta=\pi:\min$",
                "f5", C_FORMULA, 0.36, 30).move_to([0, -4.48, 0])
        formulas = Group(f1, f2, f3, f4, f5)
        FORMULA_COLORS = [f1, f2, f3, f4, f5]
        phase_formula = {"intro": 0, "grad": 1, "dir": 2,
                         "cos": 3, "max": 4, "replay": 4}

        self.add(leg, lbl_grad, lbl_dir, mul_cos)
        dynamic = Group(title, cursor, plane, num_grad, num_dir,
                        *narr_lbls.values(), formulas)
        self.add(dynamic)

        def apply(t):
            phase, prog = phase_info(t)
            plane.redraw(t)

            theta = theta_of(t)
            # 方向导数实时数值 = |∇f| cos θ
            dv = GMAG * math.cos(theta)
            if phase in ("intro", "grad"):
                num_dir.set_value(0.0)      # u 尚未出现
            else:
                num_dir.set_value(dv)
            num_grad.set_value(GMAG)

            # 中文讲述行切换
            for k, lb in narr_lbls.items():
                lb.move_to(narr_pos if k == phase else [9, 9, 0])

            # 公式链：当前行高亮，其余微暗（位置始终固定）
            hi = phase_formula[phase]
            for i, fm in enumerate(FORMULA_COLORS):
                fm.set_opacity(1.0 if i == hi else 0.48)

            # 片头与回放切换处标题闪烁
            blink = (t < 0.28) or (17.72 <= t <= 18.06)
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
