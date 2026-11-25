#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manim 数学动画：二分查找 —— 从五亿条有序数据中，29 步锁定目标

视觉规范与前作一致：
  - 1080x1350 竖屏、30fps、约 21.26s
  - 公式全程固定，只动参数 / 网格 / 动画
  - 无 LaTeX：公式用 matplotlib mathtext 预渲染为透明 PNG
  - 暗色底 #101113、中灰清晰线条、青/红/橙边缘、纯内存动态数字

画面：
  顶部 = 全局数据带（1..500,000,000 抽象为位图），青色=当前候选区间，
         红竖线=本轮中点 mid，橙竖线=目标 T
  中部 = 放大带（把当前区间拉伸铺满），直观展示"再砍一半"
  下部 = 剩余量收缩曲线（对数轴，指数下降）+ 动态数值行
  底部 = 5 行固定公式链

运行：
    manim -pql preview_scene.py BinarySearch
    manim -pqh binary_search_scene.py BinarySearch
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
from PIL import Image, ImageDraw, ImageFont

from manim import *

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
N = 500_000_000
TARGET = 314159265        # π 前 9 位，恰好走满 29 步
MAX_STEPS = 29

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
_CN_FONT = "C:/Windows/Fonts/msyh.ttc"   # 微软雅黑：中文标题/状态
C_DISCARD = (104, 46, 44)                # 被丢弃半区的暗红遮罩
M_DISCARD = "#682E2C"

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
    path = FORMULA_DIR / f"bs_{name}.png"
    if not path.exists():
        _math_img(expr, size, color).save(path)
    m = ImageMobject(str(path))
    if m.height > 0:
        m.scale(height / m.height)
    return m


def text_img(text: str, px: int, color=C_TEXT, height=0.3) -> ImageMobject:
    img = Image.new("RGBA", (1800, 200), (0, 0, 0, 0))
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
    """中文文本：微软雅黑渲染，避免衬线字体缺中文字形。"""
    img = Image.new("RGBA", (1600, 220), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(_CN_FONT, px)
    d.text((20, 16), text, font=font, fill=(*color, 255))
    bbox = img.getchannel("A").getbbox()
    img = img.crop(bbox) if bbox else img
    m = ImageMobject(np.asarray(img))
    if m.height > 0:
        m.scale(height / m.height)
    return m


class PixelNumber(ImageMobject):
    """PIL 渲染到固定 RGBA 数组，set_value 原地更新，零文件 IO。"""

    _FONT = str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf")

    def __init__(self, value=0.0, places: int = 0, color=M_TEXT,
                 px: int = 40, min_chars=None, **kwargs):
        self.places = places
        self.value = value
        self._color = tuple(int(c) for c in ManimColor(color).to_rgb() * 255) \
            if not isinstance(color, tuple) else color
        self._font = ImageFont.truetype(self._FONT, px)
        nchars = min_chars if min_chars else (1 + 1 + places + (1 if places else 0))
        self._cw, self._ch = int(px * (nchars * 0.62 + 0.4)), int(px * 1.6)
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
# 二分查找预演算：每一步一个状态快照，apply 只查表
# ---------------------------------------------------------------------------
def search_events() -> list[dict]:
    lo, hi = 1, N
    evs: list[dict] = []
    step = 0
    while lo <= hi:
        step += 1
        mid = lo + (hi - lo) // 2
        rem = hi - lo + 1
        if mid == TARGET:
            cmp = "hit"
        elif mid < TARGET:
            cmp = "low"       # A[mid] < T，去右半
        else:
            cmp = "high"      # A[mid] > T，去左半
        evs.append(dict(step=step, lo=lo, hi=hi, mid=mid, rem=rem, cmp=cmp))
        if cmp == "hit":
            break
        if cmp == "low":
            lo = mid + 1
        else:
            hi = mid - 1
    return evs


EVENTS = search_events()
NEV = len(EVENTS)
assert NEV == MAX_STEPS, f"步数不符: {NEV}"


def clamp01(v):
    return float(np.clip(v, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 数据带位图：整块 numpy 映射，原地更新 pixel_array
# ---------------------------------------------------------------------------
class DataStrip(ImageMobject):
    def __init__(self, W=1000, H=112, **kwargs):
        self.W, self.H = W, H
        canvas = np.zeros((H, W, 4), dtype=np.uint8)
        canvas[:] = (*C_GRID_DIM, 255)
        super().__init__(canvas, **kwargs)

    def redraw(self, lo, hi, mid, target, zoom=False, discard=None):
        W, H = self.W, self.H
        arr = np.zeros((H, W, 4), dtype=np.uint8)
        a, b = (lo, hi) if zoom else (1, N)

        def px(v):
            return int(np.clip((v - a) / max(1, b - a), 0, 1) * (W - 1))

        x0, x1 = px(lo), px(hi)
        xmid, xT = px(mid), px(target)

        if zoom:
            arr[:] = (*C_CYAN, 92)                       # 当前候选区间铺满放大带
        else:
            arr[:] = (*C_GRID_DIM, 255)                  # 全局带底色
            arr[:, x0:x1 + 1] = (*C_CYAN, 92)            # 候选区间

        # 离散格子线：让数据带看起来像"由元素组成的有序数组"
        step = max(20, W // 44)
        for gx in range(0, W, step):
            arr[8:H - 8, gx:gx + 1] = (*C_GRID, 95)

        # 被丢弃的一半：暗红斜纹（放大带里直观展示"砍半"）
        if zoom and discard in ("low", "high"):
            if discard == "low":       # A[mid]<T → 目标在右 → 丢左半
                d0, d1 = 0, xmid
            else:                      # A[mid]>T → 目标在左 → 丢右半
                d0, d1 = xmid, W - 1
            if d1 > d0:
                arr[:, d0:d1 + 1] = (*C_DISCARD, 135)
                sub = arr[6:H - 6, d0:d1 + 1]
                yy, xx = np.mgrid[0:sub.shape[0], 0:sub.shape[1]]
                sub[((xx + yy) % 24) < 4] = (*C_DISCARD, 225)

        # 候选区间亮边
        if not zoom:
            arr[5:8, x0:x1 + 1] = (*C_CYAN, 235)
            arr[H - 8:H - 5, x0:x1 + 1] = (*C_CYAN, 235)
            arr[:, x0:x0 + 3] = (*C_CYAN, 205)
            arr[:, max(0, x1 - 2):x1 + 1] = (*C_CYAN, 205)
        else:
            arr[5:8, :] = (*C_CYAN, 200)
            arr[H - 8:H - 5, :] = (*C_CYAN, 200)

        # 目标 T：橙色竖线 + 顶部向下三角
        if a <= target <= b:
            arr[14:H - 14, max(0, xT - 1):xT + 2] = (*C_ORANGE, 240)
            for k in range(6):
                arr[7 + k, max(0, xT - k):xT + k + 1] = (*C_ORANGE, 240)

        # 中点 mid：红色竖线 + 底部向上三角（与 T 的指针方向相反，便于区分）
        arr[:, max(0, xmid - 2):xmid + 3] = (*C_RED, 255)
        for k in range(6):
            arr[H - 8 - k, max(0, xmid - k):xmid + k + 1] = (*C_RED, 255)

        self.pixel_array[:] = arr


# ---------------------------------------------------------------------------
# 场景
# ---------------------------------------------------------------------------
class BinarySearch(Scene):
    def construct(self):
        # 标题：中文算法名 + 英文算法标题（新约定：每部动画写明算法/数学标题）
        cn_title = cn_text("二分查找", 52, C_TEXT, 0.34).move_to([0, 4.70, 0])
        en_title = text_img("B I N A R Y   S E A R C H", 26, C_CYAN, 0.13).move_to([0, 4.36, 0])
        watermark = text_img("AIMaker-Lab", 17, C_GRID, 0.09).move_to([3.05, 4.70, 0])
        title = Group(cn_title, en_title, watermark)
        cursor = Line(UP * 0.18, DOWN * 0.18, color=M_TEXT,
                      stroke_width=2.5).move_to([0, 4.70, 0])
        cursor.set_opacity(0)

        # 图例（中文 + 英文，一看就懂颜色含义）
        leg = Group(
            Group(Line(UP * 0.11, DOWN * 0.11, color=M_CYAN, stroke_width=3),
                  cn_text("候选", 24, C_CYAN, 0.13),
                  text_img("RANGE", 21, C_CYAN, 0.10)).arrange(RIGHT, buff=0.05),
            Group(Line(UP * 0.11, DOWN * 0.11, color=M_RED, stroke_width=3),
                  cn_text("中点", 24, C_RED, 0.13),
                  text_img("MID", 21, C_RED, 0.10)).arrange(RIGHT, buff=0.05),
            Group(Line(UP * 0.11, DOWN * 0.11, color=M_ORANGE, stroke_width=3),
                  cn_text("目标", 24, C_ORANGE, 0.13),
                  text_img("TARGET", 21, C_ORANGE, 0.10)).arrange(RIGHT, buff=0.05),
        ).arrange(RIGHT, buff=0.22)
        leg.move_to([0, 4.05, 0])

        # --- 全局数据带 ---
        lbl_global = text_img("GLOBAL ARRAY  /  1 .. 500,000,000", 25, C_GRID, 0.13)
        lbl_global.move_to([0, 3.78, 0])
        strip_g = DataStrip().set(width=7.20).move_to([0, 3.25, 0])

        # --- 放大带 ---
        lbl_zoom = text_img("ZOOM  /  CURRENT CANDIDATE RANGE", 25, C_GRID, 0.13)
        lbl_zoom.move_to([0, 2.58, 0])
        strip_z = DataStrip().set(width=7.20).move_to([0, 2.05, 0])

        # 决策状态标签（四选一，按 cmp 切换不透明度；中文让逻辑一眼看懂）
        st_low = cn_text("A[mid] < T  →  目标在右，丢掉左半", 25, C_ORANGE, 0.15).move_to([0, 1.46, 0])
        st_high = cn_text("A[mid] > T  →  目标在左，丢掉右半", 25, C_ORANGE, 0.15).move_to([0, 1.46, 0])
        st_hit = cn_text("A[mid] = T  →  找到了！", 27, C_CYAN, 0.16).move_to([0, 1.46, 0])
        states = {"low": st_low, "high": st_high, "hit": st_hit}
        for s in states.values():
            s.set_opacity(0)

        # --- 收缩曲线（对数轴，剩余量随步数指数下降）---
        lbl_curve = cn_text("剩余候选数（每步砍一半 · 对数轴）", 23, C_GRID, 0.13)
        lbl_curve.move_to([-2.05, 1.10, 0])

        def curve_pt(k, rem):
            x = -3.55 + (k - 1) / (NEV - 1) * 7.10
            y = 0.24 + (math.log(rem) / math.log(N)) * 0.76
            return np.array([x, y, 0.0])

        ref_points = [curve_pt(e["step"], e["rem"]) for e in EVENTS]
        curve_ref = VMobject(color=M_GRID, stroke_width=2.4)
        curve_ref.set_points_smoothly(ref_points)
        curve_ref.set_opacity(0.85)
        curve_live = VMobject(color=M_CYAN, stroke_width=3.6)
        curve_live.set_points_as_corners([ref_points[0]])
        curve_dot = Dot(color=M_RED, radius=0.075).move_to(ref_points[0])

        # --- 数值行 ---
        def num_row(expr, name, color, value, lab_h=0.18, min_chars=4, px=28, val_h=0.26):
            lab = mi(expr, name, color, lab_h, 25)
            val = PixelNumber(value, 0, color, px, min_chars).set(height=val_h)
            return Group(lab, val).arrange(RIGHT, buff=0.06)

        row_s = num_row(r"$\mathrm{step}=$", "lab_step", M_CYAN, 0, 0.18, 3)
        row_l = num_row(r"$\mathrm{remain}=$", "lab_left", M_ORANGE, N, 0.18, 11)
        row_m = num_row(r"$\mathrm{mid}=$", "lab_mid", M_RED, TARGET, 0.18, 11)
        row_t = num_row(r"$T=$", "lab_target", M_CYAN, TARGET, 0.18, 11)
        # 两行布局：remain= 标签比 left= 长，9 位数单行会把 mid= 标签挤掉
        num_row_grp = Group(
            Group(row_s, row_l).arrange(RIGHT, buff=0.55),
            Group(row_m, row_t).arrange(RIGHT, buff=0.55),
        ).arrange(DOWN, buff=0.16).move_to([0, -0.38, 0])
        num_s, num_l, num_m, num_t = row_s[1], row_l[1], row_m[1], row_t[1]

        # --- 底部固定公式链 ---
        f1 = mi(r"$A[0]\leq A[1]\leq\cdots\leq A[n-1]$",
                "f1", C_FORMULA, 0.27, 30).move_to([0, -1.42, 0])
        f2 = mi(r"$\mathrm{mid}=\left\lfloor\frac{lo+hi}{2}\right\rfloor$",
                "f2", C_FORMULA, 0.32, 30).move_to([0, -2.12, 0])
        f3 = mi(r"$A[\mathrm{mid}]<T\ \Rightarrow\ lo\leftarrow\mathrm{mid}+1$",
                "f3", C_FORMULA, 0.27, 27).move_to([0, -2.82, 0])
        f4 = mi(r"$A[\mathrm{mid}]>T\ \Rightarrow\ hi\leftarrow\mathrm{mid}-1$",
                "f4", C_FORMULA, 0.27, 27).move_to([0, -3.52, 0])
        f5 = mi(r"$\mathrm{steps}\leq\left\lceil\log_2 n\right\rceil=29\quad(n=5\times10^8)$",
                "f5", C_FORMULA, 0.27, 27).move_to([0, -4.24, 0])
        formulas = Group(f1, f2, f3, f4, f5)

        # --- 图层 ---
        self.add(formulas, leg, lbl_global, lbl_zoom, lbl_curve, curve_ref)
        dynamic = Group(title, cursor, strip_g, strip_z,
                        st_low, st_high, st_hit,
                        curve_live, curve_dot, num_row_grp)
        self.add(dynamic)

        def apply(t):
            if t < 16.15:
                p = clamp01((t - 1.5) / 14.65)
            else:
                p = clamp01((t - 16.15) / 4.6)
            idx = min(NEV - 1, int(p * NEV))
            evt = EVENTS[idx]
            lo, hi, mid = evt["lo"], evt["hi"], evt["mid"]

            # 两条数据带（原地更新像素）；放大带把"即将丢弃的一半"画成红色斜纹
            strip_g.redraw(lo, hi, mid, TARGET, zoom=False)
            strip_z.redraw(lo, hi, mid, TARGET, zoom=True, discard=evt["cmp"])

            # 决策状态
            for k, s in states.items():
                s.set_opacity(0.95 if k == evt["cmp"] else 0)

            # 收缩曲线（画到当前步）
            pts = ref_points[:idx + 1]
            if len(pts) >= 2:
                curve_live.set_points_as_corners(pts)
            else:
                curve_live.set_points_as_corners([pts[0], pts[0]])
            curve_dot.move_to(pts[-1])

            # 数值行（必须原地更新，且已在 dynamic 组内）
            num_s.set_value(evt["step"])
            num_l.set_value(evt["rem"])
            num_m.set_value(mid)
            num_t.set_value(TARGET)

            # 循环切换点闪烁
            blink = (t < 0.28) or (16.02 <= t <= 16.36)
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
