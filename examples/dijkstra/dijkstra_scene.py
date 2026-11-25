#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manim 数学动画：Dijkstra 最短路 —— 从源点出发，一步步"锁定"最短距离

视觉规范与前作一致：
  - 1080x1350 竖屏、30fps、约 21.3s
  - 公式全程固定，只动参数 / 网格 / 动画
  - 无 LaTeX：公式用 matplotlib mathtext 预渲染为透明 PNG
  - 暗色底 #101113、中灰清晰线条、青/红/橙边缘、纯内存动态数字

画面：
  顶部 = 8 节点加权图，逐节点 select + 逐边 relax
         节点色：青=已锁定 / 橙=已发现探索中 / 红=当前节点 / 灰=未访问
         边色：青=最短路树 / 橙=正在松弛 / 灰=普通
         每个节点正下方显示当前 d(v)（动态数字，∞ 用大数占位）
  数值行 = |S|（已锁定数）/ relax（松弛次数）/ d(H)（终点距离）/ updates（本步更新数）
  底部 = 5 行固定公式链

运行：
    manim -pql dijkstra_scene.py Dijkstra
    manim -pqh dijkstra_scene.py Dijkstra
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
SRC = 0        # 源点编号
DST = 7        # 终点编号（数值行展示 d(H)，这里 H=7）

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
    path = FORMULA_DIR / f"dj_{name}.png"
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


class PixelNumber(ImageMobject):
    """PIL 渲染到固定 RGBA 数组，set_value 原地更新，零文件 IO。"""

    _FONT = str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf")

    def __init__(self, value: float = 0.0, places: int = 2,
                 color=M_TEXT, px: int = 40, **kwargs):
        self.places = places
        self.value = value
        self._color = tuple(int(c) for c in ManimColor(color).to_rgb() * 255) \
            if not isinstance(color, tuple) else color
        self._font = ImageFont.truetype(self._FONT, px)
        nchars = 1 + 1 + places + (1 if places else 0)
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
# 图数据 + Dijkstra 预演算
# ---------------------------------------------------------------------------
N = 8
NAMES = ["A", "B", "C", "D", "E", "F", "G", "H"]
POS = {
    0: (-3.00, 3.40), 1: (-1.00, 3.80), 2: (1.20, 3.70), 3: (3.00, 3.30),
    4: (-3.00, 1.20), 5: (-1.00, 1.70), 6: (1.20, 1.50), 7: (3.00, 1.10),
}
EDGES = [
    (0, 1, 4), (0, 4, 2), (1, 2, 3), (1, 5, 5),
    (2, 3, 2), (2, 5, 3), (2, 6, 6), (3, 6, 1),
    (3, 7, 5), (4, 5, 1), (5, 6, 7), (6, 7, 4),
]
ADJ = {i: [] for i in range(N)}
for a, b, w in EDGES:
    ADJ[a].append((b, w))
    ADJ[b].append((a, w))

INF = 9999  # 距离标签的 ∞ 占位（PixelNumber 只显示数字）


def dijkstra_events() -> list[dict]:
    """返回完整事件链，每个事件带一个状态快照，apply 只查表。"""
    dist = [INF] * N
    dist[SRC] = 0
    visited: set[int] = set()
    parent: dict[int, int] = {}
    events: list[dict] = []
    nrelax = [0]

    def snap(kind, u=None, v=None, updated=0):
        tree = {frozenset((ch, p)) for ch, p in parent.items()}
        return {
            "kind": kind, "u": u, "v": v, "updated": updated,
            "dist": dist.copy(),
            "visited": set(visited),
            "frontier": {i for i in range(N) if i not in visited and dist[i] < INF},
            "tree": tree,
            "nvisited": len(visited),
            "nrelax": nrelax[0],
        }

    events.append(snap("init"))  # 0：全灰初始化
    while len(visited) < N:
        cand = [i for i in range(N) if i not in visited and dist[i] < INF]
        if not cand:
            break
        u = min(cand, key=lambda i: dist[i])
        visited.add(u)
        events.append(snap("select", u=u))
        for v, w in sorted(ADJ[u], key=lambda p: dist[p[0]]):
            if v in visited:
                continue
            nrelax[0] += 1
            upd = 0
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                parent[v] = u
                upd = 1
            events.append(snap("relax", u=u, v=v, updated=upd))
    events.append(snap("done", u=DST))
    return events


EVENTS = dijkstra_events()
NEV = len(EVENTS)
EDGE_KEY = {frozenset((a, b)): (a, b, w) for a, b, w in EDGES}


def clamp01(v):
    return float(np.clip(v, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 场景
# ---------------------------------------------------------------------------
class Dijkstra(Scene):
    def construct(self):
        # 标题
        title = text_img("AIMaker-Lab", 52, C_TEXT, 0.28).move_to([0, 4.58, 0])
        cursor = Line(UP * 0.16, DOWN * 0.16, color=M_TEXT,
                      stroke_width=2.5).move_to([0, 4.58, 0])
        cursor.set_opacity(0)

        # 图例（大写英文，避免衬线字体缺中文字形）
        leg = Group(
            Group(Dot(color=M_CYAN, radius=0.055), text_img("LOCKED", 26, C_CYAN, 0.13)).arrange(RIGHT, buff=0.06),
            Group(Dot(color=M_ORANGE, radius=0.055), text_img("FRONTIER", 26, C_ORANGE, 0.13)).arrange(RIGHT, buff=0.06),
            Group(Dot(color=M_RED, radius=0.055), text_img("CURRENT", 26, C_RED, 0.13)).arrange(RIGHT, buff=0.06),
        ).arrange(RIGHT, buff=0.30)
        leg.move_to([0, 4.02, 0])

        # --- 边 ---
        edge_lines: dict[frozenset, Line] = {}
        weight_lbls = []
        for a, b, w in EDGES:
            pa, pb = POS[a], POS[b]
            ln = Line([pa[0], pa[1], 0], [pb[0], pb[1], 0],
                      color=M_GRID_DIM, stroke_width=1.6)
            edge_lines[frozenset((a, b))] = ln
            mid = ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2)
            wl = text_img(str(w), 26, C_GRID, 0.135).move_to([mid[0], mid[1] + 0.15, 0])
            weight_lbls.append(wl)

        # --- 节点 ---
        node_circles: dict[int, Circle] = {}
        node_names: dict[int, ImageMobject] = {}
        node_dists: dict[int, PixelNumber] = {}
        for i in range(N):
            x, y = POS[i]
            c = Circle(radius=0.20, color=M_GRID_DIM, stroke_width=1.6)
            c.set_fill(M_GRID_DIM, opacity=0.85).move_to([x, y, 0])
            node_circles[i] = c
            node_names[i] = text_img(NAMES[i], 30, C_TEXT, 0.15).move_to([x, y, 0.01])
            nd = PixelNumber(INF, 0, C_TEXT, 34).set(height=0.19)
            nd.move_to([x, y - 0.36, 0])
            node_dists[i] = nd

        # 当前节点脉动环 + 正在松弛的边高亮
        pulse = Circle(radius=0.30, color=M_RED, stroke_width=2.6)
        pulse.set_fill(opacity=0).set_opacity(0)
        relax_hi = Line(UP, DOWN, color=M_ORANGE, stroke_width=5).set_opacity(0)

        # --- 数值行 ---
        def num_row(expr, name, color, value, lab_h=0.16, places=0, px=28):
            lab = mi(expr, name, color, lab_h, 25)
            val = PixelNumber(value, places, color, px).set(height=0.20)
            return Group(lab, val).arrange(RIGHT, buff=0.05)

        row_s = num_row(r"$|S|=$", "lab_s", M_CYAN, 0, 0.16, 0)
        row_r = num_row(r"$\mathrm{relax}=$", "lab_r", M_ORANGE, 0, 0.16, 0)
        row_d = num_row(r"$d(H)=$", "lab_d", M_RED, INF, 0.16, 0)
        row_u = num_row(r"$\mathrm{updates}=$", "lab_u", M_CYAN, 0, 0.16, 0)
        row_s.move_to([-2.85, -0.16, 0])
        row_r.move_to([-0.95, -0.16, 0])
        row_d.move_to([0.95, -0.16, 0])
        row_u.move_to([2.85, -0.16, 0])
        num_s, num_r, num_d, num_u = row_s[1], row_r[1], row_d[1], row_u[1]
        num_row_grp = Group(row_s, row_r, row_d, row_u)

        # --- 底部固定公式链 ---
        f1 = mi(r"$d(s)=0,\quad d(v)=+\infty\ (v\neq s)$",
                "f1", C_FORMULA, 0.28, 30).move_to([0, -1.65, 0])
        f2 = mi(r"$u=\mathrm{argmin}_{v\notin S}\,d(v)$",
                "f2", C_FORMULA, 0.30, 30).move_to([0, -2.30, 0])
        f3 = mi(r"$d(v)\leftarrow\min\{\,d(v),\,d(u)+w(u,v)\,\}$",
                "f3", C_FORMULA, 0.30, 29).move_to([0, -2.95, 0])
        f4 = mi(r"$S\leftarrow S\cup\{u\}$",
                "f4", C_FORMULA, 0.30, 30).move_to([0, -3.60, 0])
        f5 = mi(r"$\mathrm{at\ end}:\ d(v)=\mathrm{shortest\ distance}$",
                "f5", C_FORMULA, 0.27, 27).move_to([0, -4.25, 0])
        formulas = Group(f1, f2, f3, f4, f5)

        # --- 图层：动态数值行必须在 dynamic 组内，否则被烘焙成静态层不更新 ---
        self.add(formulas, *weight_lbls, leg)
        dynamic = Group(title, cursor, pulse, relax_hi,
                        *edge_lines.values(),
                        *node_circles.values(), *node_names.values(),
                        *node_dists.values(), num_row_grp)
        self.add(dynamic)

        def apply(t):
            if t < 16.15:
                p = clamp01((t - 1.5) / 14.65)
            else:
                p = clamp01((t - 16.15) / 4.6)
            idx = min(NEV - 1, int(p * NEV))
            evt = EVENTS[idx]

            # 边
            for key, ln in edge_lines.items():
                if key in evt["tree"]:
                    ln.set_stroke(M_CYAN, width=2.6, opacity=0.95)
                else:
                    ln.set_stroke(M_GRID_DIM, width=1.6, opacity=0.55)

            # 节点
            for i in range(N):
                c = node_circles[i]
                if evt["kind"] == "select" and i == evt["u"] or \
                   (evt["kind"] in ("relax", "done") and i == evt["u"]):
                    c.set_fill(M_RED, opacity=0.9).set_stroke(M_RED, width=3.0)
                elif i in evt["visited"]:
                    c.set_fill(M_CYAN, opacity=0.85).set_stroke(M_CYAN, width=2.6)
                elif i in evt["frontier"]:
                    c.set_fill(M_ORANGE, opacity=0.75).set_stroke(M_ORANGE, width=2.2)
                else:
                    c.set_fill(M_GRID_DIM, opacity=0.85).set_stroke(M_GRID_DIM, width=1.6)
                dv = evt["dist"][i]
                node_dists[i].set_value("∞" if dv >= INF else dv)

            # 当前节点脉动
            if evt["kind"] in ("select", "relax", "done") and evt["u"] is not None:
                u = evt["u"]
                ph = (p * NEV) % 1.0
                pulse.move_to([POS[u][0], POS[u][1], 0])
                pulse.set_stroke(M_RED, width=2.6, opacity=0.85 * (1 - ph))
                pulse.scale_to_fit_width(0.60 + 0.22 * ph)
            else:
                pulse.set_opacity(0)

            # 正在松弛的边
            if evt["kind"] == "relax" and evt["u"] is not None and evt["v"] is not None:
                a, b = POS[evt["u"]], POS[evt["v"]]
                relax_hi.set_points_by_ends([a[0], a[1], 0], [b[0], b[1], 0])
                flash = 0.95 if evt["updated"] else 0.35
                relax_hi.set_stroke(M_ORANGE, width=5, opacity=flash)
            else:
                relax_hi.set_opacity(0)

            # 数值行
            num_s.set_value(evt["nvisited"])
            num_r.set_value(evt["nrelax"])
            dv_dst = evt["dist"][DST]
            num_d.set_value("∞" if dv_dst >= INF else dv_dst)
            num_u.set_value(evt["updated"])

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
