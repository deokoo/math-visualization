# -*- coding: utf-8 -*-
"""
Eight classic dynamical systems, solved live -- chaos gallery dashboard.
1080x1350 (4:5) 30fps ~70s. Layout: row1 = 3 panels, row2 = 3 panels, row3 = 2 panels.
Self-contained: numpy numeric solvers + PIL drawing + matplotlib mathtext PNG.
No LaTeX install, no Manim required.
"""
import math
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W, H = 1080, 1350
FPS = 30
DURATION = 70.0

TOP = 112
ROW_H = [340, 340, 352]
ROW_Y = [TOP, TOP + ROW_H[0] + 12, TOP + ROW_H[0] + ROW_H[1] + 24]
CAPTION_H = 162
MARGIN = 20
GUTTER = 12
NFRAMES = int(FPS * DURATION)

HERE = os.path.dirname(os.path.abspath(__file__))
FORM = os.path.join(HERE, "formulas")
os.makedirs(FORM, exist_ok=True)

BG = (16, 17, 19)
PANEL_BG = (22, 24, 28)
GRID = (52, 56, 63)
TEXT = (214, 219, 226)
DIMTEXT = (130, 136, 145)
CYAN = (80, 214, 224)
RED = (232, 96, 92)
GOLD = (240, 186, 84)
VIOLET = (172, 138, 226)
GREEN = (120, 206, 146)
BLUE = (108, 158, 236)
WHITE = (238, 240, 243)

import matplotlib as _mpl
_MPL_TTF = os.path.join(os.path.dirname(_mpl.__file__), "mpl-data", "fonts", "ttf")
FONT_CN = "C:/Windows/Fonts/msyh.ttc"
FONT_EN = os.path.join(_MPL_TTF, "DejaVuSans.ttf")


# ---------------- formula PNGs (mathtext, no LaTeX) ----------------
FORMULAS = {
    "lorenz": r"$\dot x=\sigma(y-x),\ \dot y=x(\rho-z)-y,\ \dot z=xy-\beta z$",
    "logistic": r"$x_{n+1}=r\,x_n(1-x_n)$",
    "vdp": r"$\ddot x-\mu(1-x^2)\dot x+x=0$",
    "pendulum": r"$\ddot\theta_1=f(\theta_1,\theta_2,\dot\theta_1,\dot\theta_2)$",
    "rossler": r"$\dot x=-y-z,\ \dot y=x+ay,\ \dot z=b+z(x-c)$",
    "henon": r"$x_{n+1}=1-a x_n^2+y_n,\quad y_{n+1}=b x_n$",
    "duffing": r"$\ddot x+\delta\dot x+x^3=\gamma\cos(\omega t)$",
    "mandelbrot": r"$z_{n+1}=z_n^2+c,\quad c\in\mathbb{C}$",
}


def _fname(key):
    return os.path.join(FORM, key + ".png")


def _render_formula(tex, path, size):
    fig = plt.figure(figsize=(0.01, 0.01), dpi=200)
    fig.patch.set_alpha(0)
    fig.text(0, 0, tex, fontsize=size, color="#e6ebf2", va="bottom", ha="left")
    fig.savefig(path, dpi=200, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def ensure_formulas():
    for key, tex in FORMULAS.items():
        if not os.path.exists(_fname(key)):
            _render_formula(tex, _fname(key), 20)


def formula_img(key):
    return Image.open(_fname(key)).convert("RGBA")


# ---------------- numeric solvers ----------------
class Lorenz:
    """Lorenz attractor RK4, sigma=10 rho=28 beta=8/3."""
    def __init__(self):
        s, r, b = 10.0, 28.0, 8.0/3.0
        def f(p):
            x, y, z = p
            return np.array([s*(y-x), x*(r-z)-y, x*y-b*z])
        p = np.array([0.1, 0.0, 0.0])
        dt = 0.006
        n = 2600
        pts = np.empty((n, 3))
        for i in range(n):
            k1 = f(p); k2 = f(p+dt/2*k1); k3 = f(p+dt/2*k2); k4 = f(p+dt*k3)
            p = p + dt/6*(k1+2*k2+2*k3+k4)
            pts[i] = p
        pts = pts[200:]
        self.pts = pts
        self.lo = pts.min(axis=0); self.hi = pts.max(axis=0)


class Logistic:
    """Logistic map bifurcation diagram x_{n+1}=r x_n(1-x_n), r in [2.5,4]."""
    def __init__(self):
        rs = np.linspace(2.5, 4.0, 620)
        pts = []
        for r in rs:
            x = 0.5
            for _ in range(320):
                x = r*x*(1-x)
            for _ in range(34):
                x = r*x*(1-x)
                pts.append((r, x))
        self.pts = np.array(pts)


class VDP:
    """Van der Pol limit cycle, mu=2.5, phase portrait (x, xdot)."""
    def __init__(self):
        mu = 2.5
        def f(s):
            x, v = s
            return np.array([v, mu*(1-x*x)*v - x])
        trajs = []
        for x0, v0 in ((0.2, 0.0), (2.4, 0.0), (-2.6, 1.0), (0.5, 3.0)):
            s = np.array([x0, v0])
            dt = 0.01
            n = 2400
            out = np.empty((n, 2))
            for i in range(n):
                k1 = f(s); k2 = f(s+dt/2*k1); k3 = f(s+dt/2*k2); k4 = f(s+dt*k3)
                s = s + dt/6*(k1+2*k2+2*k3+k4)
                out[i] = s
            trajs.append(out[400:])
        self.trajs = trajs
        self.xmax = max(abs(t[:, 0]).max() for t in trajs)
        self.vmax = max(abs(t[:, 1]).max() for t in trajs)


class DoublePendulum:
    """Chaotic double pendulum RK4, equal masses/lengths."""
    def __init__(self):
        g = 9.81
        def f(s):
            t1, t2, w1, w2 = s
            d = t1 - t2
            den = 2 - math.cos(2*d)
            a1 = (-g*3*math.sin(t1) - g*math.sin(t1-2*d)
                  - 2*math.sin(d)*(w2*w2 + w1*w1*math.cos(d))) / den
            a2 = (2*math.sin(d)*(2*w1*w1 + 2*g*math.cos(t1)
                  + w2*w2*math.cos(d))) / den
            return np.array([w1, w2, a1, a2])
        s = np.array([1.95, 1.6, 0.0, 0.0])
        dt = 0.012
        n = 2600
        out = np.empty((n, 4))
        for i in range(n):
            k1 = f(s); k2 = f(s+dt/2*k1); k3 = f(s+dt/2*k2); k4 = f(s+dt*k3)
            s = s + dt/6*(k1+2*k2+2*k3+k4)
            out[i] = s
        self.s = out


class Rossler:
    """Rossler attractor RK4, a=b=0.2 c=5.7."""
    def __init__(self):
        a, b, c = 0.2, 0.2, 5.7
        def f(p):
            x, y, z = p
            return np.array([-y-z, x+a*y, b+z*(x-c)])
        p = np.array([1.0, 1.0, 1.0])
        dt = 0.02
        n = 2600
        out = np.empty((n, 3))
        for i in range(n):
            k1 = f(p); k2 = f(p+dt/2*k1); k3 = f(p+dt/2*k2); k4 = f(p+dt*k3)
            p = p + dt/6*(k1+2*k2+2*k3+k4)
            out[i] = p
        out = out[200:]
        self.pts = out


class Henon:
    """Henon map strange attractor a=1.4 b=0.3."""
    def __init__(self):
        a, b = 1.4, 0.3
        x, y = 0.1, 0.1
        pts = []
        for i in range(14000):
            x, y = 1-a*x*x+y, b*x
            if i > 100:
                pts.append((x, y))
        self.pts = np.array(pts)


class Duffing:
    """Duffing double-well driven oscillator, phase portrait."""
    def __init__(self):
        delta, gamma, omega = 0.25, 0.30, 1.0
        def f(s, t):
            x, v = s
            return np.array([v, -delta*v + x - x*x*x + gamma*math.cos(omega*t)])
        s = np.array([1.0, 0.0])
        dt = 0.02
        n = 4000
        out = np.empty((n, 2))
        t = 0.0
        for i in range(n):
            k1 = f(s, t); k2 = f(s+dt/2*k1, t+dt/2)
            k3 = f(s+dt/2*k2, t+dt/2); k4 = f(s+dt*k3, t+dt)
            s = s + dt/6*(k1+2*k2+2*k3+k4)
            out[i] = s
            t += dt
        self.s = out[400:]


class Sim:
    def __init__(self):
        self.lorenz = Lorenz()
        self.logistic = Logistic()
        self.vdp = VDP()
        self.pend = DoublePendulum()
        self.rossler = Rossler()
        self.henon = Henon()
        self.duffing = Duffing()


# ---------------- drawing helpers ----------------
def cn_text(im, d, text, xy, size, fill, anchor=None):
    d.text(xy, text, font=ImageFont.truetype(FONT_CN, size), fill=fill, anchor=anchor)


def en_label(d, xy, text, size=12, fill=DIMTEXT, anchor=None):
    d.text(xy, text, font=ImageFont.truetype(FONT_EN, size), fill=fill, anchor=anchor)


def panel_base(d, box):
    d.rounded_rectangle(box, radius=8, fill=PANEL_BG, outline=(44, 48, 54), width=1)


def draw_formula(d, im, key, box):
    x0, y0, x1, y1 = box
    f = formula_img(key)
    maxw, maxh = x1-x0-12, y1-y0-2
    r = min(maxw/f.width, maxh/f.height, 1.0)
    f = f.resize((max(1, int(f.width*r)), max(1, int(f.height*r))), Image.LANCZOS)
    im.paste(f, (int((x0+x1-f.width)/2), int(y0+1)), f)


def axes(d, x0, y0, x1, y1, gy0, color=GRID):
    d.line([(x0+12, (gy0+y1)//2), (x1-12, (gy0+y1)//2)], fill=color, width=1)
    d.line([((x0+x1)//2, gy0+8), ((x0+x1)//2, y1-12)], fill=color, width=1)


def proj3(p, cx, cy, sc, ang, tilt=0.34):
    x, y, z = p
    ca, sa = math.cos(ang), math.sin(ang)
    xr = x*ca - y*sa
    yr = x*sa + y*ca
    return (cx + xr*sc, cy + (yr*tilt - z)*sc)


# ---------------- panels ----------------
def lorenz_panel(im, d, box, sim, t, gy0):
    x0, y0, x1, y1 = box
    cx, cy = (x0+x1)//2, (gy0+y1)//2 + 10
    sc = min((x1-x0)/30, (y1-gy0)/40)
    lo, hi = sim.lorenz.lo, sim.lorenz.hi
    mid = (lo+hi)/2
    pts = sim.lorenz.pts - mid
    ang = 0.22*t
    show = min(len(pts), 700 + int((t % 9)/9*1900))
    step = 1
    prev = None
    for i in range(0, show, step):
        q = proj3(pts[i], cx, cy, sc, ang)
        if prev is not None:
            wing_l = pts[i, 0] < 0
            col = CYAN if wing_l else VIOLET
            if i % 7 == 0:
                col = RED
            d.line([prev, q], fill=col, width=1)
        prev = q
    en_label(d, (x0+12, y1-20), "LORENZ  ·  STRANGE ATTRACTOR", 11, DIMTEXT)


def logistic_panel(im, d, box, sim, t, gy0):
    x0, y0, x1, y1 = box
    px0, px1 = x0+16, x1-16
    py0, py1 = gy0+12, y1-26
    # axes frame
    d.rectangle((px0, py0, px1, py1), outline=GRID, width=1)
    p = sim.logistic.pts
    show_r = 2.5 + (t % 10)/10*1.5
    # always show the full bifurcation structure; the sweep line highlights r
    for r, xv in p:
        X = px0 + (r-2.5)/1.5*(px1-px0)
        Y = py1 - xv*(py1-py0)
        d.point((int(X), int(Y)), fill=(150, 210, 230))
    # sweep line
    sx = px0 + (show_r-2.5)/1.5*(px1-px0)
    d.line([(sx, py0), (sx, py1)], fill=GOLD, width=1)
    en_label(d, (x0+12, y0+38), "r  →", 11, DIMTEXT)
    en_label(d, (x0+12, y1-20), "BIFURCATION  ·  PERIOD DOUBLING", 11, DIMTEXT)


def vdp_panel(im, d, box, sim, t, gy0):
    x0, y0, x1, y1 = box
    cx, cy = (x0+x1)//2, (gy0+y1)//2
    sx = (x1-x0-40)/(2*sim.vdp.xmax)
    sy = (y1-gy0-40)/(2*sim.vdp.vmax)
    axes(d, x0, y0, x1, y1, gy0)
    cols = [CYAN, GOLD, VIOLET, GREEN]
    # limit cycle guide from trajectory 0
    cyc = sim.vdp.trajs[0][-700:]
    d.line([(cx+px*sx, cy-pv*sy) for px, pv in cyc], fill=(70, 90, 96), width=1)
    for k, tr in enumerate(sim.vdp.trajs):
        show = min(len(tr), 400 + int((t % 8)/8*(len(tr)-400)))
        pts = [(cx+tr[i, 0]*sx, cy-tr[i, 1]*sy) for i in range(0, show, 3)]
        d.line(pts, fill=cols[k], width=1)
    # moving dot on cycle
    idx = int(((t*40) % 700))
    px, pv = cyc[idx]
    d.ellipse((cx+px*sx-3, cy-pv*sy-3, cx+px*sx+3, cy-pv*sy+3), fill=WHITE)
    en_label(d, (x0+12, y1-20), "VAN DER POL  ·  LIMIT CYCLE", 11, DIMTEXT)


def pendulum_panel(im, d, box, sim, t, gy0):
    x0, y0, x1, y1 = box
    cx, cy = (x0+x1)//2, gy0+52
    Lpx = min((x1-x0)/5.0, (y1-gy0)/4.6)
    s = sim.pend.s
    idx = int((t*22) % (len(s)-1))
    # trail of tip
    trail = max(0, idx-320)
    pts = []
    for i in range(trail, idx, 3):
        t1, t2 = s[i, 0], s[i, 1]
        x1p = cx + Lpx*math.sin(t1)
        y1p = cy + Lpx*math.cos(t1)
        x2p = x1p + Lpx*math.sin(t2)
        y2p = y1p + Lpx*math.cos(t2)
        pts.append((x2p, y2p))
    if len(pts) > 1:
        d.line(pts, fill=(96, 130, 150), width=1)
    # current rods
    t1, t2 = s[idx, 0], s[idx, 1]
    xa, ya = cx + Lpx*math.sin(t1), cy + Lpx*math.cos(t1)
    xb, yb = xa + Lpx*math.sin(t2), ya + Lpx*math.cos(t2)
    d.line([(cx, cy), (xa, ya)], fill=(200, 204, 212), width=2)
    d.line([(xa, ya), (xb, yb)], fill=(200, 204, 212), width=2)
    d.ellipse((cx-4, cy-4, cx+4, cy+4), fill=(150, 156, 164))
    d.ellipse((xa-5, ya-5, xa+5, ya+5), fill=(200, 206, 214))
    d.ellipse((xb-6, yb-6, xb+6, yb+6), fill=GOLD)
    en_label(d, (x0+12, y1-20), "DOUBLE PENDULUM  ·  CHAOS", 11, DIMTEXT)


def rossler_panel(im, d, box, sim, t, gy0):
    x0, y0, x1, y1 = box
    cx, cy = (x0+x1)//2, (gy0+y1)//2 + 6
    sc = min((x1-x0)/30, (y1-gy0)/44)
    pts = sim.rossler.pts.copy()
    pts[:, :2] -= pts[:, :2].mean(axis=0)
    pts[:, 2] -= pts[:, 2].min()
    ang = -0.25*t
    show = min(len(pts), 650 + int((t % 11)/11*1900))
    prev = None
    for i in range(0, show, 1):
        q = proj3(pts[i], cx, cy, sc, ang, tilt=0.30)
        if prev is not None:
            col = BLUE if i % 9 == 0 else (120, 176, 226)
            d.line([prev, q], fill=col, width=1)
        prev = q
    en_label(d, (x0+12, y1-20), "ROSSLER  ·  FOLDED SPIRAL", 11, DIMTEXT)


def henon_panel(im, d, box, sim, t, gy0):
    x0, y0, x1, y1 = box
    cx, cy = (x0+x1)//2, (gy0+y1)//2
    sx = (x1-x0-40)/3.0
    sy = (y1-gy0-40)/3.0
    axes(d, x0, y0, x1, y1, gy0)
    p = sim.henon.pts
    show = min(len(p), 1200 + int((t % 12)/12*(len(p)-1200)))
    for i in range(0, show, 1):
        X = cx + p[i, 0]*sx
        Y = cy - p[i, 1]*sy
        d.point((int(X), int(Y)), fill=(196, 168, 226))
    en_label(d, (x0+12, y1-20), "HENON MAP  ·  HORSESHOE", 11, DIMTEXT)


def duffing_panel(im, d, box, sim, t, gy0):
    x0, y0, x1, y1 = box
    cx, cy = (x0+x1)//2, (gy0+y1)//2
    sx = (x1-x0-40)/5.0
    sy = (y1-gy0-40)/6.0
    axes(d, x0, y0, x1, y1, gy0)
    s = sim.duffing.s
    show = min(len(s), 800 + int((t % 10)/10*(len(s)-800)))
    pts = [(cx+s[i, 0]*sx, cy-s[i, 1]*sy) for i in range(0, show, 2)]
    d.line(pts, fill=(224, 150, 120), width=1)
    idx = min(len(s)-1, show-1)
    px, pv = s[idx]
    d.ellipse((cx+px*sx-4, cy-pv*sy-4, cx+px*sx+4, cy-pv*sy+4), fill=GOLD)
    en_label(d, (x0+12, y1-20), "DUFFING  ·  DOUBLE WELL", 11, DIMTEXT)


def _mandel_array(x0, x1, y0, y1, w, h, maxiter):
    """Vectorised escape-time array for c in [x0,x1]x[y0,y1]."""
    xs = np.linspace(x0, x1, w)
    ys = np.linspace(y0, y1, h)
    C = xs[None, :] + 1j*ys[:, None]
    Z = np.zeros_like(C)
    it = np.zeros(C.shape, dtype=np.int16)
    alive = np.ones(C.shape, dtype=bool)
    for k in range(maxiter):
        Z[alive] = Z[alive]**2 + C[alive]
        esc = (Z.real*Z.real + Z.imag*Z.imag) > 4
        it[esc & alive] = k
        alive &= ~esc
    it[alive] = maxiter
    return it


def _mandel_color(it, maxiter):
    """Classic blue->cyan->gold ramp; interior (maxiter) near black."""
    h, w = it.shape
    out = np.empty((h, w, 3), dtype=np.uint8)
    interior = it >= maxiter
    q = np.sqrt(np.clip(it.astype(float), 0, maxiter)/maxiter)
    stop = np.array([0.0, 0.28, 0.6, 1.0])
    cols = np.array([[10, 16, 34], [28, 92, 168], [92, 206, 214], [244, 214, 132]])
    for i in range(3):
        out[:, :, i] = np.interp(q, stop, cols[:, i]).astype(np.uint8)
    out[interior] = (8, 10, 14)
    return Image.fromarray(out, "RGB")


def mandelbrot_panel(im, d, box, sim, t, gy0):
    x0, y0, x1, y1 = box
    # Precompute the seahorse valley at high resolution once (numpy-vectorised).
    if not hasattr(sim, "mandel"):
        sim.mandel = _mandel_array(-0.90, -0.64, 0.02, 0.22, 520, 420, 220)
        sim.mandel_maxiter = 220
    arr = sim.mandel
    mx = sim.mandel_maxiter
    h, w = arr.shape
    # progressive zoom into the valley centre
    phase = (t % 16)/16
    zoom = 1.0 + 2.6*phase
    cw, ch = max(60, int(w/zoom)), max(50, int(h/zoom))
    ci, cj = w//2, h//2
    i0 = max(0, min(w-cw, ci-cw//2)); j0 = max(0, min(h-ch, cj-ch//2))
    sub = arr[j0:j0+ch, i0:i0+cw]
    zoomim = _mandel_color(sub, mx)
    ow, oh = x1-x0-24, y1-gy0-40
    zoomim = zoomim.resize((ow, oh), Image.BILINEAR)
    im.paste(zoomim, (x0+12, gy0+14))
    d.rectangle((x0+12, gy0+14, x0+12+ow, gy0+14+oh), outline=GRID, width=1)
    en_label(d, (x0+12, y1-20), "MANDELBROT  ·  SELF-SIMILAR", 11, DIMTEXT)


CAPTIONS = [
    "洛伦兹吸引子：确定性方程，也能画出永不重复的蝴蝶轨迹。",
    "逻辑斯蒂映射：一个参数推上去，周期不断翻倍，直到混沌。",
    "范德波尔振子：不管从哪里出发，最终都落上同一个极限环。",
    "双摆：初始角度只差一点点，轨迹很快就分道扬镳。",
    "罗斯勒吸引子：拉伸再折叠，把轨道卷成螺旋薄片。",
    "埃农映射：反复迭代，散点堆出马蹄形的奇怪吸引子。",
    "杜芬振子：两个势阱之间，周期驱动搅出跳跃与混沌。",
    "曼德博集合：把边界放大，同样的花纹会一次次重现。",
]


# ---------------- frame ----------------
def render_frame(sim, idx):
    t = idx/FPS
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)

    cn_text(im, d, "混沌与动力系统", (W//2, 38), 36, WHITE, anchor="mt")
    en_label(d, (W//2, 82), "C H A O S   ·   D Y N A M I C A L   S Y S T E M S",
             13, DIMTEXT, anchor="mt")
    en_label(d, (W-22, 40), "AIMaker-Lab", 12, (120, 126, 136), anchor="rt")

    specs = [
        ("lorenz",     0, 0, 3, lorenz_panel,   "graph"),
        ("logistic",   0, 1, 3, logistic_panel, "graph"),
        ("vdp",        0, 2, 3, vdp_panel,      "graph"),
        ("pendulum",   1, 0, 3, pendulum_panel, "graph"),
        ("rossler",    1, 1, 3, rossler_panel,  "graph"),
        ("henon",      1, 2, 3, henon_panel,    "graph"),
        ("duffing",    2, 0, 2, duffing_panel,  "graph"),
        ("mandelbrot", 2, 1, 2, mandelbrot_panel, "graph"),
    ]
    for key, row, col, ncols, painter, _ in specs:
        y0 = ROW_Y[row]; rh = ROW_H[row]
        total = W-2*MARGIN-(ncols-1)*GUTTER
        pw = total//ncols
        x0 = MARGIN+col*(pw+GUTTER)
        box = (x0, y0, x0+pw, y0+rh)
        panel_base(d, box)
        draw_formula(d, im, key, (x0, y0, x0+pw, y0+36))
        gy0 = y0 + 44
        painter(im, d, box, sim, t, gy0)

    seg = DURATION/len(CAPTIONS)
    ci = min(int(t/seg), len(CAPTIONS)-1)
    top = H-CAPTION_H
    d.rectangle((0, top, W, H), fill=(20, 22, 26))
    d.line([(0, top), (W, top)], fill=GRID, width=1)
    d.rectangle((MARGIN, top+16, W-MARGIN, top+18), fill=(48, 52, 58))
    d.rectangle((MARGIN, top+16, MARGIN+int((W-2*MARGIN)*idx/NFRAMES), top+18), fill=CYAN)
    cn_text(im, d, CAPTIONS[ci], (W//2, top+34), 20, WHITE, anchor="mt")
    cn_text(im, d, "现场数值求解 · 无预渲染路径", (W//2, top+72), 13, DIMTEXT, anchor="mt")
    en_label(d, (W//2, top+104), "8 SYSTEMS   ·   LIVE INTEGRATION", 12, DIMTEXT, anchor="mt")
    return im


def render_preview():
    sim = Sim()
    out = os.path.join(HERE, "preview_grid")
    os.makedirs(out, exist_ok=True)
    for i in range(21):
        idx = int(i*(NFRAMES-1)/20)
        im = render_frame(sim, idx)
        im.resize((W//2, H//2), Image.LANCZOS).save(os.path.join(out, "k%02d.png" % i))
    print("preview frames:", 21)


def render_full():
    ensure_formulas()
    sim = Sim()
    # warm up mandelbrot cache
    render_frame(sim, 0)
    out = os.path.join(HERE, "chaos_silent.mp4")
    cmd = [os.environ.get("FFMPEG", "ffmpeg"), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H),
           "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
           "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", out]
    import subprocess
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(NFRAMES):
        im = render_frame(sim, i)
        proc.stdin.write(im.tobytes())
        if i % 300 == 0:
            print("frame", i)
    proc.stdin.close()
    proc.wait()
    print("rendered", out)


if __name__ == "__main__":
    import sys
    ensure_formulas()
    if "--preview" in sys.argv:
        render_preview()
    else:
        render_full()
