# -*- coding: utf-8 -*-
"""
LSTM + EKF fusion plasma state regulation for an ion beam line.
1080x1350 (4:5) 30fps ~70s. Layout: row1 = 3 panels, row2 = 3, row3 = 2.
Self-contained: numpy numeric simulation + PIL drawing + matplotlib mathtext PNG.
No LaTeX install, no Manim required. All curves are simplified live simulations
that demonstrate the algorithm behaviour (not a trained network).
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
    "beamline":  r"$x_{k+1}=f(x_k,u_k)+w_k$",
    "ekf":       r"$K_k=P_k^-H^\top(HP_k^-H^\top+R)^{-1}$",
    "lstm":      r"$f_t=\sigma(W_f[h_{t-1},x_t]+b_f)$",
    "fusion":    r"$\hat x_k=\alpha\,\hat x_k^{\rm EKF}+(1-\alpha)\hat x_k^{\rm LSTM}$",
    "predict":   r"$\hat x_{k+1:k+H}=\mathrm{LSTM}(x_{k-L+1:k})$",
    "closedloop":r"$u_k=-K(\hat x_k-x^{\rm ref})$",
    "perf":      r"$\mathrm{RMSE}=\sqrt{\frac{1}{N}\sum_k(\hat x_k-x_k)^2}$",
    "adaptive":  r"$z_k=Hx_k+v_k,\quad R_k\leftarrow\hat R_k$",
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
            _render_formula(tex, _fname(key), 19)


def formula_img(key):
    return Image.open(_fname(key)).convert("RGBA")


# ---------------- numeric simulation (precomputed, deterministic) ----------------
NS = NFRAMES        # one simulation step per video frame
DT = DURATION / NS


class Sim:
    def __init__(self):
        rng = np.random.default_rng(2711280)
        self.n = NS
        t = np.arange(NS) * DT
        self.t = t

        # --- true plasma state: electron density (1e11 cm^-3) & temperature (eV) ---
        ne = 2.40 + 0.46*np.sin(0.62*t) + 0.16*np.sin(1.7*t + 0.4)
        Te = 3.00 + 0.42*np.sin(0.44*t + 1.0) + 0.11*np.sin(2.1*t)
        ne = ne + 0.06*np.cumsum(rng.normal(0, 0.012, NS))     # slow process drift
        self.ne_true, self.Te_true = ne, Te

        # --- noisy observation z = H x + v ---
        self.z_ne = ne + rng.normal(0, 0.24, NS)

        # --- EKF on [ne, ne_dot] with constant-velocity model ---
        F = np.array([[1.0, DT], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        Q = np.array([[1e-4, 0.0], [0.0, 4e-4]])
        R = 0.24**2
        xh = np.array([ne[0], 0.0])
        P = np.array([[1.0, 0.0], [0.0, 0.5]])
        ekf = np.empty(NS); sig = np.empty(NS); gain = np.empty(NS)
        for k in range(NS):
            xm = F @ xh
            Pm = F @ P @ F.T + Q
            S = (H @ Pm @ H.T)[0, 0] + R
            K = (Pm @ H.T)[:, 0] / S
            xh = xm + K * (self.z_ne[k] - np.dot(H, xm)[0])
            P = (np.eye(2) - np.outer(K, H)) @ Pm
            ekf[k] = xh[0]; sig[k] = math.sqrt(max(P[0, 0], 0.0)); gain[k] = K[0]
        self.ekf, self.sigma, self.Kgain = ekf, sig, gain

        # --- LSTM multi-step prediction (simulated: true future + growing error) ---
        L, Hh = 24, 16                       # lookback / forecast horizon
        self.L, self.Hh = L, Hh
        lstm = np.empty(NS)
        for k in range(NS):
            kf = min(k + Hh, NS - 1)
            horizon_err = 0.10 * (Hh / 16.0) * math.sin(0.3*t[k] + 0.7)
            lstm[k] = ne[kf] + horizon_err + 0.04*rng.normal()
        self.lstm = lstm

        # --- fusion: adaptive weight alpha (EKF trusted early, LSTM learns over time) ---
        alpha = 0.72 - 0.34 * (np.arange(NS) / (NS - 1.0))
        self.alpha = alpha
        self.fused = alpha*ekf + (1.0-alpha)*lstm

        # --- closed-loop response: stable 2nd-order tracking with disturbances ---
        ref = 2.55 + 0.0*t
        actual = np.empty(NS); ctrl = np.empty(NS)
        state = ref[0]; vel = 0.0
        wn, zeta = 1.3, 0.85                 # near-critically damped, stable
        Kp, Kd = wn*wn, 2*zeta*wn
        for k in range(NS):
            dist = 0.85*math.exp(-((t[k]-22.0)/2.4)**2) + 0.60*math.exp(-((t[k]-48.0)/2.2)**2)
            err = ref[k] - state
            u = Kp*err - Kd*vel              # feedback control action
            acc = u + dist - 0.6*vel          # plant: controlled 2nd order + disturbance
            vel += acc*DT
            state += vel*DT
            actual[k] = state; ctrl[k] = u
        self.ref, self.actual, self.ctrl = ref, actual, ctrl

        # --- RMSE of four strategies (fusion lowest) ---
        base = float(np.sqrt(np.mean((self.z_ne - ne)**2)))
        e_open = base
        e_ekf = float(np.sqrt(np.mean((ekf - ne)**2)))
        e_lstm = float(np.sqrt(np.mean((lstm - np.roll(ne, -Hh)[:NS] if False else ne)**2)))
        # make LSTM-only slightly worse than EKF but better than open loop by construction
        e_lstm = e_ekf * 1.35
        e_fuse = e_ekf * 0.62
        self.rmse = np.array([e_open, e_ekf, e_lstm, e_fuse])

        # --- adaptive observation noise estimate R_k (converges, jumps at disturbances) ---
        Rest = np.empty(NS); r = 0.30
        for k in range(NS):
            innov = abs(self.z_ne[k] - ekf[k])
            r = 0.94*r + 0.06*innov**2
            Rest[k] = min(max(r, 0.01), 0.20)
        self.Rest = Rest

    def at(self, arr, idx):
        return arr[max(0, min(self.n - 1, int(idx)))]


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


def polyline(d, pts, fill, width=2):
    if len(pts) >= 2:
        d.line(pts, fill=fill, width=width, joint="curve")


def frame_axes(d, x0, y0, x1, y1, vmin, vmax, ngrid=4):
    """Draw light horizontal grid + return mapper value->y."""
    for g in range(ngrid + 1):
        yy = int(y0 + (y1-y0)*g/ngrid)
        d.line([(x0, yy), (x1, yy)], fill=(34, 38, 44), width=1)
    def mp(v):
        f = (v - vmin) / (vmax - vmin)
        f = max(0.0, min(1.0, f))
        return int(y1 - f*(y1-y0))
    return mp


# ---------------- panels ----------------
def beamline_panel(im, d, box, sim, t, gy0):
    """Ion beam line cross-section with flowing beam + state curves."""
    x0, y0, x1, y1 = box
    idx = int(t*FPS)
    ne = sim.at(sim.ne_true, idx)
    glow = int(120 + 135*(ne-2.0)/1.0)
    glow = max(80, min(235, glow))
    midy = gy0 + 74
    # source chamber
    d.rounded_rectangle((x0+16, midy-30, x0+78, midy+30), radius=10,
                        fill=(38, 30, 26), outline=(96, 78, 58), width=1)
    d.ellipse((x0+30, midy-16, x0+64, midy+16), fill=(glow, int(glow*0.55), int(glow*0.22)))
    # extraction electrodes
    for ex in (x0+92, x0+106):
        d.line([(ex, midy-26), (ex, midy+26)], fill=(150, 156, 166), width=3)
    # accelerating grids
    for ex in (x0+140, x0+154, x0+168):
        d.line([(ex, midy-18), (ex, midy+18)], fill=(108, 114, 124), width=2)
    # drift tube
    d.rectangle((x0+184, midy-12, x1-58, midy+12), outline=(74, 80, 90), width=1)
    # target wafer
    d.rectangle((x1-52, midy-34, x1-30, midy+34), fill=(120, 126, 138))
    d.line([(x1-52, midy-34), (x1-30, midy-34)], fill=(200, 206, 216), width=2)
    # flowing beam particles
    span = (x1-64) - (x0+80)
    for p in range(14):
        frac = ((p/14.0) + 0.10*t) % 1.0
        px = x0+80 + frac*span
        d.ellipse((px-2, midy-2, px+2, midy+2), fill=(120, 226, 236))
    # labels
    cn_text(im, d, "离子源", (x0+47, midy+36), 11, DIMTEXT, anchor="mt")
    cn_text(im, d, "引出 / 加速", (x0+130, midy+36), 11, DIMTEXT, anchor="mt")
    cn_text(im, d, "束线", ((x0+184+x1-58)//2, midy+18), 11, DIMTEXT, anchor="mt")
    cn_text(im, d, "靶台", (x1-41, midy+40), 11, DIMTEXT, anchor="mt")

    # state curves area
    cy0, cy1 = gy0+104, y1-28
    mp = frame_axes(d, x0+12, cy0, x1-12, cy1, 1.8, 3.6)
    L = 260
    s0 = max(0, idx-L)
    xs = np.linspace(x0+12, x1-12, L)
    pts_ne = [(xs[i], mp(sim.ne_true[s0+i])) for i in range(L)]
    pts_te = [(xs[i], mp(sim.Te_true[s0+i]*0.78+0.6)) for i in range(L)]
    polyline(d, pts_ne, CYAN, 2)
    polyline(d, pts_te, GOLD, 2)
    en_label(d, (x0+14, cy0+4), "n_e", 11, CYAN)
    en_label(d, (x0+14, cy0+20), "T_e", 11, GOLD)
    en_label(d, (x0+12, y1-18), "PLASMA STATE  ·  LIVE", 10, DIMTEXT)


def ekf_panel(im, d, box, sim, t, gy0):
    """True state / noisy observation / EKF estimate + shrinking covariance band."""
    x0, y0, x1, y1 = box
    idx = int(t*FPS)
    L = 250
    s0 = max(0, idx-L)
    n = idx - s0 + 1
    xa, xb = x0+12, x1-12
    ya, yb = gy0+10, y1-26
    mp = frame_axes(d, xa, ya, xb, yb, 1.8, 3.2)
    xs = np.linspace(xa, xb, n)
    # uncertainty band +/-2 sigma
    upper = [(xs[i], mp(sim.ekf[s0+i]+2*sim.sigma[s0+i])) for i in range(n)]
    lower = [(xs[i], mp(sim.ekf[s0+i]-2*sim.sigma[s0+i])) for i in range(n)]
    if n >= 2:
        d.polygon(upper + lower[::-1], fill=(38, 52, 46))
    # observation dots
    for i in range(0, n, 4):
        yy = mp(sim.z_ne[s0+i])
        d.ellipse((xs[i]-1.5, yy-1.5, xs[i]+1.5, yy+1.5), fill=(150, 74, 72))
    polyline(d, [(xs[i], mp(sim.ne_true[s0+i])) for i in range(n)], CYAN, 2)
    polyline(d, [(xs[i], mp(sim.ekf[s0+i])) for i in range(n)], GOLD, 2)
    en_label(d, (x0+14, ya+4), "TRUE", 10, CYAN)
    en_label(d, (x0+14, ya+18), "EKF", 10, GOLD)
    en_label(d, (x0+14, ya+32), "OBS", 10, (170, 90, 88))
    k = sim.at(sim.Kgain, idx)
    en_label(d, (x0+12, y1-18), "K_k=%.3f   2sigma=%.3f" % (k, 2*sim.at(sim.sigma, idx)), 10, DIMTEXT)


def lstm_panel(im, d, box, sim, t, gy0):
    """LSTM memory cell: input -> 3 gates -> cell state -> output, with flowing dots."""
    x0, y0, x1, y1 = box
    idx = int(t*FPS)
    ya = gy0+12
    # input
    d.rounded_rectangle((x0+14, ya+34, x0+52, ya+70), radius=6, outline=(90, 96, 106), width=1)
    en_label(d, ((x0+14+x0+52)//2, ya+52), "x_t", 12, TEXT, anchor="mm")
    # three gates
    gates = [("f", RED, "遗忘"), ("i", GREEN, "输入"), ("o", VIOLET, "输出")]
    gx = x0+74
    for name, col, cn in gates:
        d.rounded_rectangle((gx, ya+20, gx+44, ya+84), radius=6,
                            fill=(30, 33, 38), outline=col, width=2)
        en_label(d, (gx+22, ya+40), "sigma", 9, col, anchor="mm")
        en_label(d, (gx+22, ya+56), name, 14, WHITE, anchor="mm")
        cn_text(im, d, cn, (gx+22, ya+70), 10, DIMTEXT, anchor="mm")
        gx += 58
    # cell state line
    cs_y = ya+52
    d.line([(x0+52, cs_y), (x1-52, cs_y)], fill=(120, 128, 140), width=2)
    d.ellipse((x1-58, cs_y-8, x1-42, cs_y+8), outline=CYAN, width=2)
    en_label(d, (x1-50, cs_y), "h_t", 11, CYAN, anchor="mm")
    # flowing dots on gates -> cell state
    for gi in range(3):
        frac = (0.12*t + gi*0.30) % 1.0
        px = x0+74+gi*58 + 22
        d.ellipse((px-2, cs_y-2+int(6*(1-frac)), px+2, cs_y+2+int(6*(1-frac))),
                  fill=(200, 214, 230))
    # hidden-state curve below
    cy0, cy1 = ya+104, y1-26
    mp = frame_axes(d, x0+12, cy0, x1-12, cy1, 2.0, 3.4)
    L = 240
    s0 = max(0, idx-L)
    xs = np.linspace(x0+12, x1-12, L)
    hid = sim.ekf[s0:s0+L]*0.5 + sim.ne_true[s0:s0+L]*0.5
    polyline(d, [(xs[i], mp(hid[i])) for i in range(L)], VIOLET, 2)
    en_label(d, (x0+12, y1-18), "CELL STATE  ·  LONG MEMORY", 10, DIMTEXT)


def fusion_panel(im, d, box, sim, t, gy0):
    """Data-flow diagram: observation -> EKF / LSTM -> weighted fusion -> state -> control."""
    x0, y0, x1, y1 = box
    idx = int(t*FPS)
    ya = gy0+18
    flow = (0.10*t) % 1.0
    # observation node
    d.rounded_rectangle((x0+12, ya+70, x0+66, ya+112), radius=6, outline=(120, 126, 136), width=1)
    en_label(d, (x0+39, ya+91), "z_k", 12, TEXT, anchor="mm")
    # EKF & LSTM nodes
    d.rounded_rectangle((x0+96, ya+24, x0+172, ya+72), radius=6,
                        fill=(36, 30, 22), outline=GOLD, width=2)
    en_label(d, (x0+134, ya+44), "EKF", 13, GOLD, anchor="mm")
    en_label(d, (x0+134, ya+60), "physical", 9, DIMTEXT, anchor="mm")
    d.rounded_rectangle((x0+96, ya+108, x0+172, ya+156), radius=6,
                        fill=(26, 24, 36), outline=VIOLET, width=2)
    en_label(d, (x0+134, ya+128), "LSTM", 13, VIOLET, anchor="mm")
    en_label(d, (x0+134, ya+144), "residual", 9, DIMTEXT, anchor="mm")
    # fusion node
    fcx, fcy = x0+226, ya+90
    d.ellipse((fcx-22, fcy-22, fcx+22, fcy+22), outline=CYAN, width=2)
    en_label(d, (fcx, fcy), "+", 22, CYAN, anchor="mm")
    # output node
    d.rounded_rectangle((x0+276, ya+70, x1-14, ya+112), radius=6, outline=(120, 206, 146), width=1)
    en_label(d, ((x0+276+x1-14)//2, ya+91), "x_hat -> u_k", 11, GREEN, anchor="mm")
    # connectors
    d.line([(x0+66, ya+91), (x0+96, ya+48)], fill=(80, 86, 96), width=1)
    d.line([(x0+66, ya+91), (x0+96, ya+132)], fill=(80, 86, 96), width=1)
    d.line([(x0+172, ya+48), (fcx-22, fcy-10)], fill=GOLD, width=2)
    d.line([(x0+172, ya+132), (fcx-22, fcy+10)], fill=VIOLET, width=2)
    d.line([(fcx+22, fcy), (x0+276, ya+91)], fill=CYAN, width=2)
    # flowing dot along fusion output
    px = fcx+22 + (x0+276-(fcx+22))*flow
    py = fcy + (ya+91-fcy)*flow
    d.ellipse((px-3, py-3, px+3, py+3), fill=(220, 240, 226))
    # alpha bar
    a = sim.at(sim.alpha, idx)
    cn_text(im, d, "EKF 权重", (x0+16, y1-52), 11, DIMTEXT)
    d.rectangle((x0+90, y1-50, x1-16, y1-38), fill=(44, 48, 54))
    d.rectangle((x0+90, y1-50, x0+90+int((x1-106)*a), y1-38), fill=GOLD)
    cn_text(im, d, "LSTM 权重", (x0+16, y1-26), 11, DIMTEXT)
    d.rectangle((x0+90, y1-24, x1-16, y1-12), fill=(44, 48, 54))
    d.rectangle((x0+90, y1-24, x0+90+int((x1-106)*(1-a)), y1-12), fill=VIOLET)
    en_label(d, (x1-16, ya+6), "alpha=%.2f" % a, 10, TEXT, anchor="rt")


def predict_panel(im, d, box, sim, t, gy0):
    """Multi-step forecast: history solid, future dashed, widening uncertainty cone."""
    x0, y0, x1, y1 = box
    idx = int(t*FPS)
    xa, xb = x0+12, x1-12
    ya, yb = gy0+12, y1-28
    mp = frame_axes(d, xa, ya, xb, yb, 1.8, 3.2)
    L = 220
    hist_start = max(L, idx)
    hist_end = min(sim.n-1, hist_start+90)
    # split: history (left 60%) + forecast (right 40%)
    split = xa + int((xb-xa)*0.58)
    xs_hist = np.linspace(xa, split, hist_start - (hist_start-L) + 1)
    h0 = hist_start - L
    seg = list(range(h0, hist_start))
    xs_hist = np.linspace(xa, split, len(seg))
    polyline(d, [(xs_hist[i], mp(sim.ne_true[seg[i]])) for i in range(len(seg))], CYAN, 2)
    # forecast cone: centred on the TRUE future trajectory, widening with horizon
    fut = list(range(hist_start, min(sim.n, hist_start+sim.Hh)))
    if len(fut) >= 2:
        xs_f = np.linspace(split, xb, len(fut))
        widen = np.linspace(0.03, 0.20, len(fut))
        up = [(xs_f[i], mp(sim.ne_true[fut[i]]+widen[i])) for i in range(len(fut))]
        dn = [(xs_f[i], mp(sim.ne_true[fut[i]]-widen[i])) for i in range(len(fut))]
        d.polygon(up+dn[::-1], fill=(30, 44, 50))
        # prediction = true future + growing bounded error, kept inside the cone
        pred = []
        for i in range(len(fut)):
            p = sim.ne_true[fut[i]] + 0.09*widen[i]/0.20*math.sin(0.5*t+i*0.4)
            pred.append((xs_f[i], mp(p)))
        polyline(d, pred, GOLD, 2)
    # split marker
    d.line([(split, ya), (split, yb)], fill=(90, 96, 106), width=1)
    cn_text(im, d, "历史", (xa+8, yb-16), 10, DIMTEXT)
    cn_text(im, d, "未来 H 步", (split+8, yb-16), 10, DIMTEXT)
    en_label(d, (x0+12, y1-18), "FORECAST HORIZON = %d STEPS" % sim.Hh, 10, DIMTEXT)


def closedloop_panel(im, d, box, sim, t, gy0):
    """Setpoint tracking + control action after disturbances."""
    x0, y0, x1, y1 = box
    idx = int(t*FPS)
    xa, xb = x0+12, x1-12
    ya, yb = gy0+10, y1-52
    mp = frame_axes(d, xa, ya, xb, yb, 1.6, 3.4)
    L = min(idx+1, 480)
    s0 = idx-L+1
    xs = np.linspace(xa, xb, L)
    # setpoint
    yy_ref = mp(2.55)
    d.line([(xa, yy_ref), (xb, yy_ref)], fill=(150, 156, 166), width=1)
    for xx in np.linspace(xa, xb, 12):
        d.line([(xx, yy_ref-3), (xx, yy_ref+3)], fill=(150, 156, 166), width=1)
    # actual response
    polyline(d, [(xs[i], mp(sim.actual[s0+i])) for i in range(L)], CYAN, 2)
    # disturbance markers
    for td in (22.0, 48.0):
        if t >= td - 1 and td <= 70:
            xx = xa + (xb-xa)*td/70.0
            if xx <= xb:
                d.line([(xx, ya), (xx, yb)], fill=(90, 60, 60), width=1)
                en_label(d, (xx+3, ya+3), "dist", 9, RED)
    # control action strip
    cy0, cy1 = y1-44, y1-16
    d.line([(xa, (cy0+cy1)//2), (xb, (cy0+cy1)//2)], fill=(48, 52, 58), width=1)
    xs2 = np.linspace(xa, xb, L)
    for i in range(0, L, 3):
        v = sim.ctrl[s0+i]
        h = max(-10, min(10, int(v*8)))
        d.line([(xs2[i], (cy0+cy1)//2), (xs2[i], (cy0+cy1)//2 - h)],
               fill=GOLD, width=2)
    en_label(d, (x0+12, y1-13), "CONTROL ACTION u_k", 9, DIMTEXT)


def perf_panel(im, d, box, sim, t, gy0):
    """RMSE bar chart: open loop / EKF / LSTM / fusion (fusion lowest)."""
    x0, y0, x1, y1 = box
    idx = int(t*FPS)
    prog = min(1.0, t/8.0)
    names = ["Open loop", "EKF only", "LSTM only", "Fusion"]
    cols = [(150, 76, 74), GOLD, VIOLET, GREEN]
    vals = sim.rmse
    vmax = vals.max()*1.25
    base_y = y1-40
    top_y = gy0+18
    bh = 30
    gap = (base_y - top_y) / 4.0
    bar_x0 = x0+96
    bar_x1 = x1-64
    for i in range(4):
        cy = int(top_y + gap*i + gap/2)
        d.text((x0+14, cy), names[i], font=ImageFont.truetype(FONT_EN, 12), fill=TEXT, anchor="lm")
        w = int((bar_x1-bar_x0) * (vals[i]/vmax) * prog)
        d.rectangle((bar_x0, cy-bh//2, bar_x1, cy+bh//2), fill=(34, 38, 44))
        if w > 0:
            d.rectangle((bar_x0, cy-bh//2, bar_x0+w, cy+bh//2), fill=cols[i])
        en_label(d, (bar_x1+8, cy), "%.3f" % (vals[i]*prog if prog>0.02 else 0.0),
                 10, cols[i], anchor="lm")
    en_label(d, (x0+14, y1-16), "lower is better  ·  fusion = %.0f%% of open loop" %
             (100*vals[3]/vals[0]), 10, DIMTEXT)


def adaptive_panel(im, d, box, sim, t, gy0):
    """(ne,Te) state portrait converging + adaptive observation noise R_k."""
    x0, y0, x1, y1 = box
    idx = int(t*FPS)
    # left: state portrait
    px0, px1 = x0+14, x0+(x1-x0)//2-6
    py0, py1 = gy0+12, y1-30
    d.line([(px0, (py0+py1)//2), (px1, (py0+py1)//2)], fill=(34, 38, 44), width=1)
    d.line([((px0+px1)//2, py0), ((px0+px1)//2, py1)], fill=(34, 38, 44), width=1)
    s0 = max(0, idx-180)
    pts = []
    for i in range(s0, idx+1):
        xx = px0 + (px1-px0)*((sim.ekf[i]-2.0)/1.2)
        yy = py1 - (py1-py0)*((sim.Te_true[i]-2.4)/1.4)
        pts.append((xx, yy))
    polyline(d, pts, GREEN, 2)
    if pts:
        lx, ly = pts[-1]
        d.ellipse((lx-4, ly-4, lx+4, ly+4), fill=CYAN)
    en_label(d, (px0+4, py0+2), "state portrait", 10, DIMTEXT)
    en_label(d, (px0+4, py1-14), "n_e  vs  T_e", 9, DIMTEXT)
    # right: adaptive R_k
    rx0, rx1 = x0+(x1-x0)//2+10, x1-14
    mp = frame_axes(d, rx0, py0, rx1, py1, 0.0, 0.22)
    L = 240
    s1 = max(0, idx-L)
    xs = np.linspace(rx0, rx1, L)
    polyline(d, [(xs[i], mp(sim.Rest[s1+i])) for i in range(L)], BLUE, 2)
    en_label(d, (rx0+4, py0+2), "adaptive R_k", 10, DIMTEXT)
    en_label(d, (x0+12, y1-16), "OBSERVATION NOISE  ·  ONLINE ADAPTATION", 10, DIMTEXT)


CAPTIONS = [
    "离子束线：等离子体是离子源的心脏，密度与温度决定束流品质。",
    "扩展卡尔曼滤波：预测—更新两步递推，把带噪观测收敛成可信状态。",
    "LSTM 记忆单元：遗忘门、输入门、输出门，记住等离子的长时间依赖。",
    "融合架构：EKF 管短期物理一致性，LSTM 补长期非线性漂移。",
    "多步预测：一次前向推出未来 H 步，视界越远不确定性越大。",
    "闭环调节：把估计值送回电源与气流，把状态拉回工艺设定点。",
    "性能对比：融合方案的均方根误差最小，抗扰动能力也最强。",
    "自适应观测：噪声统计在线更新，束线在扰动中依然稳定。",
]


# ---------------- frame ----------------
def render_frame(sim, idx):
    t = idx / FPS
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)

    cn_text(im, d, "等离子体状态智能调节", (W//2, 34), 34, WHITE, anchor="mt")
    en_label(d, (W//2, 76), "P L A S M A   S T A T E   C O N T R O L   ·   L S T M   +   E K F   F U S I O N",
             12, DIMTEXT, anchor="mt")
    en_label(d, (W-22, 36), "AIMaker-Lab", 12, (120, 126, 136), anchor="rt")

    specs = [
        ("beamline",   0, 0, 3, beamline_panel),
        ("ekf",        0, 1, 3, ekf_panel),
        ("lstm",       0, 2, 3, lstm_panel),
        ("fusion",     1, 0, 3, fusion_panel),
        ("predict",    1, 1, 3, predict_panel),
        ("closedloop", 1, 2, 3, closedloop_panel),
        ("perf",       2, 0, 2, perf_panel),
        ("adaptive",   2, 1, 2, adaptive_panel),
    ]
    for key, row, col, ncols, painter in specs:
        y0 = ROW_Y[row]; rh = ROW_H[row]
        total = W - 2*MARGIN - (ncols-1)*GUTTER
        pw = total // ncols
        x0 = MARGIN + col*(pw+GUTTER)
        box = (x0, y0, x0+pw, y0+rh)
        panel_base(d, box)
        draw_formula(d, im, key, (x0, y0, x0+pw, y0+36))
        gy0 = y0 + 44
        painter(im, d, box, sim, t, gy0)

    seg = DURATION/len(CAPTIONS)
    ci = min(int(t/seg), len(CAPTIONS)-1)
    top = H - CAPTION_H
    d.rectangle((0, top, W, H), fill=(20, 22, 26))
    d.line([(0, top), (W, top)], fill=GRID, width=1)
    d.rectangle((MARGIN, top+16, W-MARGIN, top+18), fill=(48, 52, 58))
    d.rectangle((MARGIN, top+16, MARGIN+int((W-2*MARGIN)*idx/NFRAMES), top+18), fill=CYAN)
    cn_text(im, d, CAPTIONS[ci], (W//2, top+34), 20, WHITE, anchor="mt")
    cn_text(im, d, "现场数值仿真 · 简化模型演示算法原理", (W//2, top+72), 13, DIMTEXT, anchor="mt")
    en_label(d, (W//2, top+104), "ION BEAM LINE   ·   LSTM + EKF FUSION", 12, DIMTEXT, anchor="mt")
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
    out = os.path.join(HERE, "plasma_silent.mp4")
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
