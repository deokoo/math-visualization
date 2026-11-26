#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""雅可比行列式成片专用配乐 —— 「HARMONIC-FMixRetro」

系列铁律：每部动画的配乐必须彼此不同（调性 / 音色 / 结构至少换两样）。
本片三样全换：
  - 调性：F 混合利底亚（Mixolydian，含 ♭7=E），此前未用
  - 音色：谐波泛音列合成器（1..12 次泛音 1/n^1.45 + 缓慢漂移）、
          音乐锯（连续滑音 + 弓噪）、打字机键敲击（短噪声 + 金属共振），均为新音色
  - 结构：47/8 拍 + 素数乘积间隔（2×3=6、3×5=15、5×7=35、2×7=14 半音）+ 反向变速段落
全程不循环、持续演进。

用法：python add_bgm.py [input.mp4] [output.mp4]
"""
import re
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_OLD = HERE.parent
_CANDIDATES = [HERE / "ffmpeg.exe", _OLD / "manim_gaussian" / "ffmpeg.exe"]
FFMPEG = next((p for p in _CANDIDATES if p.exists()), None)
if FFMPEG is None:
    import imageio_ffmpeg
    FFMPEG = Path(imageio_ffmpeg.get_ffmpeg_exe())
BGM = HERE / "bgm.wav"

SR = 44100
ROOT = 87.3071                              # F2
SCALE = [0, 2, 4, 5, 7, 9, 10]              # F Mixolydian
SEED = 261126
EIGHTH = 0.105
# 素数乘积间隔（半音），逐项递减/回卷，不与旧片的素数间隔或分形音程相同
PRODUCTS = [6, 15, 35, 14, 21, 10, 28, 4]
# 反向变速：每个段落的时间伸缩序列在相邻段落间正反交替
RETRO = [1.00, 0.78, 0.62, 0.50, 0.62, 0.78, 1.00]

_BASE = Path("C:/Users/admin/WorkBuddy")


def discover_old_bgms():
    found = set()
    for d in _BASE.glob("*/bgm*.wav"):
        found.add(d)
    for d in _BASE.glob("*/*/bgm.wav"):
        found.add(d)
    for d in _OLD.glob("manim_*/bgm.wav"):
        found.add(d)
    found.discard(BGM)
    return sorted(found)


OLD_BGMS = discover_old_bgms()


def pitch(deg):
    semi = SCALE[deg % len(SCALE)] + 12 * (deg // len(SCALE))
    return ROOT * 2 ** (semi / 12)


def harmonic_pad(note_f0, loc, amp):
    """谐波泛音列合成：1..12 次泛音，幅度 1/n^1.45，带慢漂移。"""
    out = np.zeros_like(loc)
    for k in range(1, 13):
        drift = 1.0 + 0.0042 * k * np.sin(2 * np.pi * (0.031 * k + 0.07) * loc)
        amp_k = (1.0 / k ** 1.45) * (1.0 + 0.18 * np.sin(2 * np.pi * 0.019 * k * loc))
        out += amp_k * np.sin(2 * np.pi * note_f0 * k * drift * loc + 0.21 * k)
    return amp * out / 2.2


def make_bgm(duration: float, path: Path) -> None:
    n = int(SR * duration)
    t = np.arange(n) / SR
    audio = np.zeros(n, dtype=np.float64)
    U = t / duration
    rng = np.random.default_rng(SEED)

    # ---- 1) 谐波泛音列 pad：47/8 拍、素数乘积间隔、反向变速段落 ----
    pos, deg, step = 0.30, 5, 0
    bar = 47 * EIGHTH
    section = 0
    while pos < duration - 0.9:
        sec_pos = (pos % bar) / bar
        ridx = int(sec_pos * len(RETRO)) % len(RETRO)
        ratio = RETRO[ridx] if section % 2 == 0 else RETRO[-1 - ridx]
        prod = PRODUCTS[step % len(PRODUCTS)]
        deg += max(1, round(prod / 3)) * (1 if (step // len(PRODUCTS)) % 2 == 0 else -1)
        deg = int(np.clip(deg, -4, 30))
        f0 = pitch(deg) * 2
        life = 0.95 + 0.55 * ((step * 3) % 7) / 6
        s0, s1 = int(pos * SR), min(int((pos + life) * SR), n)
        if s1 <= s0:
            break
        loc = np.arange(s1 - s0) / SR
        env = np.clip(loc / 0.22, 0, 1) * np.clip((life - loc) / 0.45, 0, 1) ** 1.2
        audio[s0:s1] += 0.075 * env * harmonic_pad(f0, loc, 1.0)
        pos += EIGHTH * ratio * (2.4 + 0.35 * ((step * 5) % 4))
        step += 1
        section = int(pos / bar)

    # ---- 2) 音乐锯：连续滑音主旋律，音高随进度缓慢上弯再回落 ----
    for k, a in enumerate((3.6, 8.9, 14.2, 18.6)):
        life = 2.1 + 0.6 * (k % 3)
        s0, s1 = int(a * SR), min(int((a + life) * SR), n)
        loc = np.arange(s1 - s0) / SR
        if len(loc) < 10:
            continue
        base = pitch(7 + (k * 3) % 7) * 4
        bend = 1.0 + 0.022 * np.sin(np.pi * loc / life) * (1 if k % 2 == 0 else -1)
        ph = 2 * np.pi * np.cumsum(base * bend * (1 + 0.0035 * np.sin(2 * np.pi * 5.7 * loc))) / SR
        env = np.clip(loc / 0.32, 0, 1) * np.clip((life - loc) / 0.6, 0, 1)
        bow = rng.normal(size=len(loc))
        bow *= 0.18 * np.clip(loc / 0.4, 0, 1)
        audio[s0:s1] += 0.036 * env * (np.sin(ph) + 0.28 * np.sin(2 * ph) + bow)

    # ---- 3) 打字机键敲击：极短噪声脉冲 + 2.7kHz 金属共振 ----
    click_pos = 1.1
    ci = 0
    while click_pos < duration - 0.4:
        life = 0.085
        s0, s1 = int(click_pos * SR), min(int((click_pos + life) * SR), n)
        loc = np.arange(s1 - s0) / SR
        burst = rng.normal(size=len(loc)) * np.exp(-loc * 42.0)
        ring = np.sin(2 * np.pi * (2700 + 180 * (ci % 5)) * loc) * np.exp(-loc * 26.0)
        audio[s0:s1] += 0.040 * (0.6 * burst + 0.4 * ring)
        gap = 0.31 + 0.17 * ((ci * 7) % 5) / 4 + 0.06 * (1 - U[int(click_pos * SR)] if int(click_pos * SR) < n else 0)
        click_pos += gap
        ci += 1

    # ---- 4) 分页式静默：三处短暂留白，随后音量渐回 ----
    gate = np.ones(n)
    for a, b in ((5.05, 5.42), (12.30, 12.70), (18.05, 18.40)):
        s0, s1 = int(a * SR), min(int(b * SR), n)
        gate[s0:s1] = np.linspace(1, 0.015, s1 - s0) * np.linspace(0.015, 1, s1 - s0)
    audio *= gate
    audio *= 0.88 + 0.16 * np.sin(2 * np.pi * 0.071 * U) ** 2

    audio += 0.00030 * rng.normal(size=n)
    fi, fo = int(0.7 * SR), int(1.8 * SR)
    audio[:fi] *= np.linspace(0, 1, fi)
    audio[-fo:] *= np.linspace(1, 0, fo)
    audio /= max(1e-9, float(np.max(np.abs(audio))))
    audio *= 0.80
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((audio * 32767).astype("<i2").tobytes())
    print(f"BGM（HARMONIC / F-Mixolydian-Retro，专用签名）: {path} ({duration:.1f}s)")


def _load_mono(path: Path, limit_s=6.0):
    with wave.open(str(path), "rb") as wf:
        n = min(wf.getnframes(), int(SR * limit_s))
        raw = wf.readframes(n)
    return np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0


def verify(path: Path):
    step = 8
    x = _load_mono(path)[::step]
    print("\n[配乐差异检验]")
    worst = 0.0
    for old in OLD_BGMS:
        y = _load_mono(old)[::step]
        m = min(len(x), len(y))
        a, b = x[:m] - x[:m].mean(), y[:m] - y[:m].mean()
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        worst = max(worst, abs(cos))
        print(f"  vs {old.parent.name}/{old.stem:20s}: {cos:+.4f}")
    print(f"  共检验 {len(OLD_BGMS)} 部旧片，最大 |cos| = {worst:.4f} "
          f"→ {'通过 (<0.1)' if worst < 0.1 else '未通过'}")
    z = x - x.mean()
    ac = np.fft.irfft(np.abs(np.fft.rfft(z)) ** 2)[: len(z)]
    ac /= ac[0] + 1e-12
    lo, hi = int(0.4 * SR / step), len(ac) // 2
    peak = float(np.max(ac[lo:hi]))
    print(f"  自身自相关峰（跳过 <0.4s）: {peak:.3f}  "
          f"→ {'无循环' if peak < 0.8 else '需检查'}")
    return worst, peak


def get_duration(video: Path) -> float:
    out = subprocess.run([str(FFMPEG), "-hide_banner", "-i", str(video)],
                         capture_output=True, text=True, errors="ignore")
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", out.stderr)
    if not m:
        raise RuntimeError("无法读取视频时长")
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        HERE / "media/videos/jacobian_scene/1350p30/JacobianChange.mp4"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else \
        HERE.parent / "jacobian_final.mp4"
    if not src.exists():
        raise SystemExit(f"找不到输入视频: {src}")
    dur = get_duration(src)
    print(f"视频时长: {dur:.2f}s")
    make_bgm(dur, BGM)
    worst, peak = verify(BGM)
    assert worst < 0.1 and peak < 0.8, "配乐差异或无循环检验未通过"
    cmd = [str(FFMPEG), "-y", "-loglevel", "error",
           "-i", str(src), "-i", str(BGM),
           "-map", "0:v:0", "-map", "1:a:0",
           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(dst)]
    subprocess.run(cmd, check=True)
    print(f"完成: {dst} ({dst.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
