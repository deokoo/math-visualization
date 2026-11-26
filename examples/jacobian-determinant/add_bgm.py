#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""雅可比行列式成片专用配乐（用户反馈重做版）——「MELLOW-EbMixRhodes」

用户反馈初版（谐波泛音列 + 音乐锯 + 打字机敲击）不好听，改为暖色、柔和、
有旋律感的版本；调性 / 音色 / 结构三样全部更换，仍满足「每片配乐不同」铁律：
  - 调性：Eb 混合利底亚（Eb2=77.78），此前未用
  - 音色：FM 电钢琴（Rhodes，带颤音与微合唱）+ 竖琴泛音琶音 + 软刷镲，均为新音色
  - 结构：5/7/11/6/9/13/8/10 非对称乐句分组 + 逐乐句移调漂移，无循环、无机械节拍
全程持续演进，无打字机噪声、无音乐锯滑音。

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
ROOT = 77.7817                              # Eb2
SCALE = [0, 2, 4, 5, 7, 9, 10]              # Eb Mixolydian
SEED = 2611262
PHRASES = [5, 7, 11, 6, 9, 13, 8, 10]       # 非对称乐句长度（音数），不循环

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


def rhodes(f0, loc, amp, rng):
    """温暖 FM 电钢琴：衰减调制指数 + 双声部微合唱 + 慢颤音。"""
    env = (1.0 - np.exp(-loc / 0.012)) * np.exp(-loc * 1.55)
    mod = np.sin(2 * np.pi * (2.0 * f0) * loc + 0.35) * np.exp(-loc * 3.2)
    carrier = np.sin(2 * np.pi * f0 * loc + 1.55 * mod)
    chorus = (np.sin(2 * np.pi * f0 * 1.0027 * loc)
              + np.sin(2 * np.pi * f0 * 0.9973 * loc)) * 0.5
    trem = 0.90 + 0.10 * np.sin(2 * np.pi * 4.3 * loc + 0.6)
    return amp * env * trem * (0.62 * carrier + 0.38 * chorus)


def harp(f0, loc, amp):
    """竖琴泛音琶音：轻快起音、指数衰减、轻微非谐波。"""
    env = (1.0 - np.exp(-loc / 0.005)) * np.exp(-loc * 2.6)
    out = np.zeros_like(loc)
    for k in range(1, 5):
        fk = f0 * k * (1.0 + 0.0006 * k * k)
        out += (1.0 / k ** 1.7) * np.sin(2 * np.pi * fk * loc + 0.17 * k)
    return amp * env * out / 1.55


def bass(f0, loc, amp):
    env = (1.0 - np.exp(-loc / 0.02)) * np.exp(-loc * 1.1)
    return amp * env * (np.sin(2 * np.pi * f0 * loc)
                        + 0.22 * np.sin(4 * np.pi * f0 * loc))


def brush(loc, amp, rng):
    """软刷镲：低幅度噪声，经一阶平滑去掉刺耳高频。"""
    x = rng.normal(size=len(loc))
    y = np.zeros_like(x)
    prev = 0.0
    for i, v in enumerate(x):
        prev += 0.17 * (v - prev)
        y[i] = prev
    env = np.clip(loc / 0.09, 0, 1) * np.exp(-loc * 5.0)
    return amp * y * env


def add_note(audio, start, dur, fn, *args):
    s0, s1 = int(start * SR), min(int((start + dur) * SR), len(audio))
    if s1 <= s0:
        return
    loc = np.arange(s1 - s0) / SR
    audio[s0:s1] += fn(*args, loc)


def make_bgm(duration: float, path: Path) -> None:
    n = int(SR * duration)
    audio = np.zeros(n, dtype=np.float64)
    rng = np.random.default_rng(SEED)

    # 非对称乐句：每句一个调式根音，铺 Rhodes 和弦 + 竖琴琶音 + 低音 + 软刷镲
    pos = 0.25
    pi = 0
    root_deg = 0
    while pos < duration - 1.2:
        plen = PHRASES[pi % len(PHRASES)]
        chord = [root_deg + d for d in (0, 2, 4, 6)]
        chord_life = min(1.55 + 0.09 * (plen % 5), duration - pos)
        for k, deg in enumerate(chord):
            add_note(audio, pos + 0.045 * k, chord_life,
                     lambda f, a, loc_: rhodes(f, loc_, a, rng),
                     pitch(deg) * 2, 0.030)
        add_note(audio, pos, min(1.8, duration - pos),
                 lambda f, a, loc_: bass(f, loc_, a), pitch(root_deg), 0.070)
        # 琶音向上再折返，节奏在句内逐步收紧
        arp = chord[1:] + chord[-2:0:-1]
        for j, deg in enumerate(arp):
            gap = 0.19 - 0.014 * j + 0.004 * (j % 3)
            t0 = pos + 0.16 + j * gap
            add_note(audio, t0, 1.25,
                     lambda f, a, loc_: harp(f, loc_, a),
                     pitch(deg) * 4, 0.052)
        add_note(audio, pos, 0.55,
                 lambda a, loc_: brush(loc_, a, rng), 0.012)
        pos += plen * 0.235 + 0.30 + 0.045 * (pi % 4)
        root_deg += (5 if pi % 3 == 0 else 3) % len(SCALE)
        root_deg = int(np.clip(root_deg, -3, 24))
        pi += 1

    # 柔和高频滚降：避免刺耳，保留暖色
    spec = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    spec /= 1.0 + (freqs / 2600.0) ** 4
    audio = np.fft.irfft(spec, n)

    # 缓慢演进的呼吸包络（不是固定和弦循环，也不是分页静默）
    t = np.arange(n) / SR
    audio *= 0.90 + 0.10 * np.sin(2 * np.pi * 0.055 * t + 0.3) ** 2

    fi, fo = int(0.8 * SR), int(1.6 * SR)
    audio[:fi] *= np.linspace(0, 1, fi)
    audio[-fo:] *= np.linspace(1, 0, fo)
    audio += 0.00020 * rng.normal(size=n)
    audio /= max(1e-9, float(np.max(np.abs(audio))))
    audio *= 0.72
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((audio * 32767).astype("<i2").tobytes())
    print(f"BGM（MELLOW / Eb-Mixolydian-Rhodes，重做版）: {path} ({duration:.1f}s)")


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
