#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""曲率与曲率圆成片专用配乐 —— 「REEDWALK-BbLydDom」

系列铁律：每部动画的配乐必须彼此不同（调性 / 音色 / 结构至少换两样）。
本片 vs 已用 10 部配乐（三样全换）：
  - 调性：Bb 吕底亚属（Lydian dominant, 0-2-4-6-7-9-10），此前未用过
  - 音色：双簧管哨片合成（脉冲串激励 + 双 formant 共振 + 呼吸噪声）
          + 低音提琴拨奏 + 纸片摩擦打击，三者均为新音色
  - 结构：29/8 拍 + 随机游走音程 + 分段静默呼吸（5.6-6.2s / 12.4-13.0s）

全程不循环、持续演进；无版权依赖。

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
ROOT = 116.5409                            # Bb2
SCALE = [0, 2, 4, 6, 7, 9, 10]             # Bb Lydian dominant
SEED = 241124
EIGHTH = 0.145                             # 29/8 一小节 ≈ 4.21s
SILENCES = [(5.60, 6.15), (12.40, 12.95)]

OLD_BGMS = [
    _OLD / "manim_taylor" / "bgm.wav",
    _OLD / "manim_brachistochrone" / "bgm.wav",
    _OLD / "manim_cellular" / "bgm.wav",
    _OLD / "manim_dijkstra" / "bgm.wav",
    _OLD / "manim_phase_portrait" / "bgm.wav",
    _OLD / "manim_binary_search" / "bgm.wav",
    _OLD / "manim_green_theorem" / "bgm.wav",
    _OLD / "manim_gradient" / "bgm.wav",
    _OLD / "manim_surface_integral" / "bgm.wav",
    _OLD / "manim_stokes" / "bgm.wav",
]


def pitch(deg):
    semi = SCALE[deg % 7] + 12 * (deg // 7)
    return ROOT * 2 ** (semi / 12)


def make_bgm(duration: float, path: Path) -> None:
    n = int(SR * duration)
    t = np.arange(n) / SR
    audio = np.zeros(n, dtype=np.float64)
    U = t / duration
    rng = np.random.default_rng(SEED)

    # ---- 1) 双簧管：脉冲串激励 + 双 formant 共振 + 呼吸噪声 ----
    tpos, deg = 0.55, 3
    step = 0
    walk = [1, 1, -2, 2, -1, 1, -3, 2, 1, -2, 3, -1, 1, -2, 2, -1]
    while tpos < duration - 1.2:
        deg += walk[step % len(walk)]
        deg = int(np.clip(deg, 0, 13))
        f0 = pitch(deg) * 2
        # 随机游走节奏：以八分音符为单位，逐步放慢
        gap = EIGHTH * (1 + (step % 3)) * (1.06 - 0.18 * U[int(min(tpos, duration - 0.01)) * SR])
        life = min(1.35, gap * 2.9)
        s0, s1 = int(tpos * SR), min(int((tpos + life) * SR), n)
        if s1 <= s0:
            break
        loc = np.arange(s1 - s0) / SR
        env = (np.clip(loc / 0.045, 0, 1) * np.exp(-2.2 * loc)
               * np.clip((life - loc) / 0.22, 0, 1))
        vib = 0.0032 * np.sin(2 * np.pi * 5.1 * loc)
        body = np.zeros_like(loc)
        for k in range(1, 13):
            fk = f0 * k
            form = (1.0 / (1.0 + ((fk - 820) / 300) ** 2)
                    + 0.72 / (1.0 + ((fk - 1750) / 480) ** 2) + 0.04)
            body += form / k ** 0.72 * np.sin(2 * np.pi * fk * (loc + vib * loc))
        breath = rng.normal(size=len(loc))
        # 极轻的高通感：相邻样本差分
        breath = breath[1:] - breath[:-1]
        audio[s0:s1] += env * (0.034 * body + 0.0011 * np.r_[breath, 0.0])
        tpos += gap
        step += 1

    # ---- 2) 低音提琴拨奏：音高轻微下滑 + 二次谐波 ----
    tpos, step = 0.30, 0
    bass_deg = [0, 4, 2, 5, 0, 3, 1, 4, 0, 6, 2, 0]
    while tpos < duration - 0.9:
        f0 = pitch(bass_deg[step % len(bass_deg)]) / 2
        life = 0.92
        s0, s1 = int(tpos * SR), min(int((tpos + life) * SR), n)
        loc = np.arange(s1 - s0) / SR
        env = np.exp(-5.4 * loc) * np.clip(loc / 0.004, 0, 1)
        slide = 0.28 * np.exp(-12 * loc)
        body = (np.sin(2 * np.pi * f0 * (loc + slide) / SR * SR)
                + 0.28 * np.sin(2 * np.pi * 2 * f0 * loc))
        audio[s0:s1] += 0.048 * env * body
        tpos += 4.205 * (0.5 if step % 3 == 2 else 1.0)
        step += 1

    # ---- 3) 纸片摩擦打击：短促带通噪声，落在随机游走节点 ----
    hit = 1.05
    k = 0
    while hit < duration - 0.5:
        s0, s1 = int(hit * SR), min(int((hit + 0.16) * SR), n)
        raw = rng.normal(size=s1 - s0)
        hp = np.r_[raw[1:] - raw[:-1], 0.0]
        env = np.exp(-34 * np.arange(len(hp)) / SR)
        audio[s0:s1] += 0.020 * hp * env
        hit += 0.58 + 0.31 * ((k * 7) % 5) / 4 + 0.11 * np.sin(k * 1.7)
        k += 1

    # ---- 4) 静默呼吸：两段短暂停顿，之后力度回升 ----
    gate = np.ones(n)
    for a, b in SILENCES:
        s0, s1 = int(a * SR), min(int(b * SR), n)
        gate[s0:s1] = np.linspace(1, 0.03, s1 - s0) * np.linspace(0.03, 1, s1 - s0)
    audio *= gate
    audio *= 0.88 + 0.20 * np.sin(2 * np.pi * 0.11 * U) ** 2

    audio += 0.0006 * rng.normal(size=n)
    fi, fo = int(0.9 * SR), int(1.9 * SR)
    audio[:fi] *= np.linspace(0, 1, fi)
    audio[-fo:] *= np.linspace(1, 0, fo)
    audio /= max(1e-9, float(np.max(np.abs(audio))))
    audio *= 0.80
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((audio * 32767).astype("<i2").tobytes())
    print(f"BGM（REEDWALK / Bb-Lydian-dominant，专用签名）: {path} ({duration:.1f}s)")


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
        if not old.exists():
            continue
        y = _load_mono(old)[::step]
        m = min(len(x), len(y))
        a, b = x[:m] - x[:m].mean(), y[:m] - y[:m].mean()
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        worst = max(worst, abs(cos))
        print(f"  vs {old.parent.name:24s}: {cos:+.4f}")
    print(f"  最大 |cos| = {worst:.4f}  → {'通过 (<0.1)' if worst < 0.1 else '未通过'}")
    z = x - x.mean()
    ac = np.fft.irfft(np.abs(np.fft.rfft(z)) ** 2)[: len(z)]
    ac /= ac[0] + 1e-12
    lo, hi = int(0.4 * SR / step), len(ac) // 2
    peak = float(np.max(ac[lo:hi]))
    print(f"  自身自相关峰（跳过 <0.4s）: {peak:.3f}  → {'无循环' if peak < 0.8 else '需检查'}")


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
        HERE / "media/videos/curvature_scene/1350p30/CurvatureCircle.mp4"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else \
        HERE.parent / "curvature_final.mp4"
    if not src.exists():
        raise SystemExit(f"找不到输入视频: {src}")
    dur = get_duration(src)
    print(f"视频时长: {dur:.2f}s")
    make_bgm(dur, BGM)
    verify(BGM)
    cmd = [str(FFMPEG), "-y", "-loglevel", "error",
           "-i", str(src), "-i", str(BGM),
           "-map", "0:v:0", "-map", "1:a:0",
           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(dst)]
    subprocess.run(cmd, check=True)
    print(f"完成: {dst} ({dst.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
