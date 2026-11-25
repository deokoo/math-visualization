#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""拉普拉斯变换成片专用配乐 —— 「CHORALIN-AbMajBlues」

系列铁律：每部动画的配乐必须彼此不同（调性 / 音色 / 结构至少换两样）。
本片 vs 已用 12 部配乐（三样全换）：
  - 调性：Ab 大调布鲁斯（0-4-5-7-11，含 blue note ♭5），此前未用过
  - 音色：低音单簧管（奇次谐波 + 气声）+ 钟琴槌击（非谐波金属条）
          + 合唱元音合成（三 formant 缓慢漂移），均为新音色
  - 结构：37/8 拍 + 倒计数间隔（大跳逐步收窄）+ 变速静默呼吸

注意：并发生成的「猴鞍曲面」占用了手风琴 + 吉他泛音 + 陶笛 + 31/8 质数间隔，
本片已刻意避开该组合，并将其 BGM 纳入差异检验。全程不循环、持续演进。

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
ROOT = 103.8262                            # Ab2
SCALE = [0, 4, 5, 7, 11]                   # Ab 大调布鲁斯
SEED = 2411243
EIGHTH = 0.118                             # 37/8 一小节 ≈ 4.37s
COUNTDOWN = [11, 7, 5, 3, 2]               # 倒计数间隔：大跳逐步收窄
SILENCES = [(7.15, 7.72), (15.10, 15.62)]

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
    _OLD / "manim_curvature" / "bgm.wav",
    Path(r"C:/Users/admin/WorkBuddy/2026-11-24-09-37-30/bgm_monkey_saddle.wav"),
]


def pitch(deg):
    semi = SCALE[deg % 5] + 12 * (deg // 5)
    return ROOT * 2 ** (semi / 12)


def make_bgm(duration: float, path: Path) -> None:
    n = int(SR * duration)
    t = np.arange(n) / SR
    audio = np.zeros(n, dtype=np.float64)
    U = t / duration
    rng = np.random.default_rng(SEED)

    # ---- 1) 低音单簧管：奇次谐波 + 气声，缓慢连奏 ----
    tpos, deg, step = 0.42, 1, 0
    while tpos < duration - 1.4:
        deg -= COUNTDOWN[step % len(COUNTDOWN)] % 3
        deg = int(np.clip(deg, -2, 12))
        f0 = pitch(deg)
        life = 1.65 + 0.28 * ((step * 2) % 4)
        s0, s1 = int(tpos * SR), min(int((tpos + life) * SR), n)
        if s1 <= s0:
            break
        loc = np.arange(s1 - s0) / SR
        env = np.clip(loc / 0.16, 0, 1) * np.clip((life - loc) / 0.34, 0, 1)
        vib = 0.0029 * np.sin(2 * np.pi * 4.2 * loc)
        body = np.zeros_like(loc)
        for k in range(1, 12, 2):  # 只取奇次谐波：单簧管音色
            body += ((-1) ** ((k - 1) // 2)) / k ** 0.9 * np.sin(
                2 * np.pi * f0 * k * (loc + vib * loc))
        air = rng.normal(size=len(loc))
        air = np.r_[air[1:] - air[:-1], 0.0]
        audio[s0:s1] += env * (0.031 * body + 0.0008 * air)
        tpos += 0.52 + 0.09 * (step % 3)
        step += 1

    # ---- 2) 钟琴槌击：非谐波金属条 [1, 2.45, 4.03, 6.12] ----
    hit, k = 0.95, 0
    while hit < duration - 0.5:
        f0 = pitch([3, 0, 4, 2, 5, 1][k % 6]) * 4
        life = 0.85
        s0, s1 = int(hit * SR), min(int((hit + life) * SR), n)
        loc = np.arange(s1 - s0) / SR
        env = np.exp(-7.5 * loc) * np.clip(loc / 0.002, 0, 1)
        ring = np.zeros_like(loc)
        for r, dec in zip((1.0, 2.45, 4.03, 6.12), (3.0, 4.4, 6.2, 8.5)):
            ring += np.sin(2 * np.pi * f0 * r * loc) * np.exp(-dec * loc)
        audio[s0:s1] += 0.030 * env * ring
        hit += 0.82 + 0.29 * ((k * 3) % 5) / 4
        k += 1

    # ---- 3) 合唱元音合成：三 formant 缓慢漂移 + 微失谐声部 ----
    oc = 0.25
    while oc < duration - 2.2:
        f0 = pitch([2, 4, 0, 3][int(oc * 2) % 4]) * 2
        life = 3.1
        s0, s1 = int(oc * SR), min(int((oc + life) * SR), n)
        loc = np.arange(s1 - s0) / SR
        env = np.clip(loc / 0.42, 0, 1) * np.clip((life - loc) / 0.62, 0, 1)
        f1 = 520 + 160 * np.sin(2 * np.pi * 0.13 * loc)
        f2 = 1480 + 220 * np.sin(2 * np.pi * 0.09 * loc + 1.1)
        f3 = 2530 + 120 * np.sin(2 * np.pi * 0.07 * loc)
        voice = np.zeros_like(loc)
        for det in (-0.0034, 0.0031):
            ph = 2 * np.pi * f0 * (1 + det) * loc
            for k in range(1, 15):
                fk = f0 * k
                w = (0.8 / (1 + ((fk - f1) / 180) ** 2)
                     + 0.55 / (1 + ((fk - f2) / 260) ** 2)
                     + 0.25 / (1 + ((fk - f3) / 320) ** 2))
                voice += w / k * np.sin(k * ph)
        audio[s0:s1] += 0.014 * env * voice
        oc += 3.55

    # ---- 4) 变速静默呼吸 ----
    gate = np.ones(n)
    for a, b in SILENCES:
        s0, s1 = int(a * SR), min(int(b * SR), n)
        gate[s0:s1] = np.linspace(1, 0.02, s1 - s0) * np.linspace(0.02, 1, s1 - s0)
    audio *= gate
    audio *= 0.88 + 0.18 * np.sin(2 * np.pi * 0.12 * U) ** 2

    audio += 0.0005 * rng.normal(size=n)
    fi, fo = int(0.85 * SR), int(1.8 * SR)
    audio[:fi] *= np.linspace(0, 1, fi)
    audio[-fo:] *= np.linspace(1, 0, fo)
    audio /= max(1e-9, float(np.max(np.abs(audio))))
    audio *= 0.80
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((audio * 32767).astype("<i2").tobytes())
    print(f"BGM（CHORALIN / Ab-Major-Blues，专用签名）: {path} ({duration:.1f}s)")


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
            print(f"  vs {old.stem:24s}: 文件不存在，跳过")
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
        HERE / "media/videos/laplace_scene/1350p30/LaplaceTransform.mp4"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else \
        HERE.parent / "laplace_final.mp4"
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
