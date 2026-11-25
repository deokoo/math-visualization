#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""斯托克斯公式成片专用配乐 —— 「CIRCULATE-FOctatonic」

系列铁律：每部动画的配乐必须彼此不同（调性 / 音色 / 结构至少换两样）。
本片 vs 已用 9 部配乐（三样全换）：
  - 调性：F 八声音阶（对称减音阶，0-1-3-4-6-7-9-10），此前未用过
  - 音色：克拉维键（毡阻尼击弦 + 轻微 Bebung 颤音）+ 管钟非谐波泛音
          + 低频脉冲波（占空比渐变），三者均为新音色
  - 结构：23/8 拍 + 微复节奏双声部相位漂移 + 变速段落
          + 三角数间隔音序（1,3,6,10,15,21 半音跳进）

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
ROOT = 87.3071                              # F2
SCALE = [0, 1, 3, 4, 6, 7, 9, 10]           # F 八声音阶（对称减）
SEED = 231123
EIGHTH = 0.166                              # 23/8 一小节 ≈ 3.82s
TRIANGULAR = [1, 3, 6, 10, 15, 21]          # 三角数间隔音序

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
]


def pitch(deg):
    semi = SCALE[deg % 8] + 12 * (deg // 8)
    return ROOT * 2 ** (semi / 12)


def make_bgm(duration: float, path: Path) -> None:
    n = int(SR * duration)
    t = np.arange(n) / SR
    audio = np.zeros(n, dtype=np.float64)
    U = t / duration
    rng = np.random.default_rng(SEED)

    # ---- 1) 克拉维键：毡阻尼击弦，双声部微复节奏相位漂移 ----
    for voice, drift in enumerate((1.0, 1.031)):
        tpos = 0.42 + voice * 0.19
        step, deg = 0, 2 + voice * 3
        while tpos < duration - 0.5:
            # 变速段落：整段缓慢加速再减速
            tempo = 1.0 + 0.12 * np.sin(2 * np.pi * (0.62 * U[int(min(tpos, duration - 0.01)) * SR]))
            gap = EIGHTH / drift * tempo
            deg += TRIANGULAR[step % len(TRIANGULAR)] if step % 3 else -2
            deg = int(np.clip(deg, -2, 13))
            f0 = pitch(deg)
            life = min(1.35, gap * 5.2)
            s0, s1 = int(tpos * SR), min(int((tpos + life) * SR), n)
            if s1 <= s0:
                break
            loc = np.arange(s1 - s0) / SR
            # 击弦：软毡起音 + 非谐波部分音 + 轻微 Bebung 颤音
            bev = 1.0 + 0.0016 * np.sin(2 * np.pi * 5.7 * loc)
            body = np.zeros_like(loc)
            for k in range(1, 6):
                fk = f0 * k * (1.0 + 0.00042 * k * k)
                body += ((-1) ** (k + 1) / k ** 1.35) * np.sin(2 * np.pi * fk * np.cumsum(bev) / SR)
            env = np.clip(loc / 0.012, 0, 1) * np.exp(-2.55 * loc) * np.clip((life - loc) / 0.25, 0, 1)
            audio[s0:s1] += 0.038 * env * body
            tpos += gap * (1.0 + 0.008 * np.sin(0.37 * step + voice))
            step += 1

    # ---- 2) 管钟：非谐波泛音，在叙事节点敲响 ----
    BELLS = [1.0, 2.76, 5.40, 8.93, 13.34]
    hits = [0.95, 3.55, 5.62, 7.20, 9.42, 12.10, 15.42, 17.00, 18.82, 20.35]
    for i, ht in enumerate(hits):
        if ht >= duration - 0.4:
            continue
        f0 = pitch([0, 2, 4, 6, 1, 5, 7, 3, 6, 0][i % 10]) * 2
        s0, s1 = int(ht * SR), min(int((ht + 2.2) * SR), n)
        loc = np.arange(s1 - s0) / SR
        ring = np.zeros_like(loc)
        for r, dec in zip(BELLS, (1.5, 2.1, 3.0, 4.2, 5.8)):
            ring += np.sin(2 * np.pi * f0 * r * loc) * np.exp(-dec * loc)
        audio[s0:s1] += 0.046 * ring * np.clip(loc / 0.001, 0, 1)

    # ---- 3) 低频脉冲波：占空比缓变，随进度增强 ----
    duty = 0.32 + 0.15 * U
    ph = (t * ROOT / 2) % 1.0
    pulse = np.where(ph < duty, 1.0, -1.0)
    # 低通掉高频阶梯，只保留脉冲波的低频骨架
    lp = np.zeros(n)
    acc = 0.0
    for i in range(n):
        acc = 0.93 * acc + 0.07 * pulse[i]
        lp[i] = acc
    audio += 0.052 * (0.45 + 0.55 * U) * lp

    # ---- 4) 收尾与归一化 ----
    audio += 0.0008 * rng.normal(size=n)
    fi, fo = int(0.85 * SR), int(1.9 * SR)
    audio[:fi] *= np.linspace(0, 1, fi)
    audio[-fo:] *= np.linspace(1, 0, fo)
    audio /= max(1e-9, float(np.max(np.abs(audio))))
    audio *= 0.80
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((audio * 32767).astype("<i2").tobytes())
    print(f"BGM（CIRCULATE / F-Octatonic，专用签名）: {path} ({duration:.1f}s)")


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
        HERE / "media/videos/stokes_scene/1350p30/StokesTheorem.mp4"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else \
        HERE.parent / "stokes_theorem_final.mp4"
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
