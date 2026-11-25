#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""曲面积分成片专用配乐 —— 「FLUX-BLocrian」

系列铁律：每部动画的配乐必须彼此不同（调性 / 音色 / 结构至少换两样）。
本片 vs 已用配乐（三样全换）：
  - 泰勒/最速降线 D 小调五声、元胞 A 小调五声、Dijkstra E 小调五声、
    相图 F# 多利亚、二分查找 C 利底亚、格林 C# 全音阶、梯度 G Hijaz
  - 本片：**B 洛克里亚（含减五度，暗色无解决感）**，
           **玻璃琴（旋转玻璃碗的拍频正弦）+ 水琴（非谐波滑音金属共振）
            + THX 深呼吸噪声**，
           **19/8 拍 + Lucas 数间隔音序（2/1/3/4/7/11/18 半音跳进）**。

结构（全程不循环、持续演进）：
  - 玻璃琴：3 声部连续滑音，2f/3f 泛音与慢拍频，力度随进度起伏
  - 水琴：非整数比泛音 [1,1.41,2.13,3.27,4.79] + 下滑音，长衰减
  - THX 深呼吸：粉噪声低通 + 0.07Hz 呼吸包络，极弱铺底
  - 19/8 拍微漂移；无版权依赖

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
_WS = HERE.parent.parent / "2026-11-17-10-23-42"
_CANDIDATES = [HERE / "ffmpeg.exe", _WS / "manim_gaussian" / "ffmpeg.exe"]
FFMPEG = next((p for p in _CANDIDATES if p.exists()), None)
if FFMPEG is None:
    import imageio_ffmpeg
    FFMPEG = Path(imageio_ffmpeg.get_ffmpeg_exe())
BGM = HERE / "bgm.wav"

SR = 44100
ROOT = 123.4708                      # B2
SCALE = [0, 1, 3, 5, 6, 8, 10]       # B 洛克里亚（B C D E F G A）
SEED = 2211221
EIGHTH = 0.172                       # 19/8 一小节 ≈ 3.27s
LUCAS = [2, 1, 3, 4, 7, 11, 18]      # Lucas 数间隔音序

OLD_BGMS = [
    _OLD / "manim_taylor" / "bgm.wav",
    _OLD / "manim_brachistochrone" / "bgm.wav",
    _OLD / "manim_cellular" / "bgm.wav",
    _OLD / "manim_dijkstra" / "bgm.wav",
    _OLD / "manim_phase_portrait" / "bgm.wav",
    _OLD / "manim_binary_search" / "bgm.wav",
    _OLD / "manim_green_theorem" / "bgm.wav",
    _OLD / "manim_gradient" / "bgm.wav",
]


def make_bgm(duration: float, path: Path) -> None:
    n = int(SR * duration)
    t = np.arange(n) / SR
    audio = np.zeros(n, dtype=np.float64)
    U = t / duration
    rng = np.random.default_rng(SEED)

    # ---- 1) 玻璃琴：3 声部连续滑音 + 2f/3f 泛音 + 慢拍频 ----
    for voice in range(3):
        deg = [0, 2, 4][voice]
        tpos = 0.35 + voice * 0.55
        ph = 0.0
        while tpos < duration - 1.0:
            deg = int(np.clip(deg + rng.choice([-2, -1, 1, 1, 2]), 0, 9))
            semi = SCALE[deg % 7] + 12 * (deg // 7)
            f0 = ROOT * 2 * 2 ** (semi / 12)
            life = 1.6 + 0.7 * rng.random()
            s0, s1 = int(tpos * SR), min(int((tpos + life) * SR), n)
            if s1 <= s0:
                break
            loc = np.arange(s1 - s0) / SR
            env = np.clip(loc / 0.22, 0, 1) * np.clip((life - loc) / 0.75, 0, 1)
            glide = f0 * (1.0 - 0.012 * np.exp(-loc / 0.35))
            ph += 2 * np.pi * glide[0] * (life + 0.1)   # 相位连续，避免点击
            phase = 2 * np.pi * np.cumsum(glide) / SR + ph
            body = (np.sin(phase)
                    + 0.24 * np.sin(2 * phase + 0.4)
                    + 0.09 * np.sin(3 * phase + 0.8))
            beat = 0.72 + 0.28 * np.sin(2 * np.pi * (0.11 + 0.013 * voice) * loc)
            audio[s0:s1] += 0.026 * env * beat * body
            tpos += life * (0.88 + 0.08 * np.sin(0.23 * tpos + voice))

    # ---- 2) 水琴：非谐波滑音金属共振，Lucas 间隔触发 ----
    tpos, step, deg = 0.70, 0, 1
    while tpos < duration - 0.6:
        s0, s1 = int(tpos * SR), min(int((tpos + 1.2) * SR), n)
        if s1 <= s0:
            break
        loc = np.arange(s1 - s0) / SR
        deg += LUCAS[step % len(LUCAS)]
        semi = SCALE[deg % 7] + 12 * (deg // 7)
        f = ROOT * 2 ** (semi / 12)
        slide = 1.0 - 0.035 * np.exp(-loc / 0.5)      # 下滑音
        plate = np.zeros_like(loc)
        for ratio, decay in [(1.00, 2.6), (1.41, 3.7), (2.13, 5.1),
                             (3.27, 7.0), (4.79, 9.4)]:
            plate += np.sin(2 * np.pi * f * ratio * np.cumsum(slide) / SR) \
                * np.exp(-decay * loc)
        accent = 1.0 if (step % 19) in (0, 2, 7, 11, 18) else 0.40
        audio[s0:s1] += 0.048 * accent * plate * np.clip(loc / 0.002, 0, 1)
        tpos += EIGHTH * (1 + 0.017 * np.sin(0.31 * step)) * \
            (19 / [1, 1, 2, 3, 5, 7][step % 6])
        step += 1

    # ---- 3) THX 深呼吸噪声：粉噪声低通 + 极慢呼吸包络 ----
    white = rng.normal(size=n)
    pink = np.zeros(n)
    acc = 0.0
    for i in range(n):
        acc = 0.986 * acc + 0.014 * white[i]
        pink[i] = acc
    # 一阶低通让它更沉
    lp = np.zeros(n)
    lp[0] = pink[0]
    for i in range(1, n):
        lp[i] = 0.90 * lp[i - 1] + 0.10 * pink[i]
    breath = 0.5 - 0.5 * np.cos(2 * np.pi * 0.068 * t)
    audio += 0.045 * breath * lp * np.clip((t - 0.5) / 3.0, 0, 1)

    # ---- 4) 收尾与归一化 ----
    audio += 0.0012 * rng.normal(size=n)
    fi, fo = int(0.9 * SR), int(2.0 * SR)
    audio[:fi] *= np.linspace(0, 1, fi)
    audio[-fo:] *= np.linspace(1, 0, fo)
    audio /= max(1e-9, float(np.max(np.abs(audio))))
    audio *= 0.80
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((audio * 32767).astype("<i2").tobytes())
    print(f"BGM（FLUX / B-Locrian，专用签名）: {path} ({duration:.1f}s)")


def _load_mono(path: Path, limit_s=6.0):
    with wave.open(str(path), "rb") as wf:
        n = min(wf.getnframes(), int(SR * limit_s))
        raw = wf.readframes(n)
    return np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0


def verify(path: Path):
    step = 8
    x = _load_mono(path)[::step]
    print("\n[配乐差异检验]")
    for old in OLD_BGMS:
        if not old.exists():
            continue
        y = _load_mono(old)[::step]
        m = min(len(x), len(y))
        a, b = x[:m] - x[:m].mean(), y[:m] - y[:m].mean()
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        print(f"  vs {old.parent.name:22s}: {cos:+.4f}")
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
        HERE / "media/videos/surface_integral_scene/1350p30/SurfaceIntegral.mp4"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else \
        HERE.parent / "surface_integral_final.mp4"
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
