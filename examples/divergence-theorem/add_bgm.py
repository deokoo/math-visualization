#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""高斯散度定理成片专用配乐 —— 「CHROMABOW-DSquareTape」

系列铁律：每部动画的配乐必须彼此不同（调性 / 音色 / 结构至少换两样）。
本片三样全换：
  - 调性：D 半音阶（12 音全用，无调式中心），此前未用
  - 音色：方波弦乐合奏（三路微失谐方波 + 扫频低通）+ 磁带饱和（tanh 驱动 +
          wow/flutter 抖晃）+ 弹簧混响（啁啾脉冲卷积），均为新音色
  - 结构：43/8 拍 + 分形自相似音程（1/2/4/8/4/2/1 半音，正反向交替）+ 末段减速
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
ROOT = 73.4162                              # D2
SCALE = list(range(12))                     # D 半音阶
SEED = 251125
EIGHTH = 0.102                              # 43/8 ≈ 4.39s
FRACTAL = [1, 2, 4, 8, 4, 2, 1]             # 分形自相似音程（半音）

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
    semi = SCALE[deg % 12] + 12 * (deg // 12)
    return ROOT * 2 ** (semi / 12)


def spring_reverb(x: np.ndarray, sr: int = SR) -> np.ndarray:
    """弹簧混响：短啁啾脉冲卷积，带明显金属拖尾。"""
    dur = 0.72
    n = int(dur * sr)
    t = np.arange(n) / sr
    f = 360 + 1600 * (t / dur)
    ir = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-5.2 * t)
    ir /= np.linalg.norm(ir) + 1e-12
    nfft = 1 << (int(np.log2(len(x) + len(ir))) + 1)
    out = np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(ir, nfft), nfft)[:len(x)]
    return out


def make_bgm(duration: float, path: Path) -> None:
    n = int(SR * duration)
    t = np.arange(n) / SR
    audio = np.zeros(n, dtype=np.float64)
    U = t / duration
    rng = np.random.default_rng(SEED)

    # ---- 1) 方波弦乐合奏：三路微失谐方波，低通截止随段落扫升 ----
    pos, deg, step = 0.40, 7, 0
    while pos < duration - 1.0:
        deg += FRACTAL[step % len(FRACTAL)] * (1 if (step // len(FRACTAL)) % 2 == 0 else -1)
        deg = int(np.clip(deg, -6, 28))
        f0 = pitch(deg) * 2
        life = 1.15 + 0.42 * ((step * 3) % 5) / 4
        s0, s1 = int(pos * SR), min(int((pos + life) * SR), n)
        if s1 <= s0:
            break
        loc = np.arange(s1 - s0) / SR
        env = np.clip(loc / 0.13, 0, 1) * np.clip((life - loc) / 0.42, 0, 1) ** 1.3
        vib = 0.0018 * np.sin(2 * np.pi * (4.7 + 0.4 * (step % 3)) * loc)
        wave_sum = np.zeros_like(loc)
        for det, amp in ((-0.0041, 1.0), (0.0, 0.75), (0.0037, 0.62)):
            ph = 2 * np.pi * f0 * (1 + det) * (loc + vib * loc)
            wave_sum += amp * np.sign(np.sin(ph))
        # 扫频一阶低通：截止从 700Hz 渐升到 2.1kHz，随音符序号演进
        fc = 700 + 1400 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.043 * pos + step * 0.21))
        alpha = 1.0 - np.exp(-2 * np.pi * fc / SR)
        filt = np.zeros_like(loc)
        prev = 0.0
        for i, v in enumerate(wave_sum):
            prev += alpha * (v - prev)
            filt[i] = prev
        audio[s0:s1] += 0.020 * env * filt
        pos += EIGHTH * (43 / 8) / 1.9 + 0.055 * ((step * 7) % 4)
        step += 1

    # ---- 2) 弹簧混响层（方波弦乐的拖尾，随进度变长）----
    wet = spring_reverb(audio)
    env_wet = 0.12 + 0.16 * U
    audio += env_wet * wet

    # ---- 3) 磁带饱和 + wow/flutter 抖晃 ----
    drive = np.tanh(1.85 * audio)
    wow_t = t + 0.00075 * np.sin(2 * np.pi * 0.37 * t) \
        + 0.00021 * np.sin(2 * np.pi * 0.73 * t + 0.6)
    idx = np.clip(wow_t * SR, 0, n - 1)
    audio = np.interp(idx, np.arange(n), drive)

    # ---- 4) 末段减速 + 呼吸静默（结构不与旧片重复）----
    gate = np.ones(n)
    for a, b in ((6.9, 7.34), (14.6, 15.04)):
        s0, s1 = int(a * SR), min(int(b * SR), n)
        gate[s0:s1] = np.linspace(1, 0.02, s1 - s0) * np.linspace(0.02, 1, s1 - s0)
    audio *= gate
    audio *= 0.90 + 0.14 * np.sin(2 * np.pi * 0.085 * U) ** 2

    audio += 0.00035 * rng.normal(size=n)
    fi, fo = int(0.8 * SR), int(1.7 * SR)
    audio[:fi] *= np.linspace(0, 1, fi)
    audio[-fo:] *= np.linspace(1, 0, fo)
    audio /= max(1e-9, float(np.max(np.abs(audio))))
    audio *= 0.80
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((audio * 32767).astype("<i2").tobytes())
    print(f"BGM（CHROMABOW / D-Chromatic-Square-Tape，专用签名）: {path} ({duration:.1f}s)")


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
        HERE / "media/videos/divergence_scene/1350p30/DivergenceTheorem.mp4"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else \
        HERE.parent / "divergence_theorem_final.mp4"
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
