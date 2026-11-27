#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""拉格朗日中值定理成片专用配乐 ——「VOWELKAL-EPhrygian」

与库内旧片在调性 / 音色 / 节奏结构三样上全部不同：
  - 调性：E 弗里几亚（E F G A B C D，E3=164.81），含特征小二度，此前未用
  - 音色：人声合唱「呜」元音垫（三声部微失谐 + 共振峰加权）+ 拇指琴金属弹片 + 框鼓轻击
  - 结构：41/8 拍，音程取自 π 小数位（1,4,1,5,9,2,6,...）映射到调式级数，全程不循环

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
ROOT = 164.8138                              # E3
SCALE = [0, 1, 3, 5, 7, 8, 10]               # E Phrygian
SEED = 271127
PI_DIGITS = [1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4, 6, 2, 6]

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


def vowel_pad(f0, loc, amp, detune=0.0):
    """合唱「呜」元音：三声部微失谐 + F1/F2/F3 共振峰加权，慢起音。"""
    env = (1.0 - np.exp(-loc / 0.62)) * np.exp(-loc * 0.52)
    vibr = 1.0 + 0.0022 * np.sin(2 * np.pi * 4.7 * loc + detune)
    out = np.zeros_like(loc)
    for k in range(1, 11):
        fk = f0 * k * vibr * (1.0 + detune)
        w = (0.10
             + 0.85 * np.exp(-((fk - 310.0) / 95.0) ** 2)
             + 0.48 * np.exp(-((fk - 880.0) / 150.0) ** 2)
             + 0.24 * np.exp(-((fk - 2250.0) / 260.0) ** 2))
        out += w / (k ** 0.8) * np.sin(2 * np.pi * fk * loc + 0.31 * k)
    return amp * env * out / 3.2


def kalimba(f0, loc, amp):
    """拇指琴：金属弹片非谐波泛音、快速衰减、轻起音。"""
    env = (1.0 - np.exp(-loc / 0.003)) * np.exp(-loc * 3.05)
    out = np.zeros_like(loc)
    for k, ratio in enumerate((1.0, 2.01, 3.98, 5.43), start=1):
        out += (1.0 / k ** 1.25) * np.sin(2 * np.pi * f0 * ratio * loc + 0.19 * k)
    return amp * env * out / 1.65


def frame_drum(loc, amp, rng):
    """框鼓轻击：低通噪声 + 低频皮膜声，短促不刺耳。"""
    x = rng.normal(size=len(loc))
    y = np.zeros_like(x)
    prev = 0.0
    for i, v in enumerate(x):
        prev += 0.28 * (v - prev)
        y[i] = prev
    env = (1.0 - np.exp(-loc / 0.002)) * np.exp(-loc * 19.0)
    thump = np.sin(2 * np.pi * (92.0 - 28.0 * loc) * loc) * np.exp(-loc * 13.0)
    return amp * (0.65 * y * env + 0.35 * thump)


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

    # π 小数位决定每一句的和声根与长度：非周期、无循环
    pos = 0.20
    idx = 0
    while pos < duration - 1.6:
        digit = PI_DIGITS[idx % len(PI_DIGITS)]
        root_deg = digit % 7
        phrase_dur = 1.62 + 0.115 * digit
        # 三声部合唱垫：E 弗里几亚的三度叠置，声部逐渐移调漂移
        for j, deg in enumerate((root_deg, root_deg + 2, root_deg + 4)):
            for voice, det in enumerate((-0.0041, 0.0, 0.0037)):
                add_note(audio, pos + 0.025 * j, min(phrase_dur + 1.15, duration - pos),
                         lambda f, a, d, loc_: vowel_pad(f, loc_, a, d),
                         pitch(deg + (idx % 3)), 0.019, det)
        # 拇指琴：句内四个不规则落点，音高由 π 位推动
        for k in range(4):
            step = PI_DIGITS[(idx + k + 1) % len(PI_DIGITS)]
            deg = root_deg + step
            t0 = pos + 0.18 + k * (0.34 + 0.026 * (step % 5))
            add_note(audio, t0, 1.10,
                     lambda f, a, loc_: kalimba(f, loc_, a),
                     pitch(deg) * 2, 0.043)
        # 框鼓：41/8 拍的非对称重音，仅句首与第三落点
        add_note(audio, pos, 0.30, lambda a, loc_: frame_drum(loc_, a, rng), 0.030)
        add_note(audio, pos + 0.72 + 0.04 * (digit % 3), 0.26,
                 lambda a, loc_: frame_drum(loc_, a, rng), 0.020)
        pos += phrase_dur + 0.10 + 0.015 * (idx % 4)
        idx += 1

    # 低频与高频柔化，避免合唱垫刺耳
    spec = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    spec *= 1.0 / (1.0 + (freqs / 3400.0) ** 4)
    spec *= 1.0 / (1.0 + (70.0 / np.maximum(freqs, 1.0)) ** 2)
    audio = np.fft.irfft(spec, n)

    # 缓慢演进的呼吸包络（不循环、不分页静默）
    t = np.arange(n) / SR
    audio *= 0.88 + 0.12 * np.sin(2 * np.pi * 0.047 * t + 0.9) ** 2

    fi, fo = int(0.7 * SR), int(1.5 * SR)
    audio[:fi] *= np.linspace(0, 1, fi)
    audio[-fo:] *= np.linspace(1, 0, fo)
    audio += 0.00018 * rng.normal(size=n)
    audio /= max(1e-9, float(np.max(np.abs(audio))))
    audio *= 0.74
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((audio * 32767).astype("<i2").tobytes())
    print(f"BGM（VOWELKAL / E-Phrygian）: {path} ({duration:.1f}s)")


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
        HERE / "media/videos/mean_value_scene/1350p30/MeanValueTheorem.mp4"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else \
        HERE.parent / "mean_value_theorem_final.mp4"
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
