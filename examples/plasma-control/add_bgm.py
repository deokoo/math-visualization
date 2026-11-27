# -*- coding: utf-8 -*-
"""BGM for Plasma State Control (LSTM + EKF fusion) — Bb major additive-beam pad
+ electric-harp metallic bell + data-pulse clicks + golden-ratio interval sequence.
Signature: BEAM-BbAdditive  seed 2711280. Non-looping, continuously evolving.
Avoids all prior signatures (mode / timbres / generation rule all distinct).
"""
import math
import os
import subprocess
import numpy as np

SR = 44100
SEED = 2711280
BGM_SIG = "BEAM-BbAdditive"
ROOT = 116.54  # Bb2

HERE = os.path.dirname(os.path.abspath(__file__))
G = os.path.dirname(HERE)
LIB = os.path.join(G, "2026-11-17-10-23-42", "manim_gaussian")


def adsr(n, a=0.08, d=0.6, s=0.45, r=0.7):
    n = int(n)
    env = np.ones(n, dtype=np.float64)
    tot = a + d + r
    a, d, r = a/tot, d/tot, r/tot
    ai = max(1, int(n*a)); di = max(1, int(n*d)); ri = max(1, int(n*r))
    ai = min(ai, n-2); di = min(di, n-ai-1); ri = min(ri, n-ai-di)
    env[:ai] = np.linspace(0, 1, ai)
    env[ai:ai+di] = np.linspace(1, s, di)
    env[-ri:] = np.linspace(s, 0, ri)
    return env


_R = np.random.default_rng(SEED)


def beam_pad(freq, dur, ph0=0.0, sr=SR):
    """Additive 'beam' pad: 6 integer harmonics with slow independent drift."""
    n = int(dur*sr)
    t = np.arange(n)/sr
    out = np.zeros(n)
    harms = [(1, 0.9), (2, 0.5), (3, 0.32), (4, 0.22), (5, 0.14), (6, 0.09)]
    for h, amp in harms:
        drift = 1 + 0.0012*np.sin(2*np.pi*(0.13+0.07*h)*t + ph0 + h)
        ph = 2*np.pi*freq*h*np.cumsum(drift)/sr + ph0*h
        out += amp*np.sin(ph)
    return out*adsr(n, a=2.5, d=1.0, s=0.7, r=3.5)/2.2


def harp_bell(freq, dur, sr=SR):
    """Electric-harp metallic bell: inharmonic partials + plucked decay."""
    n = int(dur*sr)
    t = np.arange(n)/sr
    out = np.zeros(n)
    for ratio, amp, dec in [(1, 1.0, 1.5), (2.01, 0.4, 2.2), (3.92, 0.22, 2.9), (5.31, 0.12, 3.6)]:
        out += amp*np.sin(2*np.pi*freq*ratio*t + _R.random()*6.28)*np.exp(-t*dec)
    return out*adsr(n, a=0.004, d=0.2, s=0.18, r=0.5)*0.5


def data_pulse(freq, dur, sr=SR):
    """Short square-ish data tick: tanh-shaped pulse with metallic overtone."""
    n = int(dur*sr); t = np.arange(n)/sr
    core = np.tanh(2.5*np.sin(2*np.pi*freq*t))
    over = 0.3*np.sin(2*np.pi*freq*3.1*t)
    return (core+over)*np.exp(-t/0.18)*0.32


def safe_add(buf, seg, i0, gain):
    n = len(buf)
    if i0 >= n or i0 + len(seg) <= 0:
        return
    s0 = max(0, -i0); b0 = max(0, i0)
    m = min(len(seg)-s0, n-b0)
    if m > 0:
        buf[b0:b0+m] += seg[s0:s0+m]*gain


def make_bgm(duration, out_path, sr=SR):
    global _R
    _R = np.random.default_rng(SEED)
    rng = np.random.default_rng(SEED+11)
    n = int(duration*sr)
    buf = np.zeros(n)

    # Bb major: 0 2 4 5 7 9 11 semitones from Bb
    scale = [0, 2, 4, 5, 7, 9, 11]
    PHI = 1.6180339887  # golden ratio -> irrational semitone interval generator

    # additive beam pads: successive transpositions, slow drift, never repeating
    t = 0.0
    ci = 0
    while t < duration:
        deg = scale[(ci*4 + ci//5) % 7]
        f0 = ROOT*2**(deg/12)*2**((ci//6)/12)
        dur = min(12.0, duration-t)
        seg = beam_pad(f0, dur, ph0=0.7*ci)
        safe_add(buf, seg, int(t*sr), 0.32 + 0.18*(0.5+0.5*math.sin(ci*1.17)))
        # a fifth above, quieter, slightly detuned
        seg2 = beam_pad(f0*1.4983, dur, ph0=0.3*ci+1.1)
        safe_add(buf, seg2, int(t*sr), 0.14 + 0.07*math.cos(ci*0.83))
        t += 6.7 + 1.1*math.sin(ci*0.55)
        ci += 1

    # harp bells: golden-ratio interval stream (irrational leap sizes, non-metric)
    t = 0.35
    k = 0
    phase = 0.0
    while t < duration-1.0:
        phase = (phase + PHI) % 7.0
        deg = scale[int(phase) % 7]
        octv = 12*((k//5) % 3)
        f = ROOT*2**((deg+octv)/12)*4
        dur = min(2.6, duration-t)
        safe_add(buf, harp_bell(f, dur), int(t*sr), 0.42)
        t += 0.62 + 0.30*math.sin(k*0.9) + 0.12*(k % 2)
        k += 1

    # data-pulse clicks: EKF-correction-like sparse events, irregular cadence
    t = 0.0
    j = 0
    while t < duration-0.5:
        deg = scale[(j*5) % 7]
        f = ROOT*2**(deg/12)*8
        safe_add(buf, data_pulse(f, 0.6), int(t*sr), 0.30 + 0.10*math.sin(j*0.7))
        t += 1.15 + 0.55*rng.random() + 0.25*math.sin(j*1.31)
        j += 1

    fade = np.ones(n)
    fi = int(1.5*sr); fo = int(2.4*sr)
    fade[:fi] = np.linspace(0, 1, fi)
    fade[-fo:] = np.linspace(1, 0, fo)
    buf *= fade

    spec = np.fft.rfft(buf)
    fr = np.fft.rfftfreq(n, 1/sr)
    shelf = 1/(1 + (fr/5200.0)**2)
    out = np.fft.irfft(spec*shelf, n)
    tt = np.arange(n)/sr
    drift = 0.82 + 0.18*np.sin(2*np.pi*0.037*tt + 0.5)*np.sin(2*np.pi*0.109*tt + 1.4)
    out *= drift
    out = np.tanh(out/np.max(np.abs(out)+1e-12)*1.6)
    out = out/(np.max(np.abs(out))+1e-12)*0.78

    audio = np.stack([out, out], axis=1)
    pcm = (np.clip(audio, -1, 1)*32767).astype("<i2")
    import wave
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return out_path


def verify(path, threshold=0.1):
    import wave
    def load(p):
        with wave.open(str(p), "rb") as w:
            nf = min(w.getnframes(), SR*8)
            nch = w.getnchannels()
            raw = w.readframes(nf)
        x = np.frombuffer(raw, dtype="<i2").astype(np.float64)
        if nch == 2:
            x = x[:nf*2].reshape(-1, 2).mean(axis=1)
        else:
            x = x[:nf]
        return x[::12]
    a = load(path)
    a = a - a.mean(); a /= (np.linalg.norm(a)+1e-12)
    worst = (0.0, "")
    pats = []
    for root_dir in (G, LIB):
        if not os.path.isdir(root_dir):
            continue
        for d in os.listdir(root_dir):
            if d.startswith("manim_") and d != "manim_plasma_control":
                cand = os.path.join(root_dir, d, "bgm.wav")
                if os.path.exists(cand):
                    pats.append(cand)
    pats += [os.path.join(HERE, p) for p in os.listdir(HERE) if p.endswith("_old.wav")]
    for p in pats:
        try:
            b = load(p)
            if b.size < 1000: continue
            m = min(a.size, b.size)
            b = b[:m] - b[:m].mean()
            a2 = a[:m] - a[:m].mean()
            nb = np.linalg.norm(b)+1e-12; na = np.linalg.norm(a2)+1e-12
            c = abs(float(np.dot(a2, b)/(na*nb)))
            if c > worst[0]: worst = (c, os.path.basename(p))
        except Exception:
            pass
    with wave.open(str(path), "rb") as w:
        nch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    full = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    if nch == 2:
        full = full.reshape(-1, 2).mean(axis=1)
    full = full[::12]
    full = full - full.mean()
    corr = np.correlate(full, full, mode="full")
    corr = corr[len(full)-1:]
    den = corr[0]+1e-12
    minlag = int(2.0*SR/12)
    peak = float(np.max(corr[minlag:])/den)
    print("BGM signature:", BGM_SIG)
    print("max |cos| vs library:", round(worst[0], 4), "vs", worst[1])
    print("self-correlation peak:", round(peak, 4))
    ok = worst[0] < threshold and peak < 0.8
    print("VERIFY:", "PASS" if ok else "FAIL")
    return ok


def mix(video, bgm, out):
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", video, "-i", bgm,
           "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
           "-filter:a", "volume=0.85", "-shortest", out]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    import sys
    wav = os.path.join(HERE, "plasma_bgm.wav")
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 70.0
    make_bgm(dur, wav)
    ok = verify(wav)
    if not ok:
        print("BGM verification failed; not mixing.")
        sys.exit(1)
    video = os.path.join(HERE, "plasma_silent.mp4")
    final = os.path.join(HERE, "plasma_final.mp4")
    mix(video, wav, final)
    print("final:", final, os.path.getsize(final))
