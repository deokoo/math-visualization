# -*- coding: utf-8 -*-
"""BGM for Seven PDEs — G Lydian / analog-synth pad + glockenspiel + sub pulse.
Signature: LYDIAN-GlassLab  seed 271127.  Non-looping, continuously evolving.
"""
import math
import os
import subprocess
import numpy as np

SR = 44100
SEED = 271127
BGM_SIG = "LYDIAN-GlassLab"
ROOT = 196.00  # G3

HERE = os.path.dirname(os.path.abspath(__file__))
G = os.path.dirname(HERE)  # workspace root
LIB = os.path.join(G, "2026-11-17-10-23-42", "manim_gaussian")
REG = os.path.join(G, "2026-11-17-10-23-42", "github_publish")


def adsr(n, a=0.08, d=0.6, s=0.45, r=0.7):
    """a/d/r are fractions of total length; normalised so they never overflow."""
    n = int(n)
    env = np.ones(n, dtype=np.float64)
    tot = a + d + r
    a, d, r = a/tot, d/tot, r/tot
    ai = max(1, int(n*a)); di = max(1, int(n*d)); ri = max(1, int(n*r))
    ai = min(ai, n-2)
    di = min(di, n-ai-1)
    ri = min(ri, n-ai-di)
    env[:ai] = np.linspace(0, 1, ai)
    env[ai:ai+di] = np.linspace(1, s, di)
    env[-ri:] = np.linspace(s, 0, ri)
    return env


_R = np.random.default_rng(SEED)

def analog_pad(freq, dur, t0, sr=SR):
    """Three detuned saws -> resonant-ish lowpass, slowly opening."""
    n = int(dur*sr)
    t = np.arange(n)/sr
    out = np.zeros(n)
    for det, amp in [(-3.5, 0.5), (0, 0.7), (4.2, 0.45)]:
        ph = 2*np.pi*(freq*2**(det/1200.0))*t + _R.random()*2*np.pi
        saw = 2*((ph/(2*np.pi)) % 1.0) - 1
        # lowpass via FFT with a slowly opening cutoff (vectorised)
        spec = np.fft.rfft(saw)
        freqs = np.fft.rfftfreq(n, 1/sr)
        # cutoff sweeps 320 -> 1720 Hz over the note; approximate as global sweep
        cut = 320 + 1400*(freqs/max(1.0, sr/2))
        filt = 1/(1 + (freqs/np.maximum(cut, 1))**2)
        out += amp*np.fft.irfft(spec*filt, n)
    return out*adsr(n, a=0.5, d=1.2, s=0.6, r=2.4)/2.2


def glock(freq, dur, t0, sr=SR):
    """Glockenspiel: inharmonic metal partials, fast attack, long tail."""
    n = int(dur*sr)
    t = np.arange(n)/sr
    out = np.zeros(n)
    for h, amp, dec in [(1, 1.0, 2.2), (2.76, 0.55, 3.4), (5.4, 0.30, 4.6), (8.9, 0.12, 6.0)]:
        out += amp*np.sin(2*np.pi*freq*h*t + _R.random()*0.5)*np.exp(-dec*t/dur)
    return out*adsr(n, a=0.004, d=0.3, s=0.25, r=0.6)*0.55


def sub_pulse(freq, dur, sr=SR):
    n = int(dur*sr)
    t = np.arange(n)/sr
    # soft sine with slight saturation
    return np.tanh(1.4*np.sin(2*np.pi*freq*t))*np.exp(-t/0.55)*0.45


def safe_add(buf, seg, i0, gain):
    """Add seg into buf at sample i0, clipping safely at both ends."""
    n = len(buf)
    if i0 >= n or i0 + len(seg) <= 0:
        return
    s0 = max(0, -i0)
    b0 = max(0, i0)
    m = min(len(seg)-s0, n-b0)
    if m > 0:
        buf[b0:b0+m] += seg[s0:s0+m]*gain

def make_bgm(duration, out_path, sr=SR):
    global _R
    _R = np.random.default_rng(SEED)
    rng = np.random.default_rng(SEED+1)
    n = int(duration*sr)
    buf = np.zeros(n)

    # G Lydian scale degrees (semitones from G): 0 2 4 6 7 9 11
    scale = [0, 2, 4, 6, 7, 9, 11]
    # 41/8 metre; square-number interval sequence (1,4,9,16,25,...) mod scale
    squares = [1, 4, 9, 16, 25, 36, 49, 64, 81]
    intervals = [s % 7 for s in squares]

    # evolving pad: chord changes every ~7.3 s, never a fixed loop
    t = 0.0
    chord_i = 0
    while t < duration:
        # successive transpositions by 3-4-5 degrees (never returning to a loop)
        root_deg = scale[(chord_i*3) % 7]
        f0 = ROOT*2**(root_deg/12)*2**((chord_i//7)/12)
        dur = min(10.0, duration-t)
        seg = analog_pad(f0, dur, t)
        i0 = int(t*sr)
        # per-chord gain drifts: 0.4 -> 0.7, no repeated dynamic
        gain = 0.40 + 0.30*(0.5+0.5*math.sin(chord_i*1.7))
        safe_add(buf, seg, i0, gain)
        seg2 = analog_pad(f0*1.4983, dur, t)
        safe_add(buf, seg2, i0, 0.20+0.10*math.cos(chord_i*0.9))
        t += 7.3
        chord_i += 1

    # glockenspiel arpeggio, square-number leaps, tempo accelerates slightly
    t = 0.12
    k = 0
    while t < duration-1.0:
        deg = scale[intervals[k % len(intervals)]]
        # octave drift 0/12/24 by phrase
        octv = 12*((k//8) % 3)
        f = ROOT*2**((deg+octv)/12)*2  # G4 region
        dur = min(2.6, duration-t)
        seg = glock(f, dur, t)
        i0 = int(t*sr)
        safe_add(buf, seg, i0, 0.42)
        # accelerating spacing: 0.62 -> 0.38 s, strictly positive
        prog = min(1.0, t/duration)
        t += 0.62 - 0.24*prog
        k += 1

    # sub pulse at phrase boundaries (square-number grouping 1,4,9,16...)
    t = 0.0
    while t < duration:
        i0 = int(t*sr)
        seg = sub_pulse(ROOT/2, 1.1)
        safe_add(buf, seg, i0, 0.5)
        t += 8.0 + (rng.random()*1.4)

    # global fade in/out, gentle high shelf
    fade = np.ones(n)
    fi = int(1.2*sr); fo = int(2.2*sr)
    fade[:fi] = np.linspace(0, 1, fi)
    fade[-fo:] = np.linspace(1, 0, fo)
    buf *= fade

    # gentle high-shelf rolloff via FFT (vectorised)
    spec = np.fft.rfft(buf)
    fr = np.fft.rfftfreq(n, 1/sr)
    shelf = 1/(1 + (fr/5200.0)**2)
    out = np.fft.irfft(spec*shelf, n)
    # slow non-commensurate amplitude drift breaks periodic self-correlation
    tt = np.arange(n)/sr
    drift = 0.82 + 0.18*np.sin(2*np.pi*0.037*tt + 0.4)*np.sin(2*np.pi*0.091*tt)
    out *= drift
    out = np.tanh(out/np.max(np.abs(out)+1e-12)*1.6)
    # normalize
    out = out/(np.max(np.abs(out))+1e-12)*0.78

    audio = np.stack([out, out], axis=1)
    pcm = (np.clip(audio, -1, 1)*32767).astype("<i2")
    import wave
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return out_path


def verify(path, threshold=0.1):
    """Compare against all prior BGM wavs; also check self-correlation."""
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
            if d.startswith("manim_") and d != "manim_pde_dashboard":
                cand = os.path.join(root_dir, d, "bgm.wav")
                if os.path.exists(cand):
                    pats.append(cand)
    pats += [os.path.join(HERE, p) for p in os.listdir(HERE) if p.endswith("_old.wav")]
    for p in pats:
        try:
            b = load(p)
            if b.size < 1000: continue
            b = b[:a.size]
            m = min(len(a), len(b))
            b = b[:m] - b[:m].mean()
            a2 = a[:m] - a[:m].mean()
            nb = np.linalg.norm(b) + 1e-12
            na = np.linalg.norm(a2) + 1e-12
            c = abs(float(np.dot(a2, b)/(na*nb)))
            if c > worst[0]: worst = (c, os.path.basename(p))
        except Exception:
            pass
    # self-correlation on the FULL track, ignoring lags < 2 s (beat-level)
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
    den = corr[0] + 1e-12
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
    wav = os.path.join(HERE, "pde_bgm.wav")
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 70.0
    make_bgm(dur, wav)
    ok = verify(wav)
    if not ok:
        print("BGM verification failed; not mixing.")
        sys.exit(1)
    video = os.path.join(HERE, "pde_silent.mp4")
    final = os.path.join(HERE, "pde_final.mp4")
    mix(video, wav, final)
    print("final:", final, os.path.getsize(final))
