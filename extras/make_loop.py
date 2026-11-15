# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Tiny step sequencer: build a seamless classic house loop from rendered one-shots.

    uv run extras/make_loop.py                      # -> renders/house_loop_124.wav
    uv run extras/make_loop.py --bpm 122 --bars 8 --out renders/my_loop.wav

Swap sounds or patterns by editing TRACKS below (16 steps per bar, x = hit).
"""
import argparse, subprocess, wave
from pathlib import Path
import numpy as np

SR = 44100
R = Path(__file__).resolve().parent.parent / "renders"

#            sample                                         gain  choke(16ths)  pattern (1 bar)
TRACKS = {
    "kick":  (R / "v2_vars_sm-sfx/kick_house_s1.wav",          1.00, None, "x...x...x...x..."),
    "clap":  (R / "v2_vars_sm-sfx/clap_house_s1.wav",          0.70, None, "....x.......x..."),
    "chat":  (R / "v3_drum_machine_hats/hat_909_closed_s1.wav", 0.30, 1,   ".x.x.x.x.x.x.x.x"),
    "ohat":  (R / "v3_drum_machine_hats/hat_909_open_s1.wav",   0.45, 1.8, "..x...x...x...x."),
}


def load(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "2", "-ar", str(SR),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32).reshape(-1, 2).copy()
    # strip leading silence so hits land on the grid, normalize to peak 1
    on = np.argmax(np.abs(x).max(1) > 0.01 * np.abs(x).max())
    x = x[on:]
    return x / (np.abs(x).max() + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bpm", type=float, default=124)
    ap.add_argument("--bars", type=int, default=4)
    ap.add_argument("--out", default=None, help="default: renders/house_loop_<bpm>bpm_<bars>bars.wav")
    a = ap.parse_args()
    a.out = a.out or str(R / f"house_loop_{a.bpm:g}bpm_{a.bars}bars.wav")

    step = 60 / a.bpm / 4                       # one 16th in seconds
    n = round(a.bars * 16 * step * SR)
    mix = np.zeros((n, 2), np.float32)
    for name, (path, gain, choke, pat) in TRACKS.items():
        s = load(path) * gain
        if choke:                                # cut tail, 5 ms fade
            L = min(len(s), int(choke * step * SR))
            s = s[:L].copy(); f = min(220, L)
            s[-f:] *= np.linspace(1, 0, f)[:, None]
        for bar in range(a.bars):
            for i, c in enumerate(pat):
                if c != "x":
                    continue
                t = round((bar * 16 + i) * step * SR)
                idx = (np.arange(len(s)) + t) % n  # wrap tails to the start: seamless loop
                np.add.at(mix, idx, s)
    mix *= 0.89 / np.abs(mix).max()            # -1 dBFS
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((mix * 32767).astype("<i2").tobytes())
    print(f"{out}  {a.bpm} bpm, {a.bars} bars, {n / SR:.2f}s")


if __name__ == "__main__":
    main()
