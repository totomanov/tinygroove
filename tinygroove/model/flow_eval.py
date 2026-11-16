"""Checks for a trained generator, all measured on the audio it produces.

For every setting (model, number of steps), 64 sounds per drum type are generated from the type alone:
  spread   how varied the brightness is, relative to the teacher's clips of that type (1.0 = as varied)
  drift    how far the typical brightness sits from the teacher's, in octaves (0 = same)
  drum     share judged to be the requested type, by the autoencoder's encoder and its classifier
  gate     share that pass the dataset gate
and, asking for the decay / brightness / pitch of real held-out sounds, how well each knob is followed.

The "drum" score depends on the decoder as well as the generator: the same generator scored 0.68 through the
spectral-only decoder and 0.95 through the adversarially trained one.
"""
import numpy as np
import torch
from tinygroove.data.process_dataset import process
from tinygroove.model.ae_model import N, SR, D, K, frames_of
from tinygroove.model.ae_train import brightness
from tinygroove.model.flow import TOKENS, sample


def pitch(audio, device, bs=1024):
    """log2 of the strongest frequency between 30 Hz and 16 kHz, per clip (audio int16 [M, >=N])."""
    out = []
    f = torch.fft.rfftfreq(N, 1 / SR).to(device)
    ok = (f >= 30) & (f <= 16000)
    for i in range(0, len(audio), bs):
        P = torch.fft.rfft(audio[i:i + bs, :N].to(device).float() / 32768).abs()
        out.append(torch.log2(f[(P * ok).argmax(1)]))
    return torch.cat(out)


def to16(y):
    return (y / y.abs().amax(1, keepdim=True).clamp(min=1e-6) * 0.89 * 32767).short()


def gate(y16):
    """Dataset gate on int16 audio [B, N]. Returns (passed [B] bool, log2 length ms [B], 0 where failed)."""
    info = [process(w[:, None])[1] for w in (y16.float() / 32768).cpu().numpy()]
    ok = torch.tensor([not isinstance(i, str) for i in info])
    return ok, torch.tensor([np.log2(i) if not isinstance(i, str) else 0.0 for i in info])


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-9))


def judge_with(ae, n_cat, device):
    """Which drum is this? Re-encode the audio (length taken from the gate) and ask the autoencoder's classifier."""
    def judge(y16, log2_ms, ok):
        ms = torch.where(ok, torch.exp2(log2_ms), torch.full_like(log2_ms, 300.0)).clamp(20, 1000).to(device)
        z = ae.encode(y16.float() / 32768, frames_of(ms))
        return ae.aux(z.flatten(1))[:, :n_cat].argmax(1)
    return judge


@torch.no_grad()
def score(make, judge, audio, cats, knobs_raw, kmean, kstd, val_idx, names, device, per_cat=64, seed=0):
    """make(cat [B], knobs [B,3] standardized, kmask [B,3], generator) -> waveform [B, N]."""
    g = torch.Generator(device=device).manual_seed(seed)
    n_cat = len(names)
    c = torch.arange(n_cat, device=device).repeat_interleave(per_cat)
    zero = torch.zeros(len(c), 3, device=device)
    y16 = to16(make(c, zero, zero, g))
    b = brightness(y16, device)
    spread, drift = [], []
    for k in range(n_cat):
        real = brightness(audio[torch.nonzero(cats == k)[:, 0][:512]], device)
        spread.append(float(b[c == k].std() / real.std())); drift.append(abs(float(b[c == k].median() - real.median())))
    ok, ln = gate(y16)
    out = dict(spread=round(float(np.mean(spread)), 3), drift=round(float(np.mean(drift)), 3),
               drum=round(float((judge(y16, ln, ok) == c).float().mean()), 3), gate=round(float(ok.float().mean()), 3))
    idx = val_idx[:768]
    want = (knobs_raw[idx] - kmean) / kstd
    y16 = to16(make(cats[idx], want, torch.ones_like(want), g))
    ok, ln = gate(y16)
    got = (torch.stack([ln.to(device), brightness(y16, device), pitch(y16, device)], 1) - kmean) / kstd
    ok = ok.to(device)
    out["knobs"] = [round(corr(want[ok, i], got[ok, i]), 3) for i in range(3)]
    return out


def maker(flow, dec, scale, lmean, lstd, steps, student, n_tags, device):
    """Generator + decoder -> make(). The sound's length comes from the generated length token."""
    def make(cat, knobs, kmask, g):
        B = len(cat)
        tags = torch.full((B, 4), n_tags, device=device)
        x = sample(flow, cat, tags, knobs, kmask, steps, torch.randn(B, TOKENS, D, device=device, generator=g), student)
        ms = torch.exp2((x[:, K, 0] * lstd + lmean).clamp(4.5, 10.0))
        return dec((x[:, :K] * scale).clamp(-1, 1), frames_of(ms))
    return make
