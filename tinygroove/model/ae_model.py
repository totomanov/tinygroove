"""tinygroove drum autoencoder.

A 1 s mono 44.1 kHz one-shot  <->  20 tokens x 16 numbers (320 numbers) plus its length.

Design:
1. The time grid follows the sound. The median drum is 310 ms long, so tokens spread over a full second mostly
   describe silence. Here 8 attack tokens sit at fixed times (the first 35 ms), and 12 tail tokens are spread over
   whatever is left of that particular sound. `grid_mode="fixed"` spreads the same 20 tokens over the whole second
   instead; measured, the two reconstruct equally well (held-out loss 0.117 against 0.115).
2. The decoder predicts a complex spectrogram (magnitude and phase); one inverse STFT gives the waveform.
3. Attention is written out by hand and the grid is plain tensor arithmetic, so the decoder exports to ONNX as is.
4. Widths are arguments. A half-width decoder (0.51M) was trained on the same latents and was audibly worse.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 44100
N = 44032                 # samples per clip (344 hops of 128)
NFFT, HOP = 512, 128
BINS = NFFT // 2 + 1
T = N // HOP + 1          # 345 frames
D = 16                    # numbers per token
ATTACK = (1, 1, 1, 1, 2, 2, 2, 2)        # frames per attack token: 12 frames = 35 ms
KT = 12                   # tail tokens
K = len(ATTACK) + KT      # 20 tokens
A_END = sum(ATTACK)
RATIO = 1.25              # each tail token spans 1.25x the one before


def frames_of(length_ms):
    """Sound length in STFT frames, from the gate's length in ms."""
    return length_ms.float() * (SR / 1000 / HOP)


def grid(frames, mode="tail"):
    """frames [B]: each sound's length in frames. Returns pool [B,K,T], unpool [B,T,K], mask [B,T].

    pool averages STFT frames into tokens; unpool interpolates tokens back to frames; mask is 1 inside the sound.
    """
    B, dev = frames.shape[0], frames.device
    end = (frames + 4).clamp(A_END + KT, T) if mode == "tail" else frames * 0 + float(T)
    a = torch.tensor([0.0] + [float(sum(ATTACK[:i + 1])) for i in range(len(ATTACK))], device=dev)       # 9 bounds
    j = torch.arange(KT + 1, device=dev, dtype=torch.float32)
    g = (RATIO ** j - 1) / (RATIO ** KT - 1)                                                             # 0 .. 1
    tail = A_END + (end[:, None] - A_END) * g[None, 1:]                                                  # [B, KT]
    bounds = torch.cat([a[None] + frames[:, None] * 0, tail], 1)                                         # [B, K+1]
    lo, hi = bounds[:, :-1, None], bounds[:, 1:, None]
    t = torch.arange(T, device=dev, dtype=torch.float32)
    overlap = (torch.minimum(hi, t + 1) - torch.maximum(lo, t)).clamp(min=0)                             # [B, K, T]
    pool = overlap / overlap.sum(2, keepdim=True).clamp(min=1e-6)
    c = ((bounds[:, :-1] + bounds[:, 1:]) / 2)[:, None, :]                                               # centres [B,1,K]
    far = (frames * 0 + 1e6)[:, None, None]             # built from `frames` so the batch size stays free in ONNX
    prev, nxt = torch.cat([-far, c[:, :, :-1]], 2), torch.cat([c[:, :, 1:], far], 2)
    tau = (t + 0.5)[None, :, None]
    unpool = torch.minimum((tau - prev) / (c - prev), (nxt - tau) / (nxt - c)).clamp(0, 1)               # [B, T, K]
    mask = (end[:, None] - t[None]).clamp(0, 1)
    return pool, unpool, mask


def stft(x, n=NFFT, hop=HOP):
    return torch.stft(x, n, hop, window=torch.hann_window(n, device=x.device), return_complex=True)


class ConvNeXt1d(nn.Module):
    def __init__(self, c, expand=2):
        super().__init__()
        self.dw = nn.Conv1d(c, c, 7, padding=3, groups=c)
        self.norm = nn.LayerNorm(c)
        self.p1, self.p2 = nn.Linear(c, expand * c), nn.Linear(expand * c, c)
        self.scale = nn.Parameter(torch.full((c,), 0.25))

    def forward(self, x):                                   # [B, C, T]
        y = self.norm(self.dw(x).transpose(1, 2))
        return x + (self.p2(F.gelu(self.p1(y))) * self.scale).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, d, heads=4):
        super().__init__()
        self.heads, self.qkv, self.out = heads, nn.Linear(d, 3 * d), nn.Linear(d, d)

    def forward(self, x):                                   # [B, L, d]
        B, L, d = x.shape
        q, k, v = self.qkv(x).reshape(B, L, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(d // self.heads), -1) @ v
        return self.out(a.transpose(1, 2).reshape(B, L, d))


class Block(nn.Module):
    def __init__(self, d, heads=4, ff=2):
        super().__init__()
        self.n1, self.n2, self.attn = nn.LayerNorm(d), nn.LayerNorm(d), Attention(d, heads)
        self.mlp = nn.Sequential(nn.Linear(d, ff * d), nn.GELU(), nn.Linear(ff * d, d))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


def _len_feature(frames):
    return (torch.log2(frames.clamp(min=1)) / 8)[:, None]


class Encoder(nn.Module):
    def __init__(self, c=128):
        super().__init__()
        self.inp = nn.Conv1d(2 * BINS, c, 3, padding=1)
        self.blocks = nn.Sequential(*[ConvNeXt1d(c) for _ in range(4)])
        self.pos, self.len = nn.Parameter(torch.zeros(K, c)), nn.Linear(1, c)
        self.mix = nn.Sequential(Block(c), Block(c))
        self.out = nn.Linear(c, D)

    def forward(self, x, frames, pool):                     # x [B, N] -> z [B, K, D] in (-1, 1)
        X = stft(x)
        mag = X.abs().clamp_min(1e-5)
        C = X / mag * mag ** 0.3
        h = self.blocks(self.inp(torch.cat([C.real, C.imag], 1)))
        tok = pool @ h.transpose(1, 2) + self.pos + self.len(_len_feature(frames))[:, None]
        return torch.tanh(self.out(self.mix(tok)))


class Decoder(nn.Module):
    def __init__(self, c=160, grid_mode="tail"):
        super().__init__()
        self.c, self.grid_mode = c, grid_mode
        self.inp = nn.Linear(D, c)
        self.pos, self.len = nn.Parameter(torch.zeros(K, c)), nn.Linear(1, c)
        self.mix = nn.Sequential(Block(c), Block(c))
        self.frame_pos = nn.Parameter(torch.zeros(c, T))
        self.blocks = nn.Sequential(*[ConvNeXt1d(c) for _ in range(4)])
        self.norm = nn.LayerNorm(c)
        self.tonal = nn.Linear(c, 2 * BINS)                 # log magnitude, phase

    def spectrum(self, z, frames):                          # -> real, imag [B, BINS, T]; silent past the sound's end
        _, unpool, mask = grid(frames, self.grid_mode)
        tok = self.mix(self.inp(z) + self.pos + self.len(_len_feature(frames))[:, None])
        h = self.blocks((unpool @ tok).transpose(1, 2) + self.frame_pos)
        logmag, phase = self.tonal(self.norm(h.transpose(1, 2))).transpose(1, 2).chunk(2, 1)
        mag = torch.exp(logmag.clamp(max=7.0)) * mask[:, None]
        return mag * torch.cos(phase), mag * torch.sin(phase)

    def forward(self, z, frames):                           # z [B, K, D] -> wave [B, N]
        re, im = self.spectrum(z, frames)
        return torch.istft(torch.complex(re, im), NFFT, HOP, window=torch.hann_window(NFFT, device=z.device), length=N)


class AE(nn.Module):
    def __init__(self, grid_mode="tail", enc_c=128, dec_c=160, n_categories=16, n_knobs=2):
        super().__init__()
        self.grid_mode = grid_mode
        self.enc, self.dec = Encoder(enc_c), Decoder(dec_c, grid_mode)
        self.aux = nn.Sequential(nn.Linear(K * D, 256), nn.GELU(), nn.Linear(256, n_categories + n_knobs))

    def encode(self, x, frames):
        return self.enc(x, frames, grid(frames, self.grid_mode)[0])

    def decode(self, z, frames):
        return self.dec(z, frames)


if __name__ == "__main__":
    n = lambda mod: sum(p.numel() for p in mod.parameters())
    for c in (160, 112):
        m = AE(dec_c=c)
        print(f"encoder {n(m.enc) / 1e6:.2f}M  decoder(c={c}) {n(m.dec) / 1e6:.2f}M")
    fr = torch.tensor([13.0, 60.0, 107.0, 330.0])
    pool, unpool, mask = grid(fr)
    print("pool rows sum", pool.sum(2).min().item(), "unpool rows sum", unpool.sum(2).min().item(), unpool.sum(2).max().item(), "mask frames", mask.sum(1).tolist())
    x = torch.randn(4, N) * 0.1
    z = m.encode(x, fr)
    print("z", tuple(z.shape), "y", tuple(m.decode(z, fr).shape))
