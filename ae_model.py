"""tinygroove drum autoencoder (v1).

A 1 s mono 44.1 kHz one-shot  <->  32 tokens x 16 numbers (512 numbers per sound).

Design, committed without ablations:
1. Front-loaded time grid. The 345 STFT frames of a clip are pooled into 32 tokens whose spans
   grow geometrically: about 3 ms each at the attack, about 100 ms each in the tail.
2. Spectrogram decoder with a noise branch. The decoder predicts a complex STFT (magnitude and
   phase, as in Vocos) plus a spectral envelope that shapes fresh white noise; one inverse STFT
   gives the waveform. Decoding the same latent twice gives two slightly different hits.
3. Generator-friendly latent. tanh-bounded, the decoder is trained on noised latents, and side
   heads predict category and measured knobs from the latent so that it is organised by meaning.

Only the decoder ships (about 1.2M parameters). The encoder (about 0.7M) is used offline and for
making variations of recorded samples.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 44100
N = 44032            # samples per clip used by the model (344 hops of 128)
NFFT, HOP = 512, 128
BINS = NFFT // 2 + 1
T = N // HOP + 1     # 345 frames
K, D = 32, 16        # tokens per sound, numbers per token
NOISE_NORM = 13.86   # |STFT| of unit white noise with this window, so envelope 1.0 = unit noise


def make_grid(frames=T, tokens=K):
    """Token spans in frames, growing geometrically from 1. Returns pool (K,T), unpool (T,K), spans."""
    lo, hi = 1.0, 2.0
    for _ in range(60):
        g = (lo + hi) / 2
        lo, hi = (g, hi) if sum(max(1.0, g ** k) for k in range(tokens)) < frames else (lo, g)
    spans = [max(1, round(g ** k)) for k in range(tokens)]
    spans[-1] += frames - sum(spans)
    assert spans[-1] > 0
    pool, centers, start = torch.zeros(tokens, frames), [], 0
    for k, n in enumerate(spans):
        pool[k, start:start + n] = 1.0 / n
        centers.append(start + (n - 1) / 2)
        start += n
    unpool = torch.zeros(frames, tokens)          # linear interpolation between token centres
    for t in range(frames):
        if t <= centers[0]:
            unpool[t, 0] = 1
        elif t >= centers[-1]:
            unpool[t, -1] = 1
        else:
            k = max(i for i in range(tokens - 1) if centers[i] <= t)
            w = (t - centers[k]) / (centers[k + 1] - centers[k])
            unpool[t, k], unpool[t, k + 1] = 1 - w, w
    return pool, unpool, spans


def stft(x, n=NFFT, hop=HOP):
    return torch.stft(x, n, hop, window=torch.hann_window(n, device=x.device), return_complex=True)


class ConvNeXt1d(nn.Module):
    def __init__(self, c, expand):
        super().__init__()
        self.dw = nn.Conv1d(c, c, 7, padding=3, groups=c)
        self.norm = nn.LayerNorm(c)
        self.p1, self.p2 = nn.Linear(c, expand * c), nn.Linear(expand * c, c)
        self.scale = nn.Parameter(torch.full((c,), 0.25))

    def forward(self, x):                           # [B, C, T]
        y = self.norm(self.dw(x).transpose(1, 2))
        return x + (self.p2(F.gelu(self.p1(y))) * self.scale).transpose(1, 2)


def _transformer(d, layers):
    layer = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


class Encoder(nn.Module):
    def __init__(self, c=128):
        super().__init__()
        self.inp = nn.Conv1d(2 * BINS, c, 3, padding=1)
        self.blocks = nn.Sequential(*[ConvNeXt1d(c, 2) for _ in range(4)])
        self.pos = nn.Parameter(torch.zeros(K, c))
        self.mix = _transformer(c, 2)
        self.out = nn.Linear(c, D)

    def forward(self, x, pool):                     # x [B, N] -> z [B, K, D] in (-1, 1)
        X = stft(x)
        mag = X.abs().clamp_min(1e-5)
        C = X / mag * mag ** 0.3                    # compressed complex spectrum
        h = self.blocks(self.inp(torch.cat([C.real, C.imag], 1)))
        tok = (h @ pool.T).transpose(1, 2) + self.pos
        return torch.tanh(self.out(self.mix(tok)))


class Decoder(nn.Module):
    def __init__(self, c=160):
        super().__init__()
        self.inp = nn.Linear(D, c)
        self.pos = nn.Parameter(torch.zeros(K, c))
        self.mix = _transformer(c, 2)
        self.frame_pos = nn.Parameter(torch.zeros(c, T))
        self.blocks = nn.Sequential(*[ConvNeXt1d(c, 2) for _ in range(4)])
        self.norm = nn.LayerNorm(c)
        self.tonal = nn.Linear(c, 2 * BINS)         # log magnitude, phase
        self.noise = nn.Linear(c, BINS)             # log noise envelope
        nn.init.constant_(self.noise.bias, -4.0)

    def forward(self, z, unpool, noise=1.0, generator=None):   # z [B, K, D] -> wave [B, N]
        tok = self.mix(self.inp(z) + self.pos)
        h = self.blocks(tok.transpose(1, 2) @ unpool.T + self.frame_pos)
        h = self.norm(h.transpose(1, 2))
        logmag, phase = self.tonal(h).transpose(1, 2).chunk(2, 1)
        S = torch.polar(torch.exp(logmag.clamp(max=7.0)), phase)
        if noise > 0:
            white = torch.randn(z.shape[0], N, device=z.device, generator=generator)
            env = torch.exp(self.noise(h).transpose(1, 2).clamp(max=5.0))
            S = S + noise * env * stft(white) / NOISE_NORM
        return torch.istft(S, NFFT, HOP, window=torch.hann_window(NFFT, device=z.device), length=N)


class AE(nn.Module):
    def __init__(self, n_categories=16, n_knobs=2):
        super().__init__()
        self.enc, self.dec = Encoder(), Decoder()
        pool, unpool, spans = make_grid()
        self.register_buffer("pool", pool)
        self.register_buffer("unpool", unpool)
        self.spans = spans
        self.aux = nn.Sequential(nn.Linear(K * D, 256), nn.GELU(), nn.Linear(256, n_categories + n_knobs))

    def encode(self, x):
        return self.enc(x, self.pool)

    def decode(self, z, noise=1.0, generator=None):
        return self.dec(z, self.unpool, noise, generator)


if __name__ == "__main__":
    m = AE()
    n = lambda mod: sum(p.numel() for p in mod.parameters())
    print(f"encoder {n(m.enc) / 1e6:.2f}M  decoder {n(m.dec) / 1e6:.2f}M  spans(frames) {m.spans}")
    x = torch.randn(2, N) * 0.1
    z = m.encode(x)
    print("z", tuple(z.shape), "y", tuple(m.decode(z).shape))
