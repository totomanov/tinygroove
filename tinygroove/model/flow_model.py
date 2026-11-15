"""tinygroove generator: a small flow-matching transformer over the autoencoder's latents.

Input: noise shaped like one sound (32 tokens x 16 numbers) and a condition:
  category (required in practice), any number of tags (the prompt words the teacher was given),
  and up to three measured knobs (decay, brightness, pitch), each optional.
Output: a latent for ae_model.Decoder. About 4M parameters; with the decoder about 5M ship.

Every condition can be left out: training drops them at random, which also gives the
"unconditional" branch needed for guidance (cfg).
"""
import math
import torch
import torch.nn as nn
from tinygroove.model.ae_model import D, K

KNOBS = ("decay", "brightness", "pitch")


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.film = nn.Linear(d, 4 * d)             # condition -> scale and shift for both norms
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)

    def forward(self, x, c):
        s1, b1, s2, b2 = self.film(c)[:, None].chunk(4, -1)
        h = self.n1(x) * (1 + s1) + b1
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.n2(x) * (1 + s2) + b2)


class Flow(nn.Module):
    def __init__(self, n_categories, n_tags, d=224, layers=5, heads=4):
        super().__init__()
        self.n_categories, self.n_tags, self.d = n_categories, n_tags, d
        self.inp = nn.Linear(D, d)
        self.pos = nn.Parameter(torch.randn(K, d) * 0.02)
        self.time = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.cat = nn.Embedding(n_categories + 1, d)                    # last index = not given
        self.tag = nn.Embedding(n_tags + 1, d, padding_idx=n_tags)      # last index = padding
        self.knob = nn.Sequential(nn.Linear(2 * len(KNOBS), d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, D)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x, t, cat, tags, knobs, kmask):
        """x [B,K,D] noisy latent, t [B] in 0..1 (0 = noise, 1 = data), cat [B] (n_categories = none),
        tags [B,n] indices padded with n_tags, knobs [B,3] standardized, kmask [B,3] 1 where given.
        Returns the velocity (data - noise)."""
        half = self.d // 2
        f = torch.exp(-math.log(1000.0) * torch.arange(half, device=x.device) / half)
        a = t[:, None] * 1000.0 * f
        c = self.time(torch.cat([a.sin(), a.cos()], 1)) + self.cat(cat) + self.tag(tags).sum(1) \
            + self.knob(torch.cat([knobs * kmask, kmask], 1))
        h = self.inp(x) + self.pos
        for b in self.blocks:
            h = b(h, c)
        return self.out(self.norm(h))

    @torch.no_grad()
    def sample(self, cat, tags=None, knobs=None, kmask=None, steps=24, cfg=2.0, noise=None, generator=None):
        """Returns latents in the flow's own scale; multiply by the stored `scale` before decoding."""
        B, dev = len(cat), cat.device
        tags = torch.full((B, 1), self.n_tags, device=dev) if tags is None else tags
        knobs = torch.zeros(B, len(KNOBS), device=dev) if knobs is None else knobs
        kmask = torch.zeros(B, len(KNOBS), device=dev) if kmask is None else kmask
        x = torch.randn(B, K, D, device=dev, generator=generator) if noise is None else noise.clone()
        none = (torch.full_like(cat, self.n_categories), torch.full_like(tags, self.n_tags),
                torch.zeros_like(knobs), torch.zeros_like(kmask))
        for i in range(steps):
            t = torch.full((B,), i / steps, device=dev)
            v = self(x, t, cat, tags, knobs, kmask)
            if cfg != 1.0:
                u = self(x, t, *none)
                v = u + cfg * (v - u)
            x = x + v / steps
        return x


if __name__ == "__main__":
    m = Flow(16, 172)
    print(f"flow {sum(p.numel() for p in m.parameters()) / 1e6:.2f}M parameters")
    c = torch.randint(16, (3,))
    print(tuple(m.sample(c, steps=4).shape))
