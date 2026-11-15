"""Export the trained student for the browser demo.

    uv run --with torch --with numpy --with onnx --with onnxruntime python export_onnx.py

Writes demo/models/flow.onnx (one denoising step; the page runs the loop), demo/models/decoder.onnx
(latent -> waveform) and demo/models/meta.json (categories, tags, latent scale, knob ranges).

Two things differ from the PyTorch code, both checked below against it:
- attention is written out by hand (the built-in layers do not export with a free batch size);
- the inverse STFT is a transposed convolution, and the decoder's noise branch is left out
  (it contributes about -41 dB).
"""
import json, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ae_model import AE, HOP, N, NFFT, K, D
from flow_model import Flow

OUT = Path(__file__).resolve().parent / "demo" / "models"


def attention(m, h):
    """nn.MultiheadAttention self-attention, written out."""
    B, L, d = h.shape
    q, k, v = F.linear(h, m.in_proj_weight, m.in_proj_bias).reshape(B, L, 3, m.num_heads, d // m.num_heads) \
               .permute(2, 0, 3, 1, 4)
    a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(d // m.num_heads), -1) @ v
    return m.out_proj(a.transpose(1, 2).reshape(B, L, d))


class FlowStep(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x, t, cat, tags, knobs, kmask):
        f = self.f
        half = f.d // 2
        fr = torch.exp(-math.log(1000.0) * torch.arange(half) / half)
        a = t[:, None] * 1000.0 * fr
        c = f.time(torch.cat([a.sin(), a.cos()], 1)) + f.cat(cat) + f.tag(tags).sum(1) \
            + f.knob(torch.cat([knobs * kmask, kmask], 1))
        h = f.inp(x) + f.pos
        for b in f.blocks:
            s1, b1, s2, b2 = b.film(c)[:, None].chunk(4, -1)
            h = h + attention(b.attn, b.n1(h) * (1 + s1) + b1)
            h = h + b.mlp(b.n2(h) * (1 + s2) + b2)
        return f.out(f.norm(h))


class Decode(nn.Module):
    def __init__(self, ae):
        super().__init__()
        self.d = ae.dec
        self.register_buffer("unpool", ae.unpool)
        n = torch.arange(NFFT)
        k = torch.arange(NFFT // 2 + 1)[:, None]
        w = torch.hann_window(NFFT)
        c = torch.full((NFFT // 2 + 1, 1), 2.0); c[0] = c[-1] = 1.0
        ang = 2 * math.pi * k * n / NFFT
        basis = torch.cat([c * torch.cos(ang), -c * torch.sin(ang)]) / NFFT * w        # [514, 512]
        self.register_buffer("basis", basis[:, None])
        frames = N // HOP + 1
        env = F.conv_transpose1d(torch.ones(1, 1, frames), (w ** 2)[None, None], stride=HOP)[0, 0]
        self.register_buffer("env", env[NFFT // 2: NFFT // 2 + N])

    def layer(self, l, x):
        x = x + attention(l.self_attn, l.norm1(x))
        return x + l.linear2(F.relu(l.linear1(l.norm2(x))))

    def forward(self, z):
        d = self.d
        tok = d.inp(z) + d.pos
        for l in d.mix.layers:
            tok = self.layer(l, tok)
        h = d.blocks(tok.transpose(1, 2) @ self.unpool.T + d.frame_pos)
        logmag, phase = d.tonal(d.norm(h.transpose(1, 2))).transpose(1, 2).chunk(2, 1)
        mag = torch.exp(logmag.clamp(max=7.0))
        S = torch.cat([mag * torch.cos(phase), mag * torch.sin(phase)], 1)
        y = F.conv_transpose1d(S, self.basis, stride=HOP)[:, 0, NFFT // 2: NFFT // 2 + N]
        return y / self.env


def main():
    import onnxruntime as ort
    c = torch.load("flow_v1/flow.pt", map_location="cpu")
    names, vocab = c["categories"], c["tags"]
    flow = Flow(len(names), len(vocab)).eval()
    flow.load_state_dict(c["state"])
    ae = AE(len(names), 2).eval()
    ae.load_state_dict(torch.load(f"{c['ae']}/ae.pt", map_location="cpu")["state"])
    OUT.mkdir(parents=True, exist_ok=True)

    B = 3
    args = (torch.randn(B, K, D), torch.rand(B), torch.randint(len(names), (B,)),
            torch.randint(len(vocab) + 1, (B, 4)), torch.randn(B, 3), torch.ones(B, 3))
    step, dec = FlowStep(flow).eval(), Decode(ae).eval()
    dyn = lambda ks: {k: {0: "batch"} for k in ks}
    torch.onnx.export(step, args, OUT / "flow.onnx", input_names=["x", "t", "cat", "tags", "knobs", "kmask"],
                      output_names=["v"], dynamic_axes=dyn(["x", "t", "cat", "tags", "knobs", "kmask", "v"]),
                      opset_version=17, dynamo=False)
    z = torch.tanh(torch.randn(B, K, D))
    torch.onnx.export(dec, (z,), OUT / "decoder.onnx", input_names=["z"], output_names=["wave"],
                      dynamic_axes=dyn(["z", "wave"]), opset_version=17, dynamo=False)

    # check against PyTorch, at a batch size different from the export's
    B = 5
    args = (torch.randn(B, K, D), torch.rand(B), torch.randint(len(names), (B,)),
            torch.randint(len(vocab) + 1, (B, 4)), torch.randn(B, 3), torch.ones(B, 3))
    z = torch.tanh(torch.randn(B, K, D))
    with torch.no_grad():
        want_v, want_y = flow(*args), ae.decode(z, noise=0.0)
    names_in = ["x", "t", "cat", "tags", "knobs", "kmask"]
    got_v = ort.InferenceSession(str(OUT / "flow.onnx")).run(None, {k: a.numpy() for k, a in zip(names_in, args)})[0]
    got_y = ort.InferenceSession(str(OUT / "decoder.onnx")).run(None, {"z": z.numpy()})[0]
    print("flow    max abs difference", float(np.abs(got_v - want_v.numpy()).max()))
    print("decoder max abs difference", float(np.abs(got_y - want_y.numpy()).max()), "peak", float(np.abs(got_y).max()))

    json.dump(dict(categories=names, tags=vocab, scale=c["scale"], knob_range=c["knob_range"],
                   K=K, D=D, n=N, sr=44100), open(OUT / "meta.json", "w"))
    for f in sorted(OUT.iterdir()):
        print(f"{f.name}: {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
