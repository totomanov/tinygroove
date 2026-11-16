"""Export the trained student for the browser demo.

    uv run --with torch --with numpy --with onnx --with onnxruntime python -m tinygroove.model.export_onnx

Reads runs/v2/ (flow_student.pt, decoder.pt) and writes demo/models/: flow.onnx (the student: one jump from noise
to sound), decoder.onnx and meta.json. Each graph is checked against PyTorch at a batch size it was not exported with.
"""
import json, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tinygroove.model.ae_model import D, K, HOP, N, NFFT, Decoder
from tinygroove.model.flow import TOKENS, Flow

ROOT = Path(__file__).resolve().parents[2]
RUN, OUT = ROOT / "runs" / "v2", ROOT / "demo" / "models"


class Decode(nn.Module):
    """The decoder with its inverse STFT written as a transposed convolution."""
    def __init__(self, dec):
        super().__init__()
        self.dec = dec
        n, k, w = torch.arange(NFFT), torch.arange(NFFT // 2 + 1)[:, None], torch.hann_window(NFFT)
        c = torch.full((NFFT // 2 + 1, 1), 2.0); c[0] = c[-1] = 1.0
        ang = 2 * math.pi * k * n / NFFT
        self.register_buffer("basis", (torch.cat([c * torch.cos(ang), -c * torch.sin(ang)]) / NFFT * w)[:, None])
        env = F.conv_transpose1d(torch.ones(1, 1, N // HOP + 1), (w ** 2)[None, None], stride=HOP)[0, 0]
        self.register_buffer("env", env[NFFT // 2: NFFT // 2 + N])

    def forward(self, z, frames):
        re, im = self.dec.spectrum(z, frames)
        return F.conv_transpose1d(torch.cat([re, im], 1), self.basis, stride=HOP)[:, 0, NFFT // 2: NFFT // 2 + N] / self.env


def main():
    import onnxruntime as ort
    OUT.mkdir(parents=True, exist_ok=True)
    f = torch.load(RUN / "flow_student.pt", map_location="cpu")
    names, vocab = f["categories"], f["tags"]
    flow = Flow(len(names), len(vocab), f["d"], f["layers"]).eval(); flow.load_state_dict(f["state"])
    dyn = lambda ks: {k: {0: "batch"} for k in ks}
    args = lambda B: (torch.randn(B, TOKENS, D), torch.rand(B), torch.full((B,), 0.5), torch.randint(len(names), (B,)),
                      torch.randint(len(vocab) + 1, (B, 4)), torch.randn(B, 3), torch.ones(B, 3))
    ins = ["x", "t", "dt", "cat", "tags", "knobs", "kmask"]
    torch.onnx.export(flow, args(3), OUT / "flow.onnx", input_names=ins, output_names=["v"], dynamic_axes=dyn(ins + ["v"]),
                      opset_version=17, dynamo=False)
    a = args(5)
    with torch.no_grad():
        want = flow(*a).numpy()
    got = ort.InferenceSession(str(OUT / "flow.onnx")).run(None, {k: v.numpy() for k, v in zip(ins, a)})[0]
    print("flow         max abs difference", float(np.abs(got - want).max()), "| parameters", sum(p.numel() for p in flow.parameters()))
    p = torch.load(RUN / "decoder.pt", map_location="cpu")
    dec = Decoder(p["width"], p["mode"]).eval(); dec.load_state_dict(p["state"])
    torch.onnx.export(Decode(dec).eval(), (torch.tanh(torch.randn(3, K, D)), torch.tensor([20.0, 90.0, 300.0])), OUT / "decoder.onnx",
                      input_names=["z", "frames"], output_names=["wave"], dynamic_axes=dyn(["z", "frames", "wave"]),
                      opset_version=17, dynamo=False)
    z, fr = torch.tanh(torch.randn(5, K, D)), torch.tensor([13.0, 40.0, 107.0, 222.0, 340.0])
    with torch.no_grad():
        want = dec(z, fr).numpy()
    got = ort.InferenceSession(str(OUT / "decoder.onnx")).run(None, {"z": z.numpy(), "frames": fr.numpy()})[0]
    print("decoder      max abs difference", float(np.abs(got - want).max()), "| parameters", sum(q.numel() for q in dec.parameters()))
    json.dump(dict(categories=names, tags=vocab, scale=f["scale"], knob_range=f["knob_range"], K=TOKENS, D=D, tokens=K,
                   len_mean=f["knob_mean"][0], len_std=f["knob_std"][0], n=N, sr=44100), open(OUT / "meta.json", "w"))
    for q in sorted(OUT.iterdir()):
        print(f"{q.name}: {q.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
