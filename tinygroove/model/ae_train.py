"""Training loop for the drum autoencoder. Runs for a fixed time budget, then exports.

Losses: multi-resolution STFT distance (windows 64..2048, extra weight on the first 50 ms),
hinge GAN + feature matching from three spectrogram discriminators, and small side losses that
make the latent predict category and measured knobs.
"""
import copy, json, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tinygroove.model.ae_model import AE, N, SR, stft

FFTS = (2048, 1024, 512, 256, 128, 64)


def _mag(x, n):
    X = torch.stft(x, n, n // 4, window=torch.hann_window(n, device=x.device), return_complex=True)
    return torch.sqrt(X.real ** 2 + X.imag ** 2 + 1e-9)


def mrstft(y, x):
    total = 0.0
    for n in FFTS:
        Y, X = _mag(y, n), _mag(x, n)
        t = torch.arange(X.shape[-1], device=x.device) * (n // 4) / SR
        w = 1 + 4 * torch.exp(-t / 0.05)                  # the attack counts 5x
        w = w / w.mean()
        eps = n * 1e-4
        total = total + ((torch.log(Y + eps) - torch.log(X + eps)).abs().mean(1) * w).mean() \
                      + (((Y + eps) ** 0.3 - (X + eps) ** 0.3).abs().mean(1) * w).mean()
    return total / len(FFTS)


D_CROP = 32768        # discriminators look at the first 0.74 s; the rest is almost always silence
D_BATCH = 16          # ...and at this many clips of each batch: they are the expensive part


class SpecD(nn.Module):
    """Discriminator on one spectrogram resolution.

    Input is a compressed complex spectrogram plus log magnitude. On a linear scale a drum
    spectrogram is almost all zeros (mean about 0.002), which the first version could not learn from.
    """
    def __init__(self, n, ch=32):
        super().__init__()
        wn = nn.utils.parametrizations.weight_norm
        self.n = n
        self.convs = nn.ModuleList(
            [wn(nn.Conv2d(3, ch, (3, 9), stride=(1, 2), padding=(1, 4)))]
            + [wn(nn.Conv2d(ch, ch, (3, 9), stride=(1, 2), padding=(1, 4))) for _ in range(3)]
            + [wn(nn.Conv2d(ch, ch, 3, padding=1))])
        self.out = wn(nn.Conv2d(ch, 1, 3, padding=1))

    def forward(self, x):
        x = x[:, :D_CROP].float()
        X = torch.stft(x, self.n, self.n // 4, window=torch.hann_window(self.n, device=x.device), return_complex=True)
        re, im = X.real / self.n ** 0.5, X.imag / self.n ** 0.5
        mag = torch.sqrt(re ** 2 + im ** 2 + 1e-9)
        gain = (mag + 1e-4) ** 0.3 / (mag + 1e-4)
        h = torch.stack([re * gain, im * gain, torch.log(mag + 1e-4) / 4 + 1], 1).transpose(2, 3)   # [B, 3, T, F]
        feats = []
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            for c in self.convs:
                h = F.leaky_relu(c(h), 0.1)
                feats.append(h.float())
            out = self.out(h).float()
        return out, feats


def brightness(audio, device, bs=1024):
    """log2 spectral centroid per clip."""
    out = []
    for i in range(0, len(audio), bs):
        P = stft(audio[i:i + bs, :N].to(device).float() / 32768).abs() ** 2
        f = torch.linspace(0, SR / 2, P.shape[1], device=device)[None, :, None]
        out.append(torch.log2((P * f).sum((1, 2)) / (P.sum((1, 2)) + 1e-9) + 1.0))
    return torch.cat(out)


def train(audio, cats, length_ms, steps, max_s, device, batch=96, lr=4e-4, log=print, n_val=1024, seed=0,
          gan_start=0.03, on_ckpt=None, ckpt_every=5000):
    """audio: int16 [M, >=N] tensor on `device`. Trains for `steps` steps, or until `max_s` seconds if that
    comes first (the learning-rate schedule follows whichever is further along)."""
    torch.manual_seed(seed)
    M = len(audio)
    perm = torch.randperm(M, device=device)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    knobs = torch.stack([torch.log2(length_ms.float().to(device)), brightness(audio, device)], 1)
    knobs = (knobs - knobs[train_idx].mean(0)) / (knobs[train_idx].std(0) + 1e-6)
    n_cat = int(cats.max()) + 1
    cats = cats.to(device)

    model = AE(n_cat, knobs.shape[1]).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    discs = nn.ModuleList([SpecD(n) for n in (2048, 512, 128)]).to(device)
    optG = torch.optim.AdamW(model.parameters(), lr, betas=(0.8, 0.99), weight_decay=0.01)
    optD = torch.optim.AdamW(discs.parameters(), lr, betas=(0.8, 0.99), weight_decay=0.01)

    t0, step, hist = time.time(), 0, []
    while True:
        p = max(step / steps, (time.time() - t0) / max_s)
        if p >= 1:
            break
        cur = lr * min(1.0, (step + 1) / 300) * (0.5 * (1 + math.cos(math.pi * p)) * 0.98 + 0.02)
        for o in (optG, optD):
            for g in o.param_groups:
                g["lr"] = cur
        idx = train_idx[torch.randint(len(train_idx), (batch,), device=device)]
        x = audio[idx, :N].float() / 32768
        z = model.encode(x)
        zn = z + torch.randn_like(z) * 0.1 * torch.rand(batch, 1, 1, device=device)   # noise-robust decoder
        y = model.decode(zn)
        use_gan = p > gan_start
        d_loss = d_real = d_fake = torch.zeros((), device=device)
        xd, yd = x[:D_BATCH], y[:D_BATCH]
        if use_gan:
            sr = [d(xd)[0] for d in discs]
            sf = [d(yd.detach())[0] for d in discs]
            d_loss = sum(F.relu(1 - a).mean() + F.relu(1 + b).mean() for a, b in zip(sr, sf))
            d_real, d_fake = sum(a.mean() for a in sr) / len(discs), sum(b.mean() for b in sf) / len(discs)
            optD.zero_grad(set_to_none=True)
            d_loss.backward()
            optD.step()
        spec = mrstft(y, x)
        a = model.aux(z.flatten(1))
        aux = F.cross_entropy(a[:, :n_cat], cats[idx]) + F.mse_loss(a[:, n_cat:], knobs[idx])
        adv = fm = torch.zeros((), device=device)
        if use_gan:
            for d in discs:
                sy, fy = d(yd)
                with torch.no_grad():
                    _, fx = d(xd)
                adv = adv - sy.mean()
                fm = fm + sum(F.l1_loss(u, v) for u, v in zip(fy, fx)) / len(fy)
        loss = 15 * spec + 0.1 * aux + adv + 2 * fm
        optG.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optG.step()
        with torch.no_grad():
            decay = min(0.999, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - decay)
        step += 1
        if step % 200 == 0 or step == 1:
            rec = dict(step=step, minutes=round((time.time() - t0) / 60, 2), its=round(step / (time.time() - t0), 2),
                       spec=round(spec.item(), 4), aux=round(aux.item(), 3), adv=round(float(adv), 3),
                       fm=round(float(fm), 3), d=round(float(d_loss), 3), d_real=round(float(d_real), 2),
                       d_fake=round(float(d_fake), 2), lr=round(cur, 6))
            hist.append(rec)
            log(json.dumps(rec))
        if on_ckpt is not None and step % ckpt_every == 0:
            on_ckpt(ema, step)

    # validation with the EMA weights
    g = torch.Generator(device=device).manual_seed(1)
    val = {}
    with torch.no_grad():
        losses, accs = [], []
        for i in range(0, n_val, 128):
            idx = val_idx[i:i + 128]
            x = audio[idx, :N].float() / 32768
            z = ema.encode(x)
            y = ema.decode(z, generator=g)
            for j in range(len(idx)):
                losses.append((int(cats[idx[j]]), mrstft(y[j:j + 1], x[j:j + 1]).item()))
            accs.append((ema.aux(z.flatten(1))[:, :n_cat].argmax(1) == cats[idx]).float())
        val["spec"] = sum(l for _, l in losses) / len(losses)
        val["spec_by_cat"] = {c: sum(l for k, l in losses if k == c) / max(1, sum(k == c for k, _ in losses))
                              for c in range(n_cat)}
        val["category_acc_from_latent"] = torch.cat(accs).mean().item()
    return dict(ema=ema, val=val, hist=hist, val_idx=val_idx.cpu(), steps=step,
                epochs=step * batch / len(train_idx))


@torch.no_grad()
def encode_all(ema, audio, device, bs=512):
    return torch.cat([ema.encode(audio[i:i + bs, :N].to(device).float() / 32768).half().cpu()
                      for i in range(0, len(audio), bs)])
