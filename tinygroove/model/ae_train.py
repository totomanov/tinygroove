"""Training for the autoencoder, in two stages.

Stage 1 (`train_ae`): encoder and decoder together, spectral loss only. This fixes the latent.
Stage 2 (`train_decoder`): encoder frozen, a decoder trained on the stored latents with spectral + adversarial
losses. Because the latent does not move, any decoder from stage 2 can replace the stage 1 decoder, a decoder
of a different width can be trained on the same latents, and a failed adversarial run costs nothing downstream.

Why two stages: trained together in one stage, the discriminators of an earlier version sat at chance from step
10,000 on and the adversarial loss never contributed. Here they get a stage of their own, keep their learning
rate, and include waveform (multi-period) discriminators; and the run reports, and stops using them, if they
sit at chance again.
"""
import copy, json, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tinygroove.model.ae_model import AE, Decoder, N, SR, frames_of, stft

FFTS = (2048, 1024, 512, 256, 128, 64)


def _mag(x, n):
    X = torch.stft(x, n, n // 4, window=torch.hann_window(n, device=x.device), return_complex=True)
    return torch.sqrt(X.real ** 2 + X.imag ** 2 + 1e-9)


def mrstft(y, x):
    """Multi-resolution spectral distance (log and compressed magnitudes); the first 50 ms count 5x."""
    total = 0.0
    for n in FFTS:
        Y, X = _mag(y, n), _mag(x, n)
        t = torch.arange(X.shape[-1], device=x.device) * (n // 4) / SR
        w = 1 + 4 * torch.exp(-t / 0.05)
        w = w / w.mean()
        eps = n * 1e-4
        total = total + ((torch.log(Y + eps) - torch.log(X + eps)).abs().mean(1) * w).mean() \
                      + (((Y + eps) ** 0.3 - (X + eps) ** 0.3).abs().mean(1) * w).mean()
    return total / len(FFTS)

D_CROP, D_BATCH = 32768, 32


def brightness(audio, device, bs=1024):
    """log2 spectral centroid per clip (audio int16 [M, >=N])."""
    out = []
    for i in range(0, len(audio), bs):
        P = stft(audio[i:i + bs, :N].to(device).float() / 32768).abs() ** 2
        f = torch.linspace(0, SR / 2, P.shape[1], device=device)[None, :, None]
        out.append(torch.log2((P * f).sum((1, 2)) / (P.sum((1, 2)) + 1e-9) + 1.0))
    return torch.cat(out)


@torch.no_grad()
def val_spec(decode, audio, val_idx, cats, n_cat, bs=128):
    """Held-out multi-resolution spectral loss, overall and per category. decode(idx, x) -> y."""
    losses = []
    for i in range(0, len(val_idx), bs):
        idx = val_idx[i:i + bs]
        x = audio[idx, :N].float() / 32768
        y = decode(idx, x)
        losses += [(int(cats[idx[j]]), mrstft(y[j:j + 1], x[j:j + 1]).item()) for j in range(len(idx))]
    by = {c: sum(l for k, l in losses if k == c) / max(1, sum(k == c for k, _ in losses)) for c in range(n_cat)}
    return sum(l for _, l in losses) / len(losses), by


def train_ae(audio, cats, length_ms, val_idx, mode, max_s, device, steps=10 ** 9, batch=96, lr=4e-4, log=print, seed=0):
    """Stage 1. audio int16 [M, >=N] on device. Returns the EMA model, held-out loss and history."""
    torch.manual_seed(seed)
    M = len(audio)
    keep = torch.ones(M, dtype=torch.bool, device=device); keep[val_idx] = False
    train_idx = torch.nonzero(keep)[:, 0]
    frames = frames_of(length_ms.to(device))
    knobs = torch.stack([torch.log2(length_ms.float().to(device)), brightness(audio, device)], 1)
    knobs = (knobs - knobs[train_idx].mean(0)) / (knobs[train_idx].std(0) + 1e-6)
    n_cat = int(cats.max()) + 1
    cats = cats.to(device)
    model = AE(mode, n_categories=n_cat).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr, betas=(0.8, 0.99), weight_decay=0.01)
    t0, step, hist = time.time(), 0, []
    while True:
        p = max(step / steps, (time.time() - t0) / max_s)
        if p >= 1:
            break
        cur = lr * min(1.0, (step + 1) / 300) * (0.5 * (1 + math.cos(math.pi * p)) * 0.98 + 0.02)
        for g in opt.param_groups:
            g["lr"] = cur
        idx = train_idx[torch.randint(len(train_idx), (batch,), device=device)]
        x = audio[idx, :N].float() / 32768
        z = model.encode(x, frames[idx])
        zn = z + torch.randn_like(z) * 0.1 * torch.rand(batch, 1, 1, device=device)       # the decoder must tolerate imperfect latents
        y = model.decode(zn, frames[idx])
        spec = mrstft(y, x)
        a = model.aux(z.flatten(1))
        aux = F.cross_entropy(a[:, :n_cat], cats[idx]) + F.mse_loss(a[:, n_cat:], knobs[idx])
        loss = 15 * spec + 0.1 * aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        with torch.no_grad():
            d = min(0.999, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - d)
        step += 1
        if step % 500 == 0 or step == 1:
            rec = dict(step=step, minutes=round((time.time() - t0) / 60, 2), its=round(step / (time.time() - t0), 1),
                       spec=round(spec.item(), 4), aux=round(aux.item(), 3))
            hist.append(rec); log(json.dumps(rec))
    spec_all, by = val_spec(lambda idx, x: ema.decode(ema.encode(x, frames[idx]), frames[idx]), audio, val_idx, cats, n_cat)
    return dict(ema=ema, val_spec=spec_all, val_by_cat=by, hist=hist, steps=step)


@torch.no_grad()
def encode_all(ema, audio, length_ms, device, bs=512):
    frames = frames_of(length_ms.to(device))
    return torch.cat([ema.encode(audio[i:i + bs, :N].float() / 32768, frames[i:i + bs]).half() for i in range(0, len(audio), bs)])


# ---------------------------------------------------------------- stage 2: discriminators

wn = nn.utils.parametrizations.weight_norm


class SpecD(nn.Module):
    """One spectrogram resolution. Input: compressed complex spectrogram plus log magnitude."""
    def __init__(self, n, ch=48):
        super().__init__()
        self.n = n
        self.convs = nn.ModuleList([wn(nn.Conv2d(3, ch, (3, 9), stride=(1, 2), padding=(1, 4)))]
                                   + [wn(nn.Conv2d(ch, ch, (3, 9), stride=(1, 2), padding=(1, 4))) for _ in range(3)]
                                   + [wn(nn.Conv2d(ch, ch, 3, padding=1))])
        self.out = wn(nn.Conv2d(ch, 1, 3, padding=1))

    def forward(self, x):
        X = torch.stft(x.float(), self.n, self.n // 4, window=torch.hann_window(self.n, device=x.device), return_complex=True)
        re, im = X.real / self.n ** 0.5, X.imag / self.n ** 0.5
        mag = torch.sqrt(re ** 2 + im ** 2 + 1e-9)
        gain = (mag + 1e-4) ** 0.3 / (mag + 1e-4)
        h = torch.stack([re * gain, im * gain, torch.log(mag + 1e-4) / 4 + 1], 1).transpose(2, 3)
        feats = []
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            for c in self.convs:
                h = F.leaky_relu(c(h), 0.1); feats.append(h.float())
            return self.out(h).float(), feats


class PeriodD(nn.Module):
    """Waveform folded at one period (as in HiFi-GAN): sees phase and transients, which spectrogram magnitudes hide."""
    def __init__(self, p, chs=(32, 64, 128, 128)):
        super().__init__()
        self.p = p
        cin, convs = 1, []
        for c in chs:
            convs.append(wn(nn.Conv2d(cin, c, (5, 1), (3, 1), padding=(2, 0)))); cin = c
        self.convs = nn.ModuleList(convs + [wn(nn.Conv2d(cin, cin, (5, 1), 1, padding=(2, 0)))])
        self.out = wn(nn.Conv2d(cin, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        B, L = x.shape
        pad = (self.p - L % self.p) % self.p
        h = F.pad(x.float(), (0, pad)).reshape(B, 1, (L + pad) // self.p, self.p)
        feats = []
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            for c in self.convs:
                h = F.leaky_relu(c(h), 0.1); feats.append(h.float())
            return self.out(h).float(), feats


def train_decoder(audio, cats, length_ms, z_all, val_idx, mode, dec_c, max_s, device, init=None, steps=10 ** 9,
                  batch=64, lr=2e-4, gan_start=0.12, log=print, seed=0):
    """Stage 2. z_all: stored latents [M, K, D] from the frozen stage 1 encoder. Returns the EMA decoder."""
    torch.manual_seed(seed)
    M = len(audio)
    keep = torch.ones(M, dtype=torch.bool, device=device); keep[val_idx] = False
    train_idx = torch.nonzero(keep)[:, 0]
    frames = frames_of(length_ms.to(device))
    n_cat = int(cats.max()) + 1
    dec = Decoder(dec_c, mode).to(device)
    if init is not None:
        dec.load_state_dict(init)
    ema = copy.deepcopy(dec).eval().requires_grad_(False)
    discs = nn.ModuleList([SpecD(n) for n in (2048, 512, 128)] + [PeriodD(p) for p in (2, 3, 5, 7, 11)]).to(device)
    optG = torch.optim.AdamW(dec.parameters(), lr, betas=(0.8, 0.99), weight_decay=0.01)
    optD = torch.optim.AdamW(discs.parameters(), lr, betas=(0.8, 0.99), weight_decay=0.0)
    t0, step, hist, gap_log, gan_failed = time.time(), 0, [], [], False
    while True:
        p = max(step / steps, (time.time() - t0) / max_s)
        if p >= 1:
            break
        for g in optG.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / 300) * (0.5 * (1 + math.cos(math.pi * p)) * 0.9 + 0.1)
        idx = train_idx[torch.randint(len(train_idx), (batch,), device=device)]
        x = audio[idx, :N].float() / 32768
        z = z_all[idx].float()
        y = dec(z + torch.randn_like(z) * 0.1 * torch.rand(batch, 1, 1, device=device), frames[idx])
        use_gan = p > gan_start and not gan_failed
        d_loss = gap = torch.zeros((), device=device)
        xd, yd = x[:D_BATCH, :D_CROP], y[:D_BATCH, :D_CROP]      # the discriminators are the expensive part
        if use_gan:
            sr = [d(xd)[0] for d in discs]
            sf = [d(yd.detach())[0] for d in discs]
            d_loss = sum(F.relu(1 - a).mean() + F.relu(1 + b).mean() for a, b in zip(sr, sf))
            gap = sum(a.mean() - b.mean() for a, b in zip(sr, sf)) / len(discs)
            optD.zero_grad(set_to_none=True); d_loss.backward(); optD.step()
            gap_log.append(float(gap))
        spec = mrstft(y, x)
        adv = fm = torch.zeros((), device=device)
        if use_gan:
            for d in discs:
                sy, fy = d(yd)
                with torch.no_grad():
                    _, fx = d(xd)
                adv = adv - sy.mean()
                fm = fm + sum(F.l1_loss(u, v) for u, v in zip(fy, fx)) / len(fy)
        loss = 15 * spec + adv + 2 * fm
        optG.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(dec.parameters(), 5.0)
        optG.step()
        with torch.no_grad():
            d_ = min(0.999, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), dec.parameters()):
                pe.lerp_(pm, 1 - d_)
        step += 1
        # discriminators at chance for a long stretch: say so, and carry on with the spectral loss alone
        if use_gan and p > 0.5 and len(gap_log) > 2000 and sum(gap_log[-2000:]) / 2000 < 0.02:
            gan_failed = True
            log(json.dumps(dict(step=step, note="discriminators at chance: adversarial loss switched off")))
        if step % 500 == 0 or step == 1:
            rec = dict(step=step, minutes=round((time.time() - t0) / 60, 2), its=round(step / (time.time() - t0), 1),
                       spec=round(spec.item(), 4), adv=round(float(adv), 3), fm=round(float(fm), 3),
                       d=round(float(d_loss), 3), gap=round(float(gap), 3), chance=2 * len(discs))
            hist.append(rec); log(json.dumps(rec))
    spec_all, by = val_spec(lambda idx, x: ema(z_all[idx].float(), frames[idx]), audio, val_idx, cats.to(device), n_cat)
    recent = gap_log[-2000:]
    return dict(ema=ema, val_spec=spec_all, val_by_cat=by, hist=hist, steps=step, gan_failed=gan_failed,
                gap_last=sum(recent) / max(1, len(recent)))
