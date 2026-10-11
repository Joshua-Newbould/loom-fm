import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoomConfig:
    in_ch: int = 3; img: int = 32; patch: int = 2; dim: int = 256; depth: int = 4; loops: int = 1; heads: int = 4
    experts: int = 0; topk: int = 2; expert_mult: float = 1.0
    res_scale: bool = False; reinject: bool = False; loop_residual: bool = False; per_loop_router: bool = False; moe_norm: bool = False
    seg: int = 0; lam: float = 0.5; beta: float = 0.5; per_loop_adaln: bool = False; num_classes: int = 10


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        xf = x.float()
        return (self.weight * xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def sincos_2d(dim, g):
    def one(d, pos):
        om = 1.0 / 10000 ** (torch.arange(d // 2, dtype=torch.float64) / (d / 2))
        o = pos.reshape(-1).double()[:, None] * om[None]
        return torch.cat([o.sin(), o.cos()], 1)
    ys, xs = torch.meshgrid(torch.arange(g), torch.arange(g), indexing='ij')
    return torch.cat([one(dim // 2, ys), one(dim // 2, xs)], 1).float()[None]


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden, bias=True):
        super().__init__()
        self.w12, self.w3 = nn.Linear(dim, 2 * hidden, bias=bias), nn.Linear(hidden, dim, bias=bias)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, -1)
        return self.w3(F.silu(a) * b)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(self.q_norm(q), self.k_norm(k), v)
        return self.proj(o.transpose(1, 2).reshape(B, N, C))


class Layer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(c.dim), RMSNorm(c.dim)
        self.attn = Attention(c.dim, c.heads)
        self.moe_norm = RMSNorm(c.dim) if c.moe_norm else nn.Identity()
        self.ffn = (nn.ModuleList(SwiGLU(c.dim, int(c.expert_mult * c.dim), bias=False) for _ in range(c.experts)) if c.experts
                    else SwiGLU(c.dim, int(4 * c.dim * 2 / 3)))


class LoomNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg, D, M, H = cfg, cfg.dim, cfg.depth, cfg.loops
        self.gamma = cfg.lam / (H * math.sqrt(M)) if cfg.res_scale else 1.0
        self.patch_embed = nn.Conv2d(cfg.in_ch, D, cfg.patch, cfg.patch)
        self.register_buffer('pos', sincos_2d(D, cfg.img // cfg.patch), persistent=False)
        self.t_mlp = nn.Sequential(nn.Linear(256, D), nn.SiLU(), nn.Linear(D, D))
        self.y_emb = nn.Embedding(cfg.num_classes + 1, D)
        self.layers = nn.ModuleList(Layer(cfg) for _ in range(M))
        self.adaln = nn.ModuleList(nn.ModuleList(nn.Sequential(nn.SiLU(), nn.Linear(D, 6 * D)) for _ in range(M))
                                   for _ in range(H if cfg.per_loop_adaln else 1))
        self.routers = nn.ModuleList(nn.Linear(D, cfg.experts, bias=False) for _ in range(M * (H if cfg.per_loop_router else 1))) if cfg.experts else nn.ModuleList()
        self.final_norm = RMSNorm(D)
        self.final_mod = nn.Sequential(nn.SiLU(), nn.Linear(D, 2 * D))
        self.final_lin = nn.Linear(D, cfg.patch ** 2 * cfg.in_ch)
        self.apply(lambda m: isinstance(m, nn.Linear) and (nn.init.xavier_uniform_(m.weight), m.bias is not None and nn.init.zeros_(m.bias)))
        nn.init.normal_(self.y_emb.weight, std=0.02)
        for m in (self.t_mlp[0], self.t_mlp[2]):
            nn.init.normal_(m.weight, std=0.02)
        # near-zero rather than exactly zero: exact zeros would block all router gradients at init (check.py unit requires nonzero)
        for m in [s[-1] for a in self.adaln for s in a] + [self.final_mod[-1], self.final_lin]:
            nn.init.normal_(m.weight, std=1e-3)

    def _t_emb(self, t):
        f = torch.exp(-math.log(10000) * torch.arange(128, device=t.device, dtype=torch.float32) / 128)
        a = t[:, None].float() * f[None]
        return self.t_mlp(torch.cat([a.cos(), a.sin()], -1))

    def _moe(self, l, r, x):
        c, B = self.cfg, x.shape[0]
        if not c.experts:
            return self.layers[l].ffn(x), x.new_zeros(0), x.new_zeros(0)
        X = x.reshape(-1, x.shape[-1])
        with torch.autocast(x.device.type, enabled=False):
            p = F.softmax(F.linear(X.float(), self.routers[r].weight), -1)
        w, idx = p.topk(c.topk, -1)
        w = w / w.sum(-1, keepdim=True)
        out = torch.zeros_like(X)
        for e, ex in enumerate(self.layers[l].ffn):
            tok, slot = (idx == e).nonzero(as_tuple=True)
            if tok.numel():
                out.index_add_(0, tok, (ex(X[tok]) * w[tok, slot, None]).to(out.dtype))
        f = F.one_hot(idx, c.experts).sum(1).float().mean(0) / c.topk
        return out.reshape_as(x), p.mean(0), f

    def _run(self, z, t, y, all_segs=False, stats=False):
        c, H, M = self.cfg, self.cfg.loops, self.cfg.depth
        K = c.seg or H
        cond = self._t_emb(t) + self.y_emb(y)
        x = self.patch_embed(z).flatten(2).transpose(1, 2) + self.pos
        h, outs, aux = x, [], []
        S = dict(x=x.detach().float(), loop_in=[], loop_out=[], attn_out={}, resid={}, rms=[]) if stats else None
        loads = x.new_zeros(H, M, c.experts, dtype=torch.float32)
        Ng = Dg = 0
        for i in range(H):
            if i and c.reinject:
                g = c.lam / ((i + 1) * math.sqrt(M))
                h = (1 - g) * h + g * x
            if stats:
                S['loop_in'].append(h.detach().float())
            Nl = Dl = 0
            for l, L in enumerate(self.layers):
                sm, sc, gm, sm2, sc2, gm2 = self.adaln[i if c.per_loop_adaln else 0][l](cond).chunk(6, -1)
                o = self.gamma * gm.unsqueeze(1) * L.attn(modulate(L.norm1(h), sm, sc))
                if c.loop_residual:
                    Ng, Dg, Nl, Dl = c.beta * Ng + o, c.beta * Dg + 1, c.beta * Nl + o, c.beta * Dl + 1
                    r = Ng / Dg + Nl / Dl
                else:
                    r = o
                if stats:
                    S['attn_out'][(i, l)], S['resid'][(i, l)] = o.detach().float(), r.detach().float()
                ht = h + r
                m, P, f = self._moe(l, l + M * i if c.per_loop_router else l, modulate(L.norm2(ht), sm2, sc2))
                h = ht + self.gamma * gm2.unsqueeze(1) * L.moe_norm(m)
                if c.experts:
                    loads[i, l] = P.detach()
                    aux.append(c.experts * (P * f).sum())
            if stats:
                S['loop_out'].append(h.detach().float())
                S['rms'].append(h.detach().float().pow(2).mean().sqrt().item())
            if i == H - 1 or (all_segs and (i + 1) % K == 0):
                sh, sc = self.final_mod(cond).chunk(2, -1)
                o = self.final_lin(modulate(self.final_norm(h), sh, sc))
                G, p = c.img // c.patch, c.patch
                o = o.reshape(-1, G, G, p, p, c.in_ch).permute(0, 5, 1, 3, 2, 4).reshape(-1, c.in_ch, c.img, c.img)
                outs.append(o)
            if self.training and (i + 1) % K == 0:
                h, Ng = h.detach(), (Ng.detach() if torch.is_tensor(Ng) else Ng)
        if stats:
            S['loads'] = loads
        return outs, (torch.stack(aux).mean() if aux else x.new_zeros(())), S

    def forward(self, z, t, y, stats=False):
        outs, _, S = self._run(z, t, y, stats=stats)
        return (outs[-1], S) if stats else outs[-1]

    def loss(self, x, y):
        B = x.shape[0]
        t = torch.sigmoid(torch.randn(B, device=x.device) * 0.8 - 0.8).view(B, 1, 1, 1)
        y = torch.where(torch.rand(B, device=x.device) < 0.1, torch.full_like(y, self.cfg.num_classes), y)
        z = t * x + (1 - t) * torch.randn_like(x)
        outs, aux, _ = self._run(z, t.flatten(), y, all_segs=True)
        d = (1 - t).clamp(min=0.05)
        seg = [(((x - z) / d - (o.float() - z) / d) ** 2).mean() for o in outs]
        fm = torch.stack(seg).mean()
        return fm + 0.01 * aux, dict(fm=fm.detach(), aux=aux.detach(), seg_losses=seg)

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def n_active(self):
        cnt = lambda m: sum(p.numel() for p in m.parameters())
        c, M = self.cfg, self.cfg.depth
        per = cnt(self.adaln[0]) + sum(cnt(L) - cnt(L.ffn) + cnt(L.ffn) * (c.topk / c.experts if c.experts else 1) for L in self.layers)
        per += c.dim * c.experts * M
        return int(self.n_params() - cnt(self.layers) - cnt(self.adaln) - cnt(self.routers) + c.loops * per)
