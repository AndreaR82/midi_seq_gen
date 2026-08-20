"""A small decoder-only Transformer (GPT-style), hand-written rather than
built from nn.TransformerDecoder/nn.MultiheadAttention, so the architecture
itself stays part of the "built from scratch" scope — only tensors/autograd
come from PyTorch.

At the default ModelConfig this is on the order of 1-2M parameters:
small enough to train on a laptop CPU and, eventually, to quantize for
cheap embedded hardware.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)  # B,H,T,D
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # B,H,T,T
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(causal, float("-inf"))
        if key_padding_mask is not None:
            # key_padding_mask: B,T, True where PAD -> mask those columns
            att = att.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        out = att @ v  # B,H,T,D
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(out))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), key_padding_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class MelodyTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig, pad_id: int) -> None:
        super().__init__()
        assert cfg.vocab_size > 0, "ModelConfig.vocab_size must be set from the tokenizer"
        self.cfg = cfg
        self.pad_id = pad_id

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # tied embeddings, standard param saver

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (B, T) token ids, PAD included. Returns logits (B, T, V)."""
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len, f"sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}"

        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        key_padding_mask = idx.eq(self.pad_id)  # B,T — True where PAD
        for block in self.blocks:
            x = block(x, key_padding_mask)
        x = self.ln_f(x)
        return self.head(x)

    @torch.no_grad()
    def generate(
        self,
        prefix_ids: list[int],
        max_new_tokens: int,
        eos_id: int,
        temperature: float = 1.0,
        top_p: float = 0.95,
        generator: torch.Generator | None = None,
        forbidden_ids: list[int] | None = None,
    ) -> list[int]:
        """Autoregressive sampling with temperature + nucleus (top-p)
        sampling. Different `generator` seeds give different melodies from
        the same prefix — this is the mechanism behind "press the button
        again for a new one".

        `forbidden_ids` are masked out at every generation step (logits set
        to -inf) — used to keep the LEN_<bars> control token to its one
        legitimate position in the prefix. Training data never places one
        elsewhere, but nothing architecturally stops an under-trained model
        from sampling one mid-sequence and silently overriding the
        requested length, so it's blocked explicitly rather than left to
        the model to learn not to do."""
        self.eval()
        device = next(self.parameters()).device
        ids = list(prefix_ids)

        for _ in range(max_new_tokens):
            window = ids[-self.cfg.max_seq_len :]
            x = torch.tensor([window], dtype=torch.long, device=device)
            logits = self(x)[0, -1] / max(temperature, 1e-5)
            if forbidden_ids:
                logits[forbidden_ids] = float("-inf")
            next_id = _sample_top_p(logits, top_p, generator)
            ids.append(next_id)
            if next_id == eos_id:
                break
        return ids


def _sample_top_p(logits: torch.Tensor, top_p: float, generator: torch.Generator | None) -> int:
    probs = F.softmax(logits, dim=-1)
    # A heavily masked or barely-trained/randomly-initialized model can
    # occasionally produce a distribution that's degenerate after slicing
    # (sums to ~0, or carries NaN/negative dust from floating-point error)
    # — torch.multinomial raises on that rather than handling it, so guard
    # explicitly and fall back to greedy rather than let sampling crash.
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0)
    if probs.sum().item() <= 0:
        return int(torch.argmax(logits).item())

    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # keep the smallest prefix of the sorted distribution whose cumulative
    # probability is >= top_p (always keep at least the top-1 token)
    cutoff = int((cumulative < top_p).sum().item()) + 1
    kept_probs = sorted_probs[:cutoff].clamp(min=0)
    kept_idx = sorted_idx[:cutoff]

    total = kept_probs.sum()
    if total.item() <= 0:
        return int(kept_idx[0].item())
    kept_probs = kept_probs / total

    choice = torch.multinomial(kept_probs, num_samples=1, generator=generator)
    return int(kept_idx[choice].item())
