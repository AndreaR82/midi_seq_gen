"""Train the small melody Transformer on a dataset written by
melody_model.data.prepare_dataset.

    python -m melody_model.train --dataset data/fragments.jsonl --out-dir checkpoints/run1
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .config import ModelConfig, TokenizerConfig, TrainConfig
from .model import MelodyTransformer
from .tokenizer import MelodyTokenizer


class FragmentDataset(Dataset):
    def __init__(self, records: list[dict]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> list[int]:
        return self.records[idx]["tokens"]


def make_collate_fn(pad_id: int):
    def collate(batch: list[list[int]]) -> torch.Tensor:
        max_len = max(len(x) for x in batch)
        out = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        for i, ids in enumerate(batch):
            out[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        return out

    return collate


def load_dataset(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_tokenizer_config(dataset_path: Path) -> TokenizerConfig:
    meta_path = dataset_path.with_suffix(dataset_path.suffix + ".meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(
            f"missing {meta_path} — datasets must be built with melody_model.data.prepare_dataset, "
            "which writes the tokenizer config alongside the fragments"
        )
    with meta_path.open() as f:
        meta = json.load(f)
    return TokenizerConfig.from_dict(meta["tokenizer_config"])


def save_checkpoint(path: Path, model: MelodyTransformer, model_cfg: ModelConfig, tok_cfg: TokenizerConfig) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model_cfg.to_dict(),
            "tokenizer_config": tok_cfg.to_dict(),
        },
        path,
    )


@torch.no_grad()
def evaluate(model: MelodyTransformer, loader: DataLoader, pad_id: int, device: str) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        inputs, targets = batch[:, :-1], batch[:, 1:]
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=pad_id)
        total += loss.item()
        count += 1
    return total / max(1, count)


def train(dataset_path: Path, out_dir: Path, model_cfg: ModelConfig, train_cfg: TrainConfig) -> None:
    torch.manual_seed(train_cfg.seed)
    random.seed(train_cfg.seed)

    tok_cfg = load_tokenizer_config(dataset_path)
    tokenizer = MelodyTokenizer(tok_cfg)
    records = load_dataset(dataset_path)
    if not records:
        raise ValueError(f"no fragments found in {dataset_path}")
    random.shuffle(records)

    n_val = max(1, int(len(records) * train_cfg.val_fraction))
    val_records, train_records = records[:n_val], records[n_val:]

    model_cfg.vocab_size = tokenizer.vocab_size
    collate = make_collate_fn(tokenizer.pad_id)
    train_loader = DataLoader(FragmentDataset(train_records), batch_size=train_cfg.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(FragmentDataset(val_records), batch_size=train_cfg.batch_size, shuffle=False, collate_fn=collate)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon GPU — no-op fallback to cpu elsewhere
    else:
        device = "cpu"
    model = MelodyTransformer(model_cfg, pad_id=tokenizer.pad_id).to(device)
    print(
        f"model params: {model.num_params():,} | vocab size: {tokenizer.vocab_size} | "
        f"train fragments: {len(train_records)} | val fragments: {len(val_records)} | device: {device}"
    )

    optim = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    out_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    best_val = float("inf")
    for epoch in range(train_cfg.epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            inputs, targets = batch[:, :-1], batch[:, 1:]
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=tokenizer.pad_id)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optim.step()

            step += 1
            if step % train_cfg.log_every == 0:
                print(f"epoch {epoch} step {step} train_loss {loss.item():.4f}")

        val_loss = evaluate(model, val_loader, tokenizer.pad_id, device)
        print(f"epoch {epoch} val_loss {val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(out_dir / "best.pt", model, model_cfg, tok_cfg)

    save_checkpoint(out_dir / "last.pt", model, model_cfg, tok_cfg)
    print(f"done — checkpoints in {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--d-model", type=int, default=ModelConfig.d_model)
    parser.add_argument("--n-layer", type=int, default=ModelConfig.n_layer)
    parser.add_argument("--n-head", type=int, default=ModelConfig.n_head)
    parser.add_argument("--d-ff", type=int, default=ModelConfig.d_ff)
    parser.add_argument("--dropout", type=float, default=ModelConfig.dropout)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    args = parser.parse_args()

    model_cfg = ModelConfig(d_model=args.d_model, n_layer=args.n_layer, n_head=args.n_head, d_ff=args.d_ff, dropout=args.dropout)
    train_cfg = TrainConfig(batch_size=args.batch_size, lr=args.lr, epochs=args.epochs, seed=args.seed)
    train(args.dataset, args.out_dir, model_cfg, train_cfg)


if __name__ == "__main__":
    main()
