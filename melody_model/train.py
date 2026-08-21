"""Train the small melody Transformer on a dataset written by
melody_model.data.prepare_dataset.

    python -m melody_model.train --dataset data/fragments.jsonl --out-dir checkpoints/wave2

Validation is held out by **source file** (see melody_model/split.py), not by
record, because the dataset contains many transposed and overlapping copies of
the same phrase. Every checkpoint records how it was produced, so two runs can
actually be compared.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .config import ModelConfig, TokenizerConfig, TrainConfig
from .model import MelodyTransformer
from .split import split_records
from .tokenizer import MelodyTokenizer


class FragmentDataset(Dataset):
    def __init__(self, records: list[dict]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> list[int]:
        return self.records[idx]["tokens"]


class LengthBucketBatchSampler(torch.utils.data.Sampler):
    """Batch fragments of similar length together.

    Lengths run from ~20 to 259 tokens with a median around 37, so a randomly
    drawn batch pads to roughly its longest member and wastes about 3x the
    compute on PAD. Shuffling within a large chunk, sorting that chunk by
    length, then shuffling the resulting batches keeps the randomness that
    matters (which fragments meet, and in what order batches arrive) while
    dropping most of the padding.
    """

    def __init__(self, lengths: list[int], batch_size: int, chunk_batches: int = 50, seed: int = 0) -> None:
        self.lengths = lengths
        self.batch_size = batch_size
        self.chunk_size = batch_size * chunk_batches
        self.epoch = 0
        self.seed = seed

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))
        rng.shuffle(indices)

        batches = []
        for start in range(0, len(indices), self.chunk_size):
            chunk = sorted(indices[start : start + self.chunk_size], key=lambda i: self.lengths[i])
            for bstart in range(0, len(chunk), self.batch_size):
                batches.append(chunk[bstart : bstart + self.batch_size])

        rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return math.ceil(len(self.lengths) / self.batch_size)


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


def load_dataset_meta(dataset_path: Path) -> dict:
    meta_path = dataset_path.with_suffix(dataset_path.suffix + ".meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(
            f"missing {meta_path} — datasets must be built with melody_model.data.prepare_dataset, "
            "which writes the tokenizer config alongside the fragments"
        )
    with meta_path.open() as f:
        return json.load(f)


def load_tokenizer_config(dataset_path: Path) -> TokenizerConfig:
    return TokenizerConfig.from_dict(load_dataset_meta(dataset_path)["tokenizer_config"])


def git_commit() -> str | None:
    """Best-effort provenance. A checkpoint trained from a dirty or
    non-git tree is still worth keeping, so failure is not an error."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=Path(__file__).resolve().parent.parent,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def save_checkpoint(
    path: Path,
    model: MelodyTransformer,
    model_cfg: ModelConfig,
    tok_cfg: TokenizerConfig,
    provenance: dict | None = None,
) -> None:
    """Weights plus everything needed to say where they came from.

    Wave 1 stored only the two configs, which made "is this checkpoint better
    than that one" unanswerable after the fact -- there was no val loss, no
    dataset, no split policy, nothing. Anything cheap enough to store goes in.
    """
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model_cfg.to_dict(),
            "tokenizer_config": tok_cfg.to_dict(),
            **(provenance or {}),
        },
        path,
    )


def lr_at(step: int, cfg: TrainConfig, total_steps: int) -> float:
    """Linear warmup then cosine decay to `lr * lr_min_ratio`."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
    progress = min(1.0, max(0.0, progress))
    floor = cfg.lr * cfg.lr_min_ratio
    return floor + 0.5 * (cfg.lr - floor) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model: MelodyTransformer, loader: DataLoader, pad_id: int, device: str) -> float:
    """Token-weighted mean loss.

    Averaging per batch would weight a batch of short fragments the same as a
    batch of long ones, so the number would drift with however the batches
    happened to fall. Weighting by non-PAD target count makes it comparable
    across runs and across batching strategies."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        inputs, targets = batch[:, :-1], batch[:, 1:]
        logits = model(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
            ignore_index=pad_id, reduction="sum",
        )
        n_tokens = int(targets.ne(pad_id).sum().item())
        total_loss += loss.item()
        total_tokens += n_tokens
    return total_loss / max(1, total_tokens)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"  # Apple Silicon GPU — no-op fallback to cpu elsewhere
    return "cpu"


def train(dataset_path: Path, out_dir: Path, model_cfg: ModelConfig, train_cfg: TrainConfig) -> None:
    torch.manual_seed(train_cfg.seed)
    random.seed(train_cfg.seed)

    dataset_meta = load_dataset_meta(dataset_path)
    tok_cfg = TokenizerConfig.from_dict(dataset_meta["tokenizer_config"])
    tokenizer = MelodyTokenizer(tok_cfg)
    records = load_dataset(dataset_path)
    if not records:
        raise ValueError(f"no fragments found in {dataset_path}")

    salt = str(train_cfg.seed)
    train_records, val_records, split_stats = split_records(records, train_cfg.val_fraction, salt=salt)
    if not train_records or not val_records:
        raise ValueError(
            f"split produced {len(train_records)} train / {len(val_records)} val records — "
            "the dataset needs at least two distinct source files"
        )

    model_cfg.vocab_size = tokenizer.vocab_size
    collate = make_collate_fn(tokenizer.pad_id)

    train_dataset = FragmentDataset(train_records)
    if train_cfg.bucket_batches:
        sampler = LengthBucketBatchSampler(
            [len(r["tokens"]) for r in train_records], train_cfg.batch_size, seed=train_cfg.seed
        )
        train_loader = DataLoader(train_dataset, batch_sampler=sampler, collate_fn=collate)
    else:
        sampler = None
        train_loader = DataLoader(
            train_dataset, batch_size=train_cfg.batch_size, shuffle=True, collate_fn=collate
        )
    val_loader = DataLoader(
        FragmentDataset(val_records), batch_size=train_cfg.batch_size, shuffle=False, collate_fn=collate
    )

    device = pick_device()
    model = MelodyTransformer(model_cfg, pad_id=tokenizer.pad_id).to(device)
    print(
        f"model params: {model.num_params():,} | vocab size: {tokenizer.vocab_size} | device: {device}\n"
        f"split (by source file, {train_cfg.val_fraction:.0%} held out): "
        f"{split_stats['train_files']} train files / {split_stats['val_files']} val files -> "
        f"{len(train_records):,} train / {len(val_records):,} val records"
    )
    for corpus, counts in sorted(split_stats["records_per_corpus"].items()):
        print(f"  {corpus}: {counts['train']:,} train / {counts['val']:,} val records")

    optim = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / "history.jsonl"
    history_path.write_text("")

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * train_cfg.epochs
    print(f"{steps_per_epoch:,} steps/epoch, {total_steps:,} total, warmup {train_cfg.warmup_steps}")

    base_provenance = {
        "dataset_path": str(dataset_path),
        "dataset_meta": dataset_meta,
        "split_stats": split_stats,
        "train_config": train_cfg.to_dict(),
        "num_params": model.num_params(),
        "git_commit": git_commit(),
    }

    step = 0
    best_val = float("inf")
    epochs_since_best = 0
    run_start = time.time()

    for epoch in range(train_cfg.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        epoch_start = time.time()
        epoch_loss, epoch_batches = 0.0, 0
        lr = train_cfg.lr

        for batch in train_loader:
            lr = lr_at(step, train_cfg, total_steps)
            for group in optim.param_groups:
                group["lr"] = lr

            batch = batch.to(device)
            inputs, targets = batch[:, :-1], batch[:, 1:]
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=tokenizer.pad_id
            )

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optim.step()

            epoch_loss += loss.item()
            epoch_batches += 1
            step += 1
            if step % train_cfg.log_every == 0:
                print(f"epoch {epoch} step {step}/{total_steps} lr {lr:.2e} train_loss {loss.item():.4f}")

        val_loss = evaluate(model, val_loader, tokenizer.pad_id, device)
        elapsed = time.time() - epoch_start
        improved = val_loss < best_val
        print(
            f"epoch {epoch} val_loss {val_loss:.4f} "
            f"(train {epoch_loss / max(1, epoch_batches):.4f}, {elapsed:.0f}s)"
            f"{' *best*' if improved else ''}"
        )

        with history_path.open("a") as f:
            f.write(json.dumps({
                "epoch": epoch,
                "step": step,
                "train_loss": epoch_loss / max(1, epoch_batches),
                "val_loss": val_loss,
                "lr": lr,
                "seconds": round(elapsed, 1),
            }) + "\n")

        if improved:
            best_val = val_loss
            epochs_since_best = 0
            save_checkpoint(out_dir / "best.pt", model, model_cfg, tok_cfg, {
                **base_provenance,
                "val_loss": val_loss,
                "epoch": epoch,
                "step": step,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            })
        else:
            epochs_since_best += 1
            if epochs_since_best >= train_cfg.patience:
                print(
                    f"early stop: no val improvement in {train_cfg.patience} epochs "
                    f"(best {best_val:.4f})"
                )
                break

    save_checkpoint(out_dir / "last.pt", model, model_cfg, tok_cfg, {
        **base_provenance,
        "val_loss": val_loss,
        "epoch": epoch,
        "step": step,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    })
    print(f"done in {(time.time() - run_start) / 60:.1f} min — best val_loss {best_val:.4f}, checkpoints in {out_dir}")


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
    parser.add_argument("--warmup-steps", type=int, default=TrainConfig.warmup_steps)
    parser.add_argument("--patience", type=int, default=TrainConfig.patience)
    parser.add_argument("--val-fraction", type=float, default=TrainConfig.val_fraction)
    parser.add_argument("--no-bucket", action="store_true", help="disable length-bucketed batching")
    args = parser.parse_args()

    model_cfg = ModelConfig(d_model=args.d_model, n_layer=args.n_layer, n_head=args.n_head, d_ff=args.d_ff, dropout=args.dropout)
    train_cfg = TrainConfig(
        batch_size=args.batch_size, lr=args.lr, epochs=args.epochs, seed=args.seed,
        warmup_steps=args.warmup_steps, patience=args.patience, val_fraction=args.val_fraction,
        bucket_batches=not args.no_bucket,
    )
    train(args.dataset, args.out_dir, model_cfg, train_cfg)


if __name__ == "__main__":
    main()
