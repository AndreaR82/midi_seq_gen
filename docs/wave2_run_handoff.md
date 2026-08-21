# Wave 2 training run — handoff

Status as of 2026-08-22: **all code is ready, the run has not been done.**
It was attempted twice on an Apple M1 / 8 GB and abandoned both times. Pick it
up on the DGX Spark.

## Why the run moved machines

The M1 has 8 GB of unified memory and a ~9.07 GB MPS allocation ceiling.

1. **First attempt: MPS OOM.** Fixed batch size is the wrong unit when
   attention allocates `B x H x T x T` per layer. Now fixed — batching is by
   padded-token budget (`--max-tokens`), so batch size falls out of sequence
   length. Peak memory measured flat at 3.11 GB across the whole length
   distribution.
2. **Second attempt: swap thrashing.** 21 minutes without completing 50 steps,
   i.e. **>25 s/step against 1.6-2.1 s/step benchmarked on the same GPU.** The
   machine was at 9.1 GB of 10.2 GB swap with 56M pageins — 964k records as
   Python dicts plus the model plus MPS buffers does not fit in 8 GB. One epoch
   would have taken ~54 hours.

The GPU was never the bottleneck; memory capacity was. This is the specific
thing the Spark fixes.

## What you need that git does not carry

`/data/`, `/checkpoints/` and `/generated/` are gitignored, so **the dataset
and the Wave 1 checkpoint do not travel with the repo**:

- `data/fragments.jsonl` (227 MB) + `data/fragments.jsonl.meta.json` — either
  copy them across, or rebuild from scratch following
  [`melody_model_data_prep.md`](melody_model_data_prep.md) §9 (two shallow
  clones plus one `prepare_dataset` run, ~57 s).
- `checkpoints/wave1/best.pt` (3.4 MB) — only needed to re-measure the Wave 1
  baseline yourself. The numbers are already recorded below.

The train/val split is a **stable `hashlib` digest of the source path**, so it
reproduces identically on any machine and any seed-0 run stays comparable.

## Setup on the Spark

PyTorch needs the **aarch64 CUDA build** — a default `pip install torch` gets
you a CPU-only wheel on Arm. Verify before launching anything:

```sh
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`pick_device()` in `melody_model/train.py` already prefers `cuda`, so no code
change is needed once that prints `True`.

## Do this first: measure, don't assume

The Mac managed **4,256 tok/s at T=259 and 5,845 tok/s at T=37** on a 10.8M
model. That is roughly 380 GFLOP/s — this workload is *launch-bound, not
compute-bound*, because the model is small and the sequences are short
(median 37 tokens, max 259). So raw TFLOPS will not predict the speedup.
Benchmark it before committing to a long run.

Worth trying, in this order, since they target launch overhead specifically:
`torch.compile`, bf16 autocast, `DataLoader(num_workers=...)`. None are
implemented yet.

## The run

Do not just copy the Mac flags. With 128 GB the token budget should go up an
order of magnitude, and `--batch-size` becomes the binding cap for short
fragments, so raise both:

```sh
python -m melody_model.train --dataset data/fragments.jsonl --out-dir checkpoints/wave2 \
    --d-model 384 --n-layer 6 --n-head 6 --d-ff 1536 --dropout 0.2 \
    --batch-size 512 --max-tokens 65536 --lr 6e-4 --epochs 4 --patience 2
```

**`--lr` is coupled to `--batch-size`.** The 3e-4 default was chosen for
batch 128; at 512 it wants roughly 6e-4. If you change the batch size again,
change the LR with it — otherwise the comparison against the numbers below is
measuring the LR, not the model.

Config decided earlier: same corpus (POP909 + Nottingham) so this run isolates
the recipe fix, ~10.8M params, 4 epochs.

## Then evaluate

```sh
python -m melody_model.eval --checkpoint checkpoints/wave2/best.pt --dataset data/fragments.jsonl
```

Wave 1's measured profile is the bar. It is a **0.85M-param** model, so beating
it on loss is not the interesting part — these are:

| metric | Wave 1 generated | held-out real music |
|---|---|---|
| LEN_ respected | 60.0% | 93.5% |
| ...overran | **36.7%** | 0.0% |
| in-key ratio | 99.9% | 99.0% |
| notes/bar | 5.34 | 4.31 |
| degeneracy (any) | 0.0% | 0.1% |
| copy rate | **4.5%** | 1.9% |

- **Length overrun is the clearest target.** Wave 1 runs past its requested bar
  count on over a third of generations; real music never does.
- **Copy rate is the risk.** 4.5% against a 1.9% floor is mild memorization at
  0.85M params. Wave 2 is 13x larger on the same ~52k unique phrases. If this
  climbs sharply, the model is too big for the corpus — drop `--d-model` /
  `--n-layer`, or expand the corpus (Wave 3), rather than accepting it.
- Caveat on the Wave 1 row: it was trained under the old random-record split,
  so it has likely seen some of what the by-file split now holds out. Its
  generated-sample metrics are unaffected, but its 4.5% copy rate is probably
  flattering and the real gap may be wider.

Finally, listen to it — `melody_model.sample` into `generated/wave2`, then Load
and Preview in the UI.
