# FinLLM-Post-Training

> A **design + build plan** for distilling market-analysis capability into a small
> LLM, then aligning it with **realized returns** (SFT → DPO → RLVR).

**Status: design only.** This repo is the architecture and a build plan — no code
yet, and no model trained. Code gets added as I learn the material and work through
the plan; the roadmap below tracks the work (nothing checked yet).

## What this is

A finance-flavored LLM post-training pipeline that turns a strong "teacher" (any
analyst LLM) into a fast, cheap student, then keeps improving the student against
what the market *actually did*. Three ideas drive it:

- **Markdown-first distillation** — teacher analyses are gathered as plain
  markdown, so anyone with a subscription CLI (no per-token API) can contribute.
- **Data-vs-future contract** — the model only ever sees the past; the realized
  future is a reward label, never an input.
- **RLVR on realized returns** — in the final stage the teacher is irrelevant; the
  student is graded against the real forward return.

## Design highlights

- **Runs on consumer hardware.** A **Qwen3.5-4B** student on an RTX 4060 Ti (16GB):
  QLoRA keeps weights at ~4–6 GB, leaving the budget for **context** — long evidence
  generations, heaviest at RL (GRPO's *G* on-policy completions, KV ∝ G × context).
  > Note: Qwen3.5-4B has a native 256K context window — always cap `max_seq_length`
  > (e.g. 4096–8192) in training and serving, or VRAM will explode.
- **No from-scratch pretraining.** Start from an open base (Qwen3.5-4B), then
  SFT → DPO → RLVR. Pretraining is out of scope — it costs millions.
- **Leakage-safe by construction.** Evidence windows and realized outcomes live in
  separate files; prompts are built data-only.
- **Contributable teachers.** Copy a template folder, let subagents fill the
  analyses — community-sourced distillation at zero API cost.

## Pipeline

```
aipa OHLCV + MA scores
   │  questions (data-only) + outcomes.jsonl (realized future)
   ▼
teacher markdown  ──►  score (verdict vs outcome)
   │                       │
   ▼                       ▼
SFT (imitate)  ──►  DPO (prefer what worked)  ──►  RLVR (reward vs reality)
   Unsloth + QLoRA      TRL DPOTrainer            TRL GRPOTrainer
   Qwen3.5-4B                                      (teacher drops out)
```

## Roadmap

- [ ] Architecture & design
- [ ] Data factory: questions + realized-outcome index from aipa OHLCV/MA
- [ ] Markdown teacher-gathering flow (copy-and-fill template)
- [ ] Scorer: direction + confidence core, target/invalidation path layer (MFE/MAE)
- [ ] SFT / DPO / eval / RLVR scripts
- [ ] Scale the question set (hundreds → thousands of `(ticker, as_of)` windows)
- [ ] Gather teacher analyses across multiple models
- [ ] Run SFT + LLM-as-judge eval; record before/after numbers
- [ ] Run DPO on outcome-contrastive pairs
- [ ] Run RLVR (GRPO) with the realized-return reward
- [ ] Serve (vLLM / llama.cpp) and publish a demo

## Tech stack

* **Base Model**: `Qwen3.5-4B` — released March 2026. Hybrid Gated Delta Network +
  dense decoder architecture, **256K context window**, 4+ languages, native CoT/thinking
  capabilities. QLoRA training fits in ~4–6 GB VRAM on any 8GB+ card.
* **Data Platform**: [`aipriceaction`](https://github.com/quanhua92/aipriceaction) (Open-source financial data platform powering the pipeline with multi-timeframe **OHLCV** price action and precomputed **MA Scores** — the % distance from price to each moving average, sparing the LLM from error-prone arithmetic).
* **Training Acceleration**: `Unsloth` (Optimizes memory footprint, allowing full pipeline execution on a single 16GB RTX 4060 Ti).
* **Post-Training Framework**: Hugging Face `TRL` (Uses `SFTTrainer` for behavior shaping, `DPOTrainer` for outcome-contrastive preference, and **`AsyncGRPOTrainer`** for Verifiable Reward learning — async variant with vLLM rollout worker for faster RLVR throughput).
* **Inference & Deployment**: `vLLM` for high-throughput serving and `llama.cpp` (via GGUF export) for local, low-latency edge deployment.

## Status & intent

This is a **design document** — architecture and a build plan, not an
implementation. The interesting mechanics (RLVR-on-realized-returns, the teacher
becoming irrelevant, leakage-safe data contract) are **designed only**; the code and
runs come later, as I learn the material and work through the roadmap.

## License

MIT
