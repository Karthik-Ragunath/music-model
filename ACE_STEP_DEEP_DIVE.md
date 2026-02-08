# ACE-Step 1.5 — Deep Dive: Architecture, Math & Code

## Table of Contents

- [1. Intuition Behind the Paper](#1-intuition-behind-the-paper)
- [2. Architecture Diagram (ASCII)](#2-architecture-diagram-ascii)
- [3. Mathematical Intuition](#3-mathematical-intuition)
- [4. Handcrafted Toy Example](#4-handcrafted-toy-example)
- [5. Tensor Flow Through the DiT Forward Pass (ASCII)](#5-tensor-flow-through-the-dit-forward-pass-ascii)
- [6. Annotated PyTorch Snippet — Core Training & Inference](#6-annotated-pytorch-snippet--core-training--inference)
- [7. Key Takeaways](#7-key-takeaways)

---

## 1. Intuition Behind the Paper

### The Problem

Generating full songs from text is hard because music has **two fundamentally different scales**:

1. **High-level structure** (what instruments, BPM, key, verse/chorus layout, lyric timing) — changes every few seconds
2. **Low-level acoustics** (waveform samples, timbre, harmonics) — changes 48,000 times per second

A single model trying to handle both at once either produces incoherent structure (pure diffusion) or robotic sound (pure language models).

### The Solution: Two-Rate Hybrid Architecture

ACE-Step splits the problem into two models operating at different temporal resolutions:

| Component | Rate | Role |
|-----------|------|------|
| **5Hz Language Model** (Qwen3-0.6B) | 5 tokens/second | Semantic planner — decides WHAT to play |
| **25Hz Diffusion Transformer** (DiT, 24 layers) | 25 frames/second | Acoustic renderer — decides HOW it sounds |
| **VAE** (AutoencoderOobleck) | 48,000 samples/second | Waveform decoder — converts latents to audio |

### Key Novelties

1. **Chain-of-Thought (CoT) metadata reasoning** — The LM generates a `<think>...</think>` block before producing audio codes, using an FSM-constrained decoder to guarantee valid output
2. **Intrinsic reinforcement learning** — Alignment between LM and DiT is achieved using the model's own internal mechanisms (cross-attention alignment scores), not external reward models
3. **Flow matching** instead of DDPM diffusion — simpler, faster, ODE-solvable
4. **Turbo distillation** — 8 steps instead of 60+, with timestep shift scheduling

---

## 2. Architecture Diagram (ASCII)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ACE-Step 1.5 Pipeline                                │
│                                                                             │
│  USER INPUT                                                                 │
│  ┌──────────┐ ┌───────────┐ ┌──────────────┐ ┌───────────────┐            │
│  │ Caption   │ │  Lyrics   │ │ Ref. Audio   │ │ Metadata      │            │
│  │"pop song" │ │"Hello..." │ │ (optional)   │ │ BPM/Key/etc   │            │
│  └─────┬─────┘ └─────┬─────┘ └──────┬───────┘ └───────┬───────┘            │
│        │              │              │                 │                     │
│  ══════╪══════════════╪══════════════╪═════════════════╪═══════════════════  │
│  PHASE 1: 5Hz Language Model (Qwen3-0.6B, Causal LM)                       │
│  ══════╪══════════════╪══════════════╪═════════════════╪═══════════════════  │
│        │              │              │                 │                     │
│        ▼              ▼              │                 ▼                     │
│  ┌─────────────────────────┐        │  ┌─────────────────────────────┐      │
│  │  Format Prompt           │        │  │ User-provided metadata       │      │
│  │  (caption + lyrics)      │        │  │ (fills FSM constraints)      │      │
│  └────────────┬─────────────┘        │  └──────────────┬──────────────┘      │
│               │                      │                 │                     │
│               ▼                      │                 │                     │
│  ┌──────────────────────────────────┐│                 │                     │
│  │     Phase 1a: CoT Reasoning      ││  FSM-Constrained│                     │
│  │     (stop at </think>)           ││◄────Decoding────┘                     │
│  │                                  ││                                       │
│  │  <think>                         ││  Forces valid:                        │
│  │  bpm: 120                        ││  • BPM ∈ [30,300]                     │
│  │  caption: upbeat pop song...     ││  • Key ∈ {A..G}{#/b} {major/minor}   │
│  │  duration: 30                    ││  • TimeSig ∈ {2,3,4,6}               │
│  │  keyscale: G major               ││  • Duration ∈ [10,600]               │
│  │  language: en                    ││  • Language ∈ 50+ codes               │
│  │  timesignature: 4               ││                                       │
│  │  </think>                        ││                                       │
│  └────────────┬─────────────────────┘│                                       │
│               │                      │                                       │
│               ▼                      │                                       │
│  ┌──────────────────────────────────┐│                                       │
│  │     Phase 1b: Audio Codes Gen    ││  With CFG:                            │
│  │     (5Hz semantic tokens)        ││  logits = uncond + s*(cond - uncond)  │
│  │                                  ││                                       │
│  │  <|audio_code_0|>42311           ││  5 codes/second                       │
│  │  <|audio_code_1|>18729           ││  30s song → 150 codes                 │
│  │  <|audio_code_2|>55102           ││                                       │
│  │  ... (150 codes for 30s)         ││                                       │
│  └────────────┬─────────────────────┘│                                       │
│               │                      │                                       │
│  ══════════════╪══════════════════════╪══════════════════════════════════════  │
│  PHASE 2: DiT (Diffusion Transformer, 24 layers, hidden=2048)               │
│  ══════════════╪══════════════════════╪══════════════════════════════════════  │
│               │                      │                                       │
│               ▼                      ▼                                       │
│  ┌────────────────────┐  ┌─────────────────────┐                            │
│  │ Audio Detokenizer   │  │ Timbre Encoder       │                            │
│  │ 5Hz codes → 25Hz   │  │ Ref audio → emb      │                            │
│  │ latent hints        │  │ (CLS pooling)        │                            │
│  └────────┬───────────┘  └──────────┬───────────┘                            │
│           │                         │                                        │
│           ▼                         ▼                                        │
│  ┌─────────────────────────────────────────────────────────┐                │
│  │              Condition Encoder (packs all conditions)    │                │
│  │                                                         │                │
│  │  ┌──────────┐  ┌───────────────┐  ┌──────────────────┐ │                │
│  │  │Text Proj  │  │Lyric Encoder  │  │Timbre Encoder    │ │                │
│  │  │Qwen3-Emb  │  │8-layer bidir  │  │4-layer + CLS     │ │                │
│  │  │→ linear   │  │transformer    │  │pooling           │ │                │
│  │  └─────┬─────┘  └──────┬────────┘  └────────┬─────────┘ │                │
│  │        │               │                    │           │                │
│  │        │    pack_sequences(lyrics, timbre)   │           │                │
│  │        │         ┌─────┴────────────────────┘           │                │
│  │        │         ▼                                      │                │
│  │        │   pack_sequences(lyrics+timbre, text)          │                │
│  │        │         │                                      │                │
│  │        └────────►│                                      │                │
│  │                  ▼                                      │                │
│  │  encoder_hidden_states: [B, L_enc, 2048]                │                │
│  │  encoder_attention_mask: [B, L_enc]                     │                │
│  └──────────────────────────┬──────────────────────────────┘                │
│                             │                                                │
│                             │  context_latents = [src_latents ‖ chunk_mask]  │
│                             │                    [B, T, 128]  (64+64)       │
│                             ▼                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                   DiT Denoising Loop (8 steps, ODE Euler)            │    │
│  │                                                                      │    │
│  │  x_T = noise ~ N(0,I)      t_schedule = [1.0, 0.95, 0.9, ..., 0.3] │    │
│  │                                                                      │    │
│  │  for step in t_schedule:                                             │    │
│  │    ┌──────────────────────────────────────────────────────────┐      │    │
│  │    │  DiT Decoder Forward (24 transformer layers)             │      │    │
│  │    │                                                          │      │    │
│  │    │  input: [context_latents ‖ x_t] → Conv1D patchify       │      │    │
│  │    │                                                          │      │    │
│  │    │  Each layer:                                             │      │    │
│  │    │   Self-Attn(AdaLN) → Cross-Attn(enc_states) → MLP(AdaLN)│      │    │
│  │    │                                                          │      │    │
│  │    │  output: v_t (predicted velocity)                        │      │    │
│  │    └──────────────────────────────────────────────────────────┘      │    │
│  │                                                                      │    │
│  │    ODE step: x_{t-dt} = x_t - v_t * dt                              │    │
│  │    Final step: x_0 = x_t - v_t * t                                  │    │
│  │                                                                      │    │
│  └────────────────────────────┬─────────────────────────────────────────┘    │
│                               │                                              │
│                               ▼  pred_latents: [B, T, 64]                   │
│  ══════════════════════════════╪════════════════════════════════════════════  │
│  PHASE 3: VAE Decode (AutoencoderOobleck)                                    │
│  ══════════════════════════════╪════════════════════════════════════════════  │
│                               │                                              │
│                               ▼                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  VAE Decoder: [B, 64, T] → [B, 2, T*1920]                            │  │
│  │  (latent_dim=64, upsample 1920x to 48kHz stereo waveform)            │  │
│  └────────────────────────────┬───────────────────────────────────────────┘  │
│                               │                                              │
│                               ▼                                              │
│                    OUTPUT: 48kHz FLAC audio file                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Mathematical Intuition

### 3.1 Flow Matching (Core Diffusion Framework)

ACE-Step uses **Conditional Flow Matching** (CFM), not DDPM. The key idea:

**Define a probability path** from data distribution `p₀` to noise distribution `p₁ = N(0, I)`:

```
x_t = t · x₁ + (1 - t) · x₀,    t ∈ [0, 1]
```

where `x₀` is clean data and `x₁ ~ N(0, I)` is noise.

**The velocity field** (what the model learns to predict):

```
v_t = dx_t / dt = x₁ - x₀
```

**Training loss** — simple MSE on velocity prediction:

```
L = E_{t, x₀, x₁} [ ‖ f_θ(x_t, t, c) - (x₁ - x₀) ‖² ]
```

where `c` is the conditioning (text, lyrics, timbre).

**Inference** — solve the ODE backwards from noise to data:

```
x_{t - Δt} = x_t - v_t · Δt     (Euler method)
```

**Final step** — recover clean data directly:

```
x₀ = x_t - v_t · t
```

### 3.2 Why Flow Matching > DDPM

| Property | DDPM | Flow Matching |
|----------|------|---------------|
| Forward process | Complex noise schedule β_t | Simple linear interpolation |
| Prediction target | Noise ε | Velocity v = ε - x₀ |
| Inference | Stochastic (SDE), many steps | Deterministic (ODE), fewer steps |
| Path geometry | Curved (variance-preserving) | Straight lines |
| Math complexity | Higher | Lower |

### 3.3 Timestep Shift (Why 8 Steps Works)

The turbo model uses a **shifted timestep schedule** with `shift=3`:

```
t' = s · t / (1 + (s - 1) · t)
```

This concentrates sampling steps in the high-noise regime where the model needs to make the biggest decisions. For shift=3, the 8-step schedule is:

```
[1.0, 0.955, 0.9, 0.833, 0.75, 0.643, 0.5, 0.3]
```

Notice how steps are denser near t=1.0 (pure noise) and sparser near t=0 (clean data).

### 3.4 Adaptive Layer Norm (AdaLN) in DiT

Each DiT layer modulates its normalization based on the timestep embedding:

```
h' = RMSNorm(h) · (1 + γ_t) + β_t
output = h + Attn(h') · g_t
```

where `(γ_t, β_t, g_t)` are scale, shift, and gate parameters derived from the timestep embedding via a learned `scale_shift_table`. This allows the model to behave differently at different noise levels.

### 3.5 Classifier-Free Guidance (CFG) for the LM

The LM uses CFG on logits for better prompt adherence:

```
logits_cfg = logits_uncond + s · (logits_cond - logits_uncond)
```

where `s` is the guidance scale (~2.5), running conditional and unconditional prompts in parallel.

---

## 4. Handcrafted Toy Example

Let's trace through generating a **2-second song** with the prompt *"happy piano"*.

### Step 1: LM CoT Phase (5Hz = 10 codes for 2s)

```
Input prompt: "Generate audio: happy piano, [Instrumental]"

LM generates (with FSM-constrained decoding):
┌──────────────────────────────────┐
│ <think>                          │
│ bpm: 120                         │
│ caption: happy upbeat piano      │
│   melody with bright chords.     │
│ duration: 2                      │
│ keyscale: C major                │
│ language: en                     │
│ timesignature: 4                 │
│ </think>                         │
│                                  │
│ <|audio_code_0|>42311            │  ← 5Hz semantic codes
│ <|audio_code_1|>18729            │
│ <|audio_code_2|>55102            │
│ <|audio_code_3|>31847            │
│ <|audio_code_4|>09263            │
│ <|audio_code_5|>47591            │
│ <|audio_code_6|>22038            │
│ <|audio_code_7|>63104            │
│ <|audio_code_8|>15826            │
│ <|audio_code_9|>50472            │
└──────────────────────────────────┘
```

### Step 2: Audio Codes → Latent Hints

```
10 codes at 5Hz
  → FSQ dequantize: [1, 10, 2048]
  → Detokenizer (each code expands to 5 frames): [1, 50, 64]

Result: 50 frames at 25Hz (pool_window_size = 5)
Shape: [1, 50, 64]  (batch=1, time=50, latent_dim=64)
```

### Step 3: Condition Encoding

```
Text "happy upbeat piano..."
  → Qwen3-Embedding: [1, 77, 1024]
  → Linear projection: [1, 77, 2048]

Lyrics "[Instrumental]"
  → embed_tokens: [1, 512, 1024]
  → LyricEncoder (8-layer bidir transformer): [1, 512, 2048]

Timbre (silence — no reference audio)
  → TimbreEncoder: [1, 1, 2048]

Pack all together:
  pack(lyrics, timbre) → [1, 513, 2048]
  pack(result, text)   → [1, 590, 2048]

encoder_hidden_states: [1, 590, 2048]

Context: [src_latents ‖ chunk_mask]
  = [1, 50, 64] ‖ [1, 50, 64]
  = [1, 50, 128]
```

### Step 4: DiT Denoising (8 Steps)

```
x_8 = randn(1, 50, 64)     ← pure noise

Step 1: t=1.000  → DiT predicts v_8 → x_7 = x_8 - v_8 × 0.045
Step 2: t=0.955  → DiT predicts v_7 → x_6 = x_7 - v_7 × 0.055
Step 3: t=0.900  → DiT predicts v_6 → x_5 = x_6 - v_6 × 0.067
Step 4: t=0.833  → DiT predicts v_5 → x_4 = x_5 - v_5 × 0.083
Step 5: t=0.750  → DiT predicts v_4 → x_3 = x_4 - v_4 × 0.107
Step 6: t=0.643  → DiT predicts v_3 → x_2 = x_3 - v_3 × 0.143
Step 7: t=0.500  → DiT predicts v_2 → x_1 = x_2 - v_2 × 0.200
Step 8: t=0.300  → DiT predicts v_1 → x_0 = x_1 - v_1 × 0.300  ← FINAL

Result: x_0 shape [1, 50, 64] ← clean audio latent
```

### Step 5: VAE Decode

```
Transpose: [1, 50, 64] → [1, 64, 50]
VAE decode: [1, 64, 50] → [1, 2, 96000]

  2 channels (stereo)
  50 latent frames × 1920 upsample = 96,000 samples
  96,000 / 48,000 Hz = 2 seconds

Save as FLAC at 48kHz ✅
```

---

## 5. Tensor Flow Through the DiT Forward Pass (ASCII)

```
INPUT TENSORS
═══════════════════════════════════════════════════════════════════

  hidden_states (x_t):        [B, T, 64]        e.g. [1, 750, 64]
  context_latents:             [B, T, 128]       e.g. [1, 750, 128]   (src_lat ‖ chunk_mask)
  encoder_hidden_states:       [B, L_enc, 2048]  e.g. [1, 590, 2048]  (packed conditions)
  timestep t:                  [B]               e.g. [1]              scalar per sample
  timestep_r:                  [B]               e.g. [1]              (same as t for turbo)


STEP 1: Concatenate context + hidden states
═══════════════════════════════════════════════════════════════════

  cat([context_latents, hidden_states], dim=-1)
  [B, T, 128] ‖ [B, T, 64] → [B, T, 192]
                                     ↑ in_channels = 192


STEP 2: Patchify via Conv1D (patch_size=2)
═══════════════════════════════════════════════════════════════════

  [B, T, 192] → transpose → [B, 192, T]
  Conv1D(192 → 2048, kernel=2, stride=2)
  [B, 192, T] → [B, 2048, T//2] → transpose → [B, T//2, 2048]

  e.g. [1, 750, 192] → [1, 375, 2048]


STEP 3: Timestep Embedding
═══════════════════════════════════════════════════════════════════

  t: [B] → sinusoidal(256-dim) → MLP → temb: [B, 2048]
  r: [B] → sinusoidal(256-dim) → MLP → temb_r: [B, 2048]
  temb = temb_t + temb_r: [B, 2048]

  time_proj = SiLU(temb) → Linear → [B, 6, 2048]
    → used for AdaLN in each DiT layer (shift, scale, gate × 2)


STEP 4: Project encoder hidden states
═══════════════════════════════════════════════════════════════════

  encoder_hidden_states: [B, L_enc, 2048] → Linear(2048 → 2048)
  encoder_attention_mask: [B, L_enc]


STEP 5: 24 DiT Layers (alternating sliding/full attention)
═══════════════════════════════════════════════════════════════════

  For each layer i ∈ [0..23]:

  ┌─────────────────────────────────────────────────────────────┐
  │  Extract AdaLN params from scale_shift_table + time_proj     │
  │  [1, 6, 2048] → chunk into 6 → (shift, scale, gate) × 2    │
  │                                                              │
  │  ── Self-Attention (with AdaLN) ──────────────────────────  │
  │  h_norm = RMSNorm(h) * (1 + scale_sa) + shift_sa            │
  │  Q = q_norm(W_q · h_norm)     [B, T//2, 16, 128]            │
  │  K = k_norm(W_k · h_norm)     [B, T//2, 8, 128]  (GQA)     │
  │  V = W_v · h_norm              [B, T//2, 8, 128]            │
  │  + RoPE on Q, K                                              │
  │  + sliding_window=128 on odd layers                          │
  │  attn_out = softmax(QK^T / √d) · V → W_o                   │
  │  h = h + attn_out * gate_sa                       (gated)   │
  │                                                              │
  │  ── Cross-Attention (to encoder states) ─────────────────── │
  │  h_norm = RMSNorm(h)                                         │
  │  Q = q_norm(W_q · h_norm)     [B, T//2, 16, 128]            │
  │  K = k_norm(W_k · enc_states) [B, L_enc, 8, 128]            │
  │  V = W_v · enc_states          [B, L_enc, 8, 128]           │
  │  (KV cached after first step for efficiency)                 │
  │  attn_out = softmax(QK^T / √d) · V → W_o                   │
  │  h = h + attn_out                              (no gate)     │
  │                                                              │
  │  ── MLP (with AdaLN) ───────────────────────────────────── │
  │  h_norm = RMSNorm(h) * (1 + scale_ff) + shift_ff            │
  │  h_ff = SiLU(W_gate · h_norm) ⊙ (W_up · h_norm)            │
  │  h_ff = W_down · h_ff                                        │
  │  h = h + h_ff * gate_ff                           (gated)   │
  └─────────────────────────────────────────────────────────────┘

  Output after 24 layers: [B, T//2, 2048]


STEP 6: Adaptive Output Norm + De-patchify
═══════════════════════════════════════════════════════════════════

  shift, scale = (scale_shift_table + temb).chunk(2)
  h = RMSNorm(h) * (1 + scale) + shift

  ConvTranspose1d(2048 → 64, kernel=2, stride=2)
  [B, T//2, 2048] → transpose → [B, 2048, T//2]
                  → deconv   → [B, 64, T]
                  → transpose → [B, T, 64]

  Crop to original_seq_len: [B, T, 64]


OUTPUT: v_t (predicted velocity field): [B, T, 64]
═══════════════════════════════════════════════════════════════════
```

---

## 6. Annotated PyTorch Snippet — Core Training & Inference

```python
"""
ACE-Step 1.5 — Core Flow Matching Training & Inference (Simplified)

This snippet illustrates the two key novelties:
1. Flow Matching (not DDPM) for the DiT diffusion model
2. The LM → DiT two-rate pipeline with audio code bridging

Based on:
  - acestep/training/trainer.py (training_step)
  - checkpoints/acestep-v15-turbo/modeling_acestep_v15_turbo.py (generate_audio)
"""

import torch
import torch.nn.functional as F
import math


# ============================================================
# PART 1: FLOW MATCHING TRAINING
# ============================================================
# The key insight: instead of learning to predict noise (DDPM),
# we learn to predict the VELOCITY FIELD (flow) that transports
# noise to data along a straight-line path.

def flow_matching_training_step(model_decoder, batch, device):
    """
    One training step for the DiT using flow matching.

    The forward process is a simple linear interpolation:
        x_t = t * noise + (1 - t) * data

    The model learns to predict:
        v = noise - data  (the velocity/flow direction)

    This is simpler than DDPM's noise prediction because:
    - No complex noise schedule (beta_t) needed
    - Straight-line paths = fewer steps needed at inference
    - ODE-solvable = deterministic generation possible
    """

    # ---- Unpack batch ----
    x0 = batch["target_latents"].to(device)           # [B, T, 64] — clean audio latents
    attention_mask = batch["attention_mask"].to(device)
    encoder_hidden_states = batch["encoder_hidden_states"].to(device)  # [B, L_enc, 2048]
    encoder_attention_mask = batch["encoder_attention_mask"].to(device)
    context_latents = batch["context_latents"].to(device)              # [B, T, 128]

    bsz = x0.shape[0]

    # ---- Step 1: Sample pure Gaussian noise ----
    x1 = torch.randn_like(x0)  # [B, T, 64] — noise (target distribution)

    # ---- Step 2: Sample timesteps from turbo schedule ----
    # Turbo model uses 8 discrete timesteps with shift=3.0
    # This concentrates steps in the high-noise region where
    # the model needs to make the most critical decisions.
    TURBO_TIMESTEPS = [1.0, 0.955, 0.9, 0.833, 0.75, 0.643, 0.5, 0.3]
    indices = torch.randint(0, len(TURBO_TIMESTEPS), (bsz,), device=device)
    t = torch.tensor(TURBO_TIMESTEPS, device=device, dtype=torch.bfloat16)[indices]

    # ---- Step 3: Create noisy samples via linear interpolation ----
    # This is the "forward process" in flow matching.
    # At t=0: x_t = x0 (clean data)
    # At t=1: x_t = x1 (pure noise)
    # At t=0.5: x_t = 50% noise + 50% data
    t_ = t.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1] for broadcasting
    xt = t_ * x1 + (1.0 - t_) * x0     # [B, T, 64] — interpolated sample

    # ---- Step 4: Predict velocity through DiT ----
    # The DiT takes:
    #   - xt: the noisy audio latents
    #   - t: which point on the noise-to-data path we're at
    #   - context_latents: [src_latents || chunk_mask]
    #   - encoder_hidden_states: packed (lyrics + timbre + text) embeddings
    decoder_outputs = model_decoder(
        hidden_states=xt,
        timestep=t,
        timestep_r=t,
        attention_mask=attention_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        context_latents=context_latents,
    )
    v_predicted = decoder_outputs[0]  # [B, T, 64] — predicted velocity

    # ---- Step 5: Compute flow matching loss ----
    # The ground-truth velocity is simply: v = noise - data
    # MSE loss: how well does the model predict this velocity?
    v_target = x1 - x0  # [B, T, 64] — true velocity (flow field)
    loss = F.mse_loss(v_predicted, v_target)

    return loss


# ============================================================
# PART 2: FLOW MATCHING INFERENCE (ODE EULER SOLVER)
# ============================================================
# At inference time, we start from pure noise and follow the
# learned velocity field backwards to arrive at clean data.

@torch.no_grad()
def flow_matching_inference(model_decoder, encoder_hidden_states,
                            encoder_attention_mask, context_latents,
                            attention_mask, seed=42):
    """
    Generate audio latents by solving the flow ODE backwards.

    Starting from x_T ~ N(0,I), we iteratively apply:
        x_{t-dt} = x_t - v_t * dt     (Euler method)

    The turbo model does this in just 8 steps (vs 60-100 for base).
    """
    device = context_latents.device
    dtype = context_latents.dtype
    bsz = context_latents.shape[0]
    T = context_latents.shape[1]
    latent_dim = context_latents.shape[-1] // 2  # 128/2 = 64

    # ---- Step 1: Sample initial noise ----
    generator = torch.Generator(device=device).manual_seed(seed)
    xt = torch.randn(bsz, T, latent_dim, generator=generator,
                     device=device, dtype=dtype)

    # ---- Step 2: Define timestep schedule (shift=3, 8 steps) ----
    # NOT evenly spaced! Shifted to spend more time in high-noise region.
    #
    # Formula: t_shifted = shift * t / (1 + (shift - 1) * t)
    # With shift=3 and 8 uniform steps:
    t_schedule = [1.0, 0.955, 0.9, 0.833, 0.75, 0.643, 0.5, 0.3]

    # ---- Step 3: Euler ODE solver ----
    for step_idx in range(len(t_schedule)):
        t_curr = t_schedule[step_idx]
        t_tensor = t_curr * torch.ones(bsz, device=device, dtype=dtype)

        # Forward pass through DiT — predict velocity at current timestep
        decoder_out = model_decoder(
            hidden_states=xt,
            timestep=t_tensor,
            timestep_r=t_tensor,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            context_latents=context_latents,
            use_cache=True,  # Cache cross-attention KV for speed
        )
        vt = decoder_out[0]  # [B, T, 64] — predicted velocity

        # Final step: directly recover x_0
        if step_idx == len(t_schedule) - 1:
            # x_0 = x_t - v_t * t  (jump directly to data)
            xt = xt - vt * t_curr
            break

        # Euler step: move along negative velocity direction
        # x_{t-dt} = x_t - v_t * dt
        t_next = t_schedule[step_idx + 1]
        dt = t_curr - t_next  # positive (moving from noise toward data)
        dt_tensor = dt * torch.ones(bsz, 1, 1, device=device, dtype=dtype)
        xt = xt - vt * dt_tensor

    # xt is now the predicted clean latent x_0: [B, T, 64]
    return xt


# ============================================================
# PART 3: THE TWO-RATE BRIDGE (5Hz LM → 25Hz DiT)
# ============================================================
# This is the core architectural novelty: bridging the semantic
# gap between coarse LM planning and fine acoustic rendering.

def audio_code_to_latent_hints(audio_tokenizer, audio_detokenizer,
                               audio_code_indices, pool_window_size=5):
    """
    Convert LM-generated 5Hz audio codes into 25Hz latent hints
    that condition the DiT's diffusion process.

    The LM generates 5 codes per second (coarse semantic sketch).
    The DiT needs 25 frames per second (fine acoustic detail).
    This function bridges the two rates.

    Pipeline:
        5Hz codes → FSQ dequantize → Detokenizer (expand 5x) → 25Hz hints

    Args:
        audio_code_indices: [B, T_5hz] integers in [0, 64000)
            Example: 150 codes for a 30-second song

    Returns:
        lm_hints_25Hz: [B, T_25hz, 64] continuous latent hints
            Example: [B, 750, 64] for a 30-second song
    """
    # Step 1: Dequantize codes using Finite Scalar Quantization (FSQ)
    # FSQ uses [8,8,8,5,5,5] levels with 1 quantizer
    # Each integer code maps to a point in a 2048-dim continuous space
    lm_hints_5Hz = audio_tokenizer.quantizer.get_output_from_indices(
        audio_code_indices
    )
    # Shape: [B, T_5hz, 2048]

    # Step 2: Detokenize — expand each 5Hz token into 5 frames at 25Hz
    # Uses learnable special tokens + 2-layer encoder to "upscale"
    lm_hints_25Hz = audio_detokenizer(lm_hints_5Hz)
    # Shape: [B, T_5hz * 5, 64] = [B, T_25hz, 64]

    return lm_hints_25Hz


# ============================================================
# PART 4: END-TO-END GENERATION (Putting it all together)
# ============================================================

def generate_song_e2e(lm, dit_model, vae, text_encoder, text_tokenizer,
                      caption="happy piano", lyrics="[Instrumental]",
                      duration=30, seed=42):
    """
    Full end-to-end song generation showing the LM → DiT → VAE pipeline.

    Returns: waveform tensor [1, 2, num_samples] at 48kHz
    """

    # ═══ PHASE 1: LM Chain-of-Thought + Audio Codes ═══
    # The LM generates metadata AND semantic audio codes
    # Phase 1a: CoT reasoning (generates BPM, key, duration, etc.)
    # Phase 1b: 5Hz audio codes (5 × duration tokens)

    prompt = f"Generate audio: {caption}, {lyrics}"
    lm_output = lm.generate(prompt)
    # Produces structured <think>...</think> + audio codes

    metadata = parse_metadata(lm_output)
    # e.g. {"bpm": 120, "keyscale": "C major", ...}

    audio_codes = parse_audio_codes(lm_output)
    # e.g. [42311, 18729, ..., 55102] (150 codes for 30s)

    # ═══ BRIDGE: 5Hz codes → 25Hz latent hints ═══
    lm_hints_25Hz = audio_code_to_latent_hints(
        dit_model.tokenizer, dit_model.detokenizer,
        audio_codes, pool_window_size=5
    )  # [1, 750, 64] for 30s

    # ═══ PHASE 2: Condition Encoding ═══
    # Encode text caption → Qwen3-Embedding → projection
    text_emb = text_encoder(text_tokenizer(caption))  # [1, 77, 1024]
    text_proj = dit_model.encoder.text_projector(text_emb)  # [1, 77, 2048]

    # Encode lyrics → embedding → 8-layer bidirectional transformer
    lyric_emb = text_encoder.embed_tokens(
        text_tokenizer(lyrics)
    )  # [1, 512, 1024]
    lyric_enc = dit_model.encoder.lyric_encoder(lyric_emb)  # [1, 512, 2048]

    # Pack all conditions
    enc_states, enc_mask = pack_sequences(
        lyric_enc, timbre_emb, text_proj
    )  # [1, ~590, 2048]

    # Context: LM hints (as "source latents") + chunk mask
    context = torch.cat(
        [lm_hints_25Hz, chunk_mask], dim=-1
    )  # [1, 750, 128]

    # ═══ PHASE 3: DiT Diffusion (8-step flow matching) ═══
    pred_latents = flow_matching_inference(
        dit_model.decoder, enc_states, enc_mask, context,
        attention_mask, seed=seed
    )  # [1, 750, 64]

    # ═══ PHASE 4: VAE Decode to Waveform ═══
    pred_latents_t = pred_latents.transpose(1, 2)  # [1, 64, 750]
    waveform = vae.decode(pred_latents_t).sample
    # [1, 2, 750*1920] = [1, 2, 1440000]
    # 1,440,000 samples / 48,000 Hz = 30 seconds of stereo audio

    return waveform  # Save as FLAC at 48kHz
```

---

## 7. Key Takeaways

| Aspect | Detail |
|--------|--------|
| **Two rates** | LM at 5Hz (planning), DiT at 25Hz (rendering), VAE at 48kHz (waveform) |
| **Flow matching** | Linear interpolation `x_t = t·ε + (1-t)·x₀`, predict `v = ε - x₀` |
| **Turbo** | Distilled to 8 steps with shift=3.0 timestep schedule |
| **DiT architecture** | 24 layers × {Self-Attn(AdaLN) + Cross-Attn + MLP(AdaLN)}, hidden=2048, GQA (16 heads, 8 KV heads) |
| **Condition encoding** | Text (Qwen3-Embedding) + Lyrics (8-layer bidir) + Timbre (4-layer CLS-pool), all packed into one sequence |
| **FSM-constrained decoding** | LM metadata generation uses a finite state machine to guarantee valid BPM/key/duration/etc. |
| **Audio tokenization** | FSQ with levels [8,8,8,5,5,5] → codebook size 64,000 |
| **VAE** | AutoencoderOobleck, latent_dim=64, upsample factor 1920 (25Hz → 48kHz) |

---

## Source Files Reference

All content derived from direct reading of the codebase:

| File | What it contains |
|------|-----------------|
| `checkpoints/acestep-v15-turbo/modeling_acestep_v15_turbo.py` | Full DiT + encoder + tokenizer/detokenizer + flow matching + ODE solver |
| `checkpoints/acestep-v15-turbo/configuration_acestep_v15.py` | All model hyperparameters |
| `acestep/training/trainer.py` | Flow matching training loop with discrete turbo timesteps |
| `acestep/llm_inference.py` | Two-phase CoT → codes generation with CFG |
| `acestep/constrained_logits_processor.py` | Finite state machine for valid metadata |
| `acestep/inference.py` | End-to-end LM → DiT → VAE orchestration |
| `acestep/handler.py` | VAE encode/decode, batch preparation, tiled decode |
| `acestep/training/dataset_builder_modules/preprocess_vae.py` | VAE encoding for training |
| `acestep/training/dataset_builder_modules/preprocess_lyrics.py` | Lyrics encoding for training |
