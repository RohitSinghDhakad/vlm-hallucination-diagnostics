# Object Grounding and Hallucination Diagnostics in Vision-Language Models

This repository contains the empirical benchmark suite, evaluation code, model probing scripts, and research paper exploring the mechanisms of object hallucination and grounding degradation across Vision-Language Models (VLMs).

## Repository Overview

```
├── eval_advanced_metrics.py     # CHAIR (CHAi, CHAr), BLEU, ROUGE, METEOR, CIDEr metrics
├── pope_dataloader.py           # POPE question parser and COCO annotation alignment
├── clip_probe.py                # Visual feature probe via CLIP
├── experiments/
│   ├── exp0/                    # Baseline POPE verification (BLIP affirmative bias)
│   ├── exp1/                    # Negative sampling sensitivity (Random, Popular, Adversarial)
│   ├── exp2/                    # Degradation analysis & Prior Dependency Index (PDI)
│   ├── exp3/                    # Multi-VLM generative profiling (6 models across Prompts A, B, C)
│   ├── exp4/                    # Hallucination vs. Omission seesaw & CIDEr prefix alignment
│   ├── exp5/                    # Diagnostic divergence: Probing vs. Free-form Captioning
│   └── exp6/                    # Environmental stress testing (Gaussian blur & spatial occlusion)
└── research_paper/
    ├── research_paper.tex       # IEEE conference format paper (5 pages)
    └── *.png                    # Empirical result figures and trend curves
```

## Evaluated Models
- **LLaVA-1.5-7B** (Linear Projection, Vicuna)
- **InstructBLIP-7B** (Q-Former Architecture)
- **Qwen2-VL-7B** (NaViT Dynamic Resolution)
- **Qwen2-VL-2B** (Compact Dynamic Resolution)
- **Idefics2-8B** (Perceiver Resampler, Mistral)
- **PaliGemma-3B** (Linear Projection, Gemma)

## Key Experiments Summary

- **Exp 0 — Baseline POPE Validation:** Confirms affirmative "Yes-bias" in basic probing setups.
- **Exp 1 — Sampling Protocol Sensitivity:** Evaluates how co-occurrence frequencies distort accuracy across Random (86.8%), Popular (83.0%), and Adversarial (80.8%) negative sampling splits.
- **Exp 2 — Prior Dependency Index (PDI):** Quantifies shift toward language priors under visual Gaussian degradation ($\sigma = 0 \dots 12$).
- **Exp 3 — Multi-Architecture Profiling:** Evaluates hallucination incidence across prompts: Standard (A), Descriptive (B), and Grounded (C).
- **Exp 4 — The Hallucination-Omission Seesaw:** Analyzes trade-offs between detail expansion and object hallucination, addressing prompt echo artifacts.
- **Exp 5 — Probing vs. Captioning Diagnostic Divergence:** Demonstrates near-zero correlation ($r \approx 0.00$) between POPE probing accuracy and CHAIR generative metrics.
- **Exp 6 — Compound Grounding Breakdown:** 4,200 trials under physical stress (blur and occlusion) demonstrating simultaneously collapsing recall and surging hallucinations.

## Citation / Research Paper
The full academic manuscript is available in `research_paper/research_paper.tex`.
