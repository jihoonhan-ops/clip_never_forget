# CLIP Never Forgets 🧊

**Frozen VLM features are a surprisingly strong class-incremental baseline — and forgetting comes from the classifier head, not the representation.**

A minimal, reproducible study of class-incremental learning (CIL) on CIFAR-100 with a **frozen CLIP ViT-B/32**. Three classifiers share the exact same backbone, so any difference in forgetting can be attributed to the classifier alone.

> **TL;DR** — The frozen representation never forgets. What collapses under sequential training is the linear head, which develops a *recency bias* toward the most recently seen classes. A non-parametric prototype classifier over the same features forgets almost nothing.

---

## Motivation

Catastrophic forgetting — the tendency of a network to lose earlier knowledge when trained on new classes — is the central obstacle in class-incremental learning. Recent pre-training-based CIL methods (e.g. *PriViLege*, CVPR 2024) build on a simple observation: a frozen vision-language model already carries enough knowledge that incremental learning becomes far easier.

This repo verifies the **most basic version of that premise directly**: with the CLIP backbone frozen, *where does forgetting actually come from* — the representation, or the classifier on top of it?

## Setup

- **Dataset:** CIFAR-100, split into **10 tasks × 10 classes** (class order shuffled with a fixed seed).
- **Backbone:** CLIP ViT-B/32, **frozen** throughout. Image embeddings are extracted once and cached; every experiment runs on those cached features.
- **Few-shot budget:** 16 images per class, shared by the prototype and linear-probe methods.
- **Evaluation:** class-incremental — each sample is classified among **all classes seen so far**, not just within its own task.

| Method | Classifier | Trained? | Updates past classes? |
|---|---|---|---|
| **Zero-shot CLIP** | text embeddings of `"a photo of a {class}"` | no | — |
| **Prototype (NCM)** | mean of few-shot image embeddings per class | no | never |
| **Seq. Linear Probe** | `nn.Linear(512, 100)` fine-tuned task-by-task (no replay) | yes | overwrites |

## Results

![Main results](assets/results.png)

**1. The sequential linear probe collapses.** Final average accuracy drops to **0.117**, and Task-1 accuracy hits **0** after just three tasks (center panel). This is not a representation failure — see the diagnostic below.

**2. The prototype classifier barely forgets.** With only 16 shots per class it keeps a final average accuracy of **0.611** and a forgetting measure of **0.096**. Its gentle decline tracks the zero-shot curve, reflecting the growing number of candidate classes rather than actual forgetting.

**3. Zero-shot text prompts stay strong.** A single prompt per class remains a competitive baseline end-to-end (final **0.642**).

| Method | Final avg. acc | Forgetting |
|---|---|---|
| Zero-shot CLIP | 0.642 | 0.118 |
| Prototype (NCM) | 0.611 | 0.096 |
| Seq. Linear Probe | 0.117 | 0.844 |

*(The ~0.1 "forgetting" of zero-shot and prototype is an artifact of the growing candidate set, not memory loss — neither method ever modifies past-class representations.)*

### Diagnosis: forgetting is a classifier-head recency bias

Evaluating the **final** linear probe on **Task-1** samples, changing only the candidate set:

| Evaluation | Candidates | Task-1 accuracy |
|---|---|---|
| 10-way (within task) | Task-1 classes only | **0.848** |
| 100-way (class-incremental) | all seen classes | **0.000** |

Same head, same samples — yet 0.848 vs 0.000. The ability to *tell Task-1 classes apart* is fully intact; the head only loses when old and new classes **compete** as candidates. The cross-entropy updates on each new task inflate recent-class logits and suppress old ones, so predictions are funneled toward the newest classes. That is the entire failure mode — a **task-recency bias** in the head, not degraded features. (This is a known phenomenon in CIL; here it is isolated in a minimal frozen-backbone setting.)

## Ablation: how many shots does a prototype need?

![Shots ablation](assets/ablation_shots.png)

Prototype accuracy vs. shots per class, averaged over **3 sampling seeds** (±std shown).

| Shots | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| Final avg. acc | 0.286 | 0.397 | 0.499 | 0.562 | 0.605 |

Accuracy scales roughly log-linearly with shots, but even a 16-shot prototype (0.605) does **not** beat zero-shot text prompts (0.642) — in this setting a single text embedding is worth more than 16 images per class.

## Text–image ensemble

If text prompts and image prototypes each work well alone, are they complementary? Classify with a normalized convex combination `α · text + (1−α) · prototype`:

| α | 0.0 | 0.25 | **0.5** | 0.75 | 1.0 |
|---|---|---|---|---|---|
| Final avg. acc | 0.611 | 0.662 | **0.692** | 0.686 | 0.642 |

**α = 0.5 beats both endpoints** (0.692 vs 0.642 / 0.611) → the two modalities carry complementary information. (α = 0 and α = 1 exactly recover the prototype and zero-shot numbers, sanity-checking the implementation.)

## Reproduce

The whole study runs end-to-end on a free Google Colab T4 in roughly 15 minutes.

1. Open `clip_never_forgets.ipynb` in Colab (Runtime → T4 GPU).
2. Runtime → Run all.

The heavy step (CLIP embedding extraction) runs once and is cached; every downstream experiment then runs in seconds. All randomness is seeded, so the numbers above reproduce exactly.

## Key takeaways

1. A frozen CLIP backbone does not forget — its representation is unchanged by construction, and class separability survives the full task sequence.
2. Sequential fine-tuning of a linear head fails through **recency bias**, not representation loss — demonstrated by the 0.848 vs 0.000 split.
3. A non-parametric **prototype (NCM)** classifier over frozen features is a strong, forgetting-free CIL baseline from just a few shots.
4. **Text and image** class representations are complementary; a simple ensemble beats either alone.

## Limitations

- Single dataset (CIFAR-100), single backbone (ViT-B/32).
- Zero-shot uses one basic prompt template — no prompt engineering or prompt ensembling.
- The prototype classifier is a plain class mean (isotropic; ignores feature covariance).

## Related

- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), 2021.
- Park et al., *Pre-trained Vision and Language Transformers Are Few-Shot Incremental Learners* (PriViLege), CVPR 2024.
