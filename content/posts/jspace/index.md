---
title: "Does Chain of Thought Replace a Model's Internal Workspace? Ablating J-Space in Qwen3-4B"
date: 2026-10-15
author: ["Dev Desai"]
draft: false
categories: ["projects"]
tags: ["AI safety", "interpretability", "LLMs", "chain of thought", "in-context learning"]
showtoc: true
math: true
---

*This post covers my Homework 0 for Harvard's CS 2881 (AI Safety), plus a follow-up experiment on in-context learning that I ran about a month later.*

In July 2026, Gurnee, Sofroniew et al. at Anthropic published [*Verbalizable Representations Form a Global Workspace in Language Models*](https://arxiv.org/abs/2607.15495). One of its claims is that a language model has a small set of internal representations, which they call the **J-space**, that it uses for multi-step reasoning. If you ablate that space, the model can still talk fluently and do a lot of automatic inference, but internal multi-step reasoning falls apart.

The finding that caught my attention: on GSM8K, problems solved **with explicit chain of thought (CoT)** were much more robust to J-space ablation than the same problems answered directly. The idea is that writing steps down on the page and holding them in the internal workspace are partially interchangeable scratchpads.

Our homework asked us to dig into this trade-off: does it keep holding as problems get harder?

---

## What is J-Space?

"J" stands for **Jacobian**. For a layer $\ell$, the paper averages the Jacobian of the final-layer residual stream with respect to the layer-$\ell$ residual stream, over positions and prompts:

$$
J_\ell = \mathbb{E}_{t,\, t' \ge t,\, \text{prompt}} \left[ \frac{\partial h_{\text{final}, t'}}{\partial h_{\ell, t}} \right]
$$

This is one $d_{\text{model}} \times d_{\text{model}}$ matrix per layer. Roughly, it captures a representation's standing tendency to eventually produce a given output. You can use it as a lens, much like the logit lens:

$$
\text{lens}(h_\ell) = \text{softmax}\left( W_U \cdot \text{norm}(J_\ell \, h_\ell) \right)
$$

Each row of $W_U J_\ell$ is a **J-lens vector** $v_{\ell, w}$, one per token $w$. A token's score at a position is $v_{\ell,w} \cdot h_{\ell,t}$. The **J-space** of an activation is the small, sparse set of J-lens vectors that are currently most active.

**Ablating** it works like this. At every position in a band of layers:

1. Take the top $k = 10$ J-lens vectors by score.
2. Skip any token that is already in the clean model's top-10 next-token predictions. This keeps the ablation from simply deleting the answer the model is about to say.
3. Project those directions out: $h \leftarrow h - (h \cdot \hat v)\hat v$.

---

## The Hypothesis

My hypothesis was that the CoT substitution **weakens** as problems get harder (GSM8K → MATH-500 → AIME). Writing a step down should help when the model just needs to *remember* earlier results. It should help less when coming up with a non-obvious next step still needs the internal workspace.

To test this, I defined a **robustness gap** as the difference in *relative* accuracy drop between the two modes:

$$
RG(d) = \frac{\text{CoT}_{\text{ablated}}(d)}{\text{CoT}_{\text{clean}}(d)} - \frac{\text{noCoT}_{\text{ablated}}(d)}{\text{noCoT}_{\text{clean}}(d)}
$$

I used retention ratios rather than raw accuracy differences on purpose. CoT helps more on harder problems anyway, so the raw CoT minus no-CoT gap would widen with difficulty regardless of anything J-space related.

I pre-registered this (committed before running any experiments): if the hypothesis holds, $RG$ should **shrink** with difficulty. A stable or widening gap counts as evidence against it.

---

## Setup

**Model:** Qwen3-4B (36 layers, $d_{\text{model}} = 2560$), bf16, greedy decoding, on a rented RTX 4090.

**The Jacobian:** built over 25 wikitext-2 prompts of 128 tokens each. Each prompt's first 4 positions are skipped because their norms are anomalous. This follows Neel Nanda's lighter-weight replication rather than the paper's 1,000-prompt corpus.

**Finding the workspace band:** the paper's analysis was on Claude, so I had to locate the band of layers where the J-space lives in Qwen myself. An "ignition" test, which needs no J-lens, put the onset at **layer 13 (36% of depth)**, close to the paper's ~38% for Claude. Lens hit rate stays flat through the middle layers and then climbs in the late "motor" layers, where the model is just doing next-token prediction. I set the band at L13–27.

{{< figure src="./images/band_evidence.png" width="750px" align="center" caption="Locating the workspace band in Qwen3-4B. Ignition onset at layer 13 (~36% depth); lens hit rate rises sharply after ~65% depth, marking the late 'motor' layers.">}}

**The grid:** 3 difficulties × 3 ablation conditions (clean / J-space / random) × 2 reasoning modes (CoT / no-CoT), all on the same problems so comparisons are paired. The main grid ablates only the first third of the band (L13–17).

| Difficulty | Source | N |
| --- | --- | ---: |
| GSM8K | openai/gsm8k, test | 40 |
| MATH-500 | HuggingFaceH4/MATH-500, test | 30 |
| AIME | aime 2024 + aime 2025 | 20 |

**Controls:**
- **A norm-matched random-direction ablation.** It removes the same norm as the J-space ablation, but along random directions. Matching only the *number* of directions would confuse "removed more" with "removed the right thing."
- **A positive control.** One-hop vs. two-hop factual recall (e.g. currency → country → capital), with no math. This checks that the ablation does anything interesting before trusting it on harder tasks.
- **Coherence and removed fraction per cell.** Ablation can wreck the output entirely before producing anything interpretable, so I measured this rather than assuming it.

**Prompting gotchas:** the no-CoT arm is prefilled with `The final answer is $\boxed{` to force an immediate answer. Without the prefill, Qwen3-4B just wrote a full solution anyway and got truncated. A uniform 512-token cap also truncated 2.5% of GSM8K answers but 76% of MATH-500 ones, so I switched to per-difficulty caps (512 / 1024 / 1024).

---

## Results

### Positive control

Ablating L13–17 drops two-hop recall from 0.775 to 0.600. The random control removes *more* norm but does less two-hop damage (0.700). The J-space vs. random difference has a 95% CI of [0.013, 0.188], so the ablation is **specific**.

It isn't really **selective**, though: one-hop recall also drops, from 0.938 to 0.787. Wider bands damage both and destroy coherence, which is why the main grid stays at L13–17.

### Main grid

| Difficulty | Reasoning | Clean | J-space | Random |
| --- | --- | ---: | ---: | ---: |
| GSM8K | no-CoT | 0.35 | 0.23 | 0.23 |
| GSM8K | CoT | 0.88 | 0.85 | 0.88 |
| MATH-500 | no-CoT | 0.20 | 0.07 | 0.13 |
| MATH-500 | CoT | 0.50 | 0.60* | 0.47 |

<small>*Ablation shortened the generations, so they were truncated less often. Restricted to problems that both arms finished within budget, the two are equal.</small>

CoT accuracy barely moves under ablation, while no-CoT drops more and more as difficulty rises. The robustness gap:

| Difficulty | CoT retention | no-CoT retention | Gap | 95% CI |
| --- | ---: | ---: | ---: | --- |
| GSM8K | 0.94 | 0.69 | 0.25 | [−0.12, 0.56] |
| MATH-500 | 1.00 | 0.25 | 0.75 | [0.06, 1.30] |

{{< figure src="./images/robustness.png" width="800px" align="center" caption="Left: accuracy in every cell. Right: the robustness gap, which my hypothesis predicted would shrink to the right. It grows instead.">}}

**AIME** gave no usable data. Qwen3-4B basically can't solve it: no-CoT got 0/60. Raising the token cap eightfold (1024 → 8192) on the 5 problems that had been truncated dropped truncation from 5/5 to 1/5, but accuracy stayed at 0/5. The limit is ability, not budget.

### So, the hypothesis was wrong

The gap **widens**, from 0.25 to 0.75. CoT retention stays near 1.0 at both difficulties, while no-CoT retention falls from 0.69 to 0.25. Written CoT protected *more* on the harder dataset, not less. Direct answering has further to fall as problems get harder, and CoT has little left to lose.

There are some big caveats:

1. **It isn't shown to be J-space specific.** On GSM8K, the random control produces an equal or larger gap, and every math cell's random-vs-J-space interval spans zero.
2. **Only the lightest ablation was tested.** Widening the band on GSM8K drops CoT retention from 0.971 (L13–17) to 0.857 (L13–22) to 0.371 (L13–27). At the widest band, coherence itself breaks down. "CoT is robust" only held at light severity.
3. **There's no data at the hardest difficulty.** AIME sits at a capability ceiling for a 4B model.

---

## Follow-up: Can In-Context Learning Route Around the Ablation?

At the end of the homework I had a nagging thought. The paper notes that fluent, pattern-continuation-style generation keeps going under ablation even when the relevant concept is in the J-space. "Autocomplete" seems to route around the workspace.

A math textbook is full of complex reasoning that an LLM could just *autofill*. So what if, instead of instructing the model to solve a problem, you present the problem as **a document to be continued**? Does the ablated model recover the accuracy it loses when asked directly?

### Pre-registered hypotheses

- **H-A (autocomplete bypass):** demonstrations raise retention under J-space ablation, and the random control doesn't show the same rise.
- **H-B (contextual inference needs the workspace):** using demonstrations is itself the kind of inference the ablation removes, so the in-context gain shrinks.
- **H-0 (framing is irrelevant):** retention doesn't change.

My prior was H-0 or H-B, not H-A.

### Setup

Same model, same ablation (top-10, clean-top-10 exclusion) at every position, *including the demonstrations*. I ran it at a light band (L13–17) and a medium band (L13–22), with the norm-matched random control in every ablated cell. Sample sizes are larger this time: N=200 for GSM8K direct answers and N=100 for MATH-500.

The contexts I tried, in order of how much of the *procedure* they contain:

| Context | What the model sees before the test problem |
| --- | --- |
| Instruction (chat) | The original homework's chat-template instruction |
| Plain text | `Problem: … / Solution:`, with no instruction or examples |
| 5 unrelated examples | 5 worked GSM8K training problems |
| Textbook + examples | A 272-word textbook-style chapter on word problems, then the 5 examples |
| Long textbook + examples | A 1,285-word chapter, then the examples |
| **Same problem, new numbers** | Up to 3 copies of the *test problem* with different numbers, solved |

For the last one, I generated variants from GSM8K's calculator annotations and checked every equation in each rewritten solution. 112 of the 200 problems had safe variants.

### Unrelated examples and textbooks do nothing

Demonstrations of *other* problems changed neither clean accuracy (0.26–0.27 throughout) nor retention under ablation. Adding a textbook chapter, short or long, also changed nothing, even though the long-textbook prompts have about 20× more ablated positions than the instruction prompt. The confidence intervals were tight enough to call this a real null, not just an underpowered result.

### The same problem with new numbers undoes the damage

| Band | Context | Clean | J-space | Random | Retention (J-space) |
| --- | --- | ---: | ---: | ---: | --- |
| L13–17 | Instruction | 0.321 | 0.170 | 0.223 | 0.53 |
| L13–17 | 5 unrelated examples | 0.330 | 0.196 | 0.205 | 0.60 |
| L13–17 | **Same problem** | 0.375 | **0.384** | 0.330 | **1.02** [0.84, 1.24] |
| L13–22 | Instruction | 0.321 | 0.098 | 0.232 | 0.31 |
| L13–22 | 5 unrelated examples | 0.330 | 0.125 | 0.214 | 0.38 |
| L13–22 | **Same problem** | 0.375 | **0.268** | 0.259 | **0.71** [0.53, 0.92] |

At the light band, the same-problem context **erases the J-space damage entirely**: the ablated model is as accurate as the clean one. At the medium band, it removes the J-space-*specific* part of the damage and leaves only what the random control also causes (0.71 vs 0.69). The retention gain over the instruction baseline is +0.50 [0.23, 0.80] at light and +0.41 [0.17, 0.67] at medium. Under the random control there's no significant gain at either band.

That meets the pre-registered rule for **H-A**, the hypothesis I didn't expect.

{{< figure src="./images/icl_retention_by_context.png" width="800px" align="center" caption="Retention (ablated / clean accuracy) by context, for J-space ablation (red) and the norm-matched random control (blue). Only the 'same problem, new numbers' context lifts J-space retention.">}}

Some sanity checks:
- Clean accuracy rises only slightly with the same-problem context (+0.05, CI spanning zero). The gain is specifically in the *ablated* model (+0.21 [0.13, 0.30]).
- 11 problems had a variant whose answer happened to match the real answer. Excluding them doesn't change anything.
- The model predicts a demonstration's answer 8–13 times out of 112 in *every* condition, clean included, so it isn't just copying.

**MATH-500** showed the same thing, more weakly. Four *subject-matched* examples, which share procedures like "how to reduce mod n" with the test problem, raised J-space retention from 0.61 to 1.10 (+0.49 [0.16, 0.91]). Under random ablation they did nothing.

### Where the ablation actually bites

To see *which tokens* the ablation disrupts, I ran teacher-forced comparisons of clean vs. ablated next-token predictions on few-shot prompts with the reference solution appended. Two things came out of this:

1. **Generic text continuation isn't spared relative to noise.** On wikitext, J-space ablation flips 29% / 41% of top-1 predictions (light / medium), compared with 21% / 31% for the random control. The paper's "leaves the bulk of ordinary text prediction intact" holds only in absolute terms.
2. **The arithmetic tokens are the *most* stable, not the least.** This went against my own prediction. Computed numbers agree with the clean model 97% of the time, and their likelihood under ablation is as high as clean. The disruption is concentrated in the *reading* positions: the demonstrations, the question, and the opening of the solution. Those are where the model figures out what the problem is and what to do.

### Interpretation

The pattern across all the conditions follows a single gradient: **the more of the required procedure the context contains, the more it substitutes for the J-space**. It only ever substitutes for the J-space-*specific* damage, never for the random control's.

What routes around the ablation isn't the *format* of continuation. Plain text, unrelated examples and textbook prose all fail. It's having the *procedure* present in the context. That fits the paper's own explanation of why CoT is robust: CoT externalizes intermediate results onto the page, and here a same-problem demonstration externalizes the *plan*. Given a plan, whether by teacher forcing or by demonstration, the ablated model still computes the numbers correctly. What it loses is the ability to work out *what* to compute.

There's also a by-product for the original homework. With N=200, the specificity question finally resolves. At the light band, J-space and random ablation do equal damage. At the medium band, J-space ablation costs direct answering about **twice** what the random control does, while removing *less* norm. That's the first J-space-specific effect on a math task in this project. It also means the homework's "not J-space-specific" conclusion was really a statement about the light band.

### Limitations

- One model, one seed, greedy decoding. The CoT cells (N≈40) are only powered for large effects.
- The same-problem result is on the 112 problems that had safe variants, which are easier than average (clean accuracy 0.32 vs 0.265).
- Few-shot prompts get 4–20× more ablated positions than the baseline. The norm-matched control keeps the J-space vs. random comparisons fair, but retention comparisons *across* formats aren't intervention-matched.
- The random control matches norm, not output relevance. "Specific" here always means "worse than matched-norm noise."

---

## References

[1] W. Gurnee, N. Sofroniew, et al., "Verbalizable Representations Form a Global Workspace in Language Models," *Transformer Circuits Thread*, 2026. [arXiv:2607.15495](https://arxiv.org/abs/2607.15495)

[2] N. Nanda, external commentary on *Verbalizable Representations Form a Global Workspace in Language Models*, published by Anthropic, 2026. Includes an independent replication on Qwen.
