---
title: "Can We Route Around J-Space with Chain of Thought and In-Context Learning?"
date: 2026-10-15
author: ["Dev Desai"]
draft: false
categories: ["projects"]
tags: ["AI safety", "interpretability", "LLMs", "chain of thought", "in-context learning"]
showtoc: true
math: true
---

*This is some work I did inspired by my AI safety course.*

In July 2026, Gurnee, Sofroniew et al. at Anthropic published [*Verbalizable Representations Form a Global Workspace in Language Models*](https://arxiv.org/abs/2607.15495). One of its claims is that a language model has a small set of internal representations, which they call the **J-space**, that it uses for multi-step reasoning. If you ablate that space, the model can still talk fluently and do a lot of automatic inference, but internal multi-step reasoning falls apart.

The finding that caught my attention: on GSM8K, problems solved **with explicit chain of thought (CoT)** were much more robust to J-space ablation than the same problems answered directly. The idea is that writing steps down on the page and holding them in the internal workspace are partially interchangeable scratchpads.

I wanted to dig into this trade-off: does it keep holding as problems get harder?

---

## What is J-Space?

For a layer $\ell$, the paper averages the Jacobian of the final-layer residual stream with respect to the layer-$\ell$ residual stream, over positions and prompts:

$$
J_\ell = \mathbb{E}_{t,\, t' \ge t,\, \text{prompt}} \left[ \frac{\partial h_{\text{final}, t'}}{\partial h_{\ell, t}} \right]
$$

This is one $d_{\text{model}} \times d_{\text{model}}$ matrix per layer. It roughly captures an activation's tendency to produce some output, and you can use it like a lens:

$$
\text{lens}(h_\ell) = \text{softmax}\left( W_U \cdot \text{norm}(J_\ell \, h_\ell) \right)
$$

Basically, every token in the vocabulary has a **J-lens vector** at each layer. At every position in the prompt, the inner product of the model's activation vector with each vocabulary token's J-lens vector gives a reading of how much the model is "thinking about" every single token in the vocabulary. The **J-space** is just the top-$k$ tokens the model is thinking about most strongly.

Then we can try to **ablate** (get rid of) this internal workspace. At every position, across a band of layers:

1. Take the top-$k$ J-lens vectors ($k = 10$, for instance).
2. Check whether each token is already in the model's prediction for its actual next token. If it is, leave it alone, so we don't just delete the answer the model is about to say.
3. Project the rest of the directions out of the activation: $h \leftarrow h - (h \cdot \hat v)\hat v$.

---

## The Hypothesis

My hypothesis was that the CoT substitution **weakens** as problems get harder (GSM8K → MATH-500 → AIME). Writing a step down should help when the model just needs to *remember* earlier results. It should help less when coming up with a non-obvious next step still needs the internal workspace.

To test this, I defined a **robustness gap** as the difference in *relative* accuracy drop between the two modes:

$$
RG(d) = \frac{\text{CoT}_{\text{ablated}}(d)}{\text{CoT}_{\text{clean}}(d)} - \frac{\text{noCoT}_{\text{ablated}}(d)}{\text{noCoT}_{\text{clean}}(d)}
$$

If my hypothesis holds, $RG$ should **shrink** with difficulty. A stable or widening gap would be evidence against it.

---

## Setup

**Model:** Qwen3-4B (36 layers, $d_{\text{model}} = 2560$), bf16, greedy decoding, on a rented RTX 4090.

**The Jacobian:** built over 25 wikitext-2 prompts of 128 tokens each. Each prompt's first 4 positions are skipped because their norms are anomalous. This follows Neel Nanda's lighter-weight replication [[2]](#references) rather than the paper's 1,000-prompt corpus.

**Finding the workspace band:** the paper's analysis was on Claude, so I had to find which layers the J-space lives in for Qwen myself. I used two measurements: one for where the band starts and one for where it ends.

- **Where it starts (the "ignition" test):** take a sentence like "The capital city of ___ is called" and fill the blank with a *blend* of two countries' input embeddings, say 70% France and 30% Japan. Then slowly shift the blend from all-Japan to all-France and watch the activations at each layer. In early layers, the activations change smoothly as the blend shifts. Starting at some layer, they *snap* to either France or Japan instead, which is the model committing to one concept. For Qwen this starts at **layer 13 (36% of depth)**, close to the paper's ~38% for Claude.
- **Where it ends (lens hit rate):** at each position, check whether any of the top-10 J-lens tokens is the token the model actually predicts next. In the middle layers, this almost never happens (~5% of positions), because the J-space holds concepts the model is working with, not the next word. In the late layers it shoots up (to ~86% by layer 33), because those layers are just getting ready to output the next token. The paper calls these "motor" layers, and they're not the workspace we want.

Putting those together, I set the band at **L13–27** (36–75% of depth).

{{< figure src="./images/band_evidence.png" width="750px" align="center" caption="Blue: how sharply the activation snaps between the two blended countries (1× = smooth, no snapping), which jumps at layer 13. Red: how often the top J-lens tokens match the next-token prediction, which shoots up in the late motor layers. The shaded region is the band I chose.">}}

**The grid:** 3 difficulties × 3 ablation conditions (clean / J-space / random) × 2 reasoning modes (CoT / no-CoT), all on the same problems so comparisons are paired. The main grid ablates only the first third of the band (L13–17).

| Difficulty | Source | N |
| --- | --- | ---: |
| GSM8K | openai/gsm8k, test | 40 |
| MATH-500 | HuggingFaceH4/MATH-500, test | 30 |
| AIME | aime 2024 + aime 2025 | 20 |

**Controls:**
- **A norm-matched random-direction ablation.** It removes the same norm as the J-space ablation, but along random directions. Matching only the *number* of directions would confuse "removed more" with "removed the right thing."
- **A positive control.** Before the actual experiment, I tested that the ablation actually works using a factual recall task (more on this in the paper).
- **A sanity check.** I made sure the ablation didn't completely wreck the model's output.

---

## Results

### Checking the ablation works

I ran the positive control and the sanity check, and compared the J-space ablation against a random-direction ablation of the same size. The J-space ablation hurt factual recall more than the random one did, so it proved to work.

One more thing came out of this: ablating more layers damaged everything. Recall dropped a lot more at L13–22, and by L13–27 the model's output became incoherent. So for the main experiment I only ablate the first few layers of the band, **L13–17**, where the ablation hurts reasoning without breaking the model.

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

With chain of thought, the J-space ablation barely affected accuracy. So, at least for math problems at this difficulty, a written workspace seems to be able to stand in for the internal one.

There are some big caveats:

1. **It isn't shown to be J-space specific.** On GSM8K, the random control produces an equal or larger gap, and every math cell's random-vs-J-space interval spans zero.
2. **Only the lightest ablation was tested.** Widening the band on GSM8K drops CoT retention from 0.971 (L13–17) to 0.857 (L13–22) to 0.371 (L13–27). At the widest band, coherence itself breaks down. "CoT is robust" only held at light severity.
3. **There's no data at the hardest difficulty.** AIME sits at a capability ceiling for a 4B model.

A logical follow-up would be to use a larger model that can actually solve harder datasets, to see whether this still holds on much harder problems, since I wasn't able to test AIME.

---

## Follow-up: Can In-Context Learning Route Around the Ablation?

After finishing this, I had a separate idea I wanted to test. The paper notes that fluent, pattern-continuation-style generation keeps going under ablation even when the relevant concept is in the J-space. "Autocomplete" seems to route around the workspace.

A math textbook is full of complex reasoning that an LLM could just *autofill*. So what if, instead of instructing the model to solve a problem, you present the problem as **a document to be continued**? Does the ablated model recover the accuracy it loses when asked directly?

### Setup

I used the same setup as before, at two bands this time: light (L13–17) and medium (L13–22).

The contexts I tried, in order of how much of the *procedure* they contain:

| Context | What the model sees before the test problem |
| --- | --- |
| Instruction (chat) | The chat-template instruction from the first experiment |
| Plain text | `Problem: … / Solution:`, with no instruction or examples |
| 5 unrelated examples | 5 worked GSM8K training problems |
| Textbook + examples | A 272-word textbook-style chapter on word problems, then the 5 examples |
| Long textbook + examples | A 1,285-word chapter, then the examples |
| **Same problem, new numbers** | Up to 3 copies of the *test problem* with different numbers, solved |

### Results

| Band | Context | Clean | J-space | Random | Retention (J-space) |
| --- | --- | ---: | ---: | ---: | --- |
| L13–17 | Instruction | 0.321 | 0.170 | 0.223 | 0.53 |
| L13–17 | 5 unrelated examples | 0.330 | 0.196 | 0.205 | 0.60 |
| L13–17 | **Same problem** | 0.375 | **0.384** | 0.330 | **1.02** [0.84, 1.24] |
| L13–22 | Instruction | 0.321 | 0.098 | 0.232 | 0.31 |
| L13–22 | 5 unrelated examples | 0.330 | 0.125 | 0.214 | 0.38 |
| L13–22 | **Same problem** | 0.375 | **0.268** | 0.259 | **0.71** [0.53, 0.92] |

The unrelated examples and textbook chapters didn't really help. The ablated model lost just as much accuracy as it did with a plain instruction.

Showing the model **the same problem with different numbers** did help. At the light band, the ablated model was as accurate as the clean one.

At the medium band, here's how much of the clean accuracy survived each ablation:

| Prompt | J-space ablation | Random ablation |
| --- | ---: | ---: |
| Instruction | 0.31 | 0.72 |
| Same problem | **0.71** | 0.69 |

The same-problem examples brought the J-space-ablated model up to about the same level as the randomly ablated one. In other words, they undid the extra damage from removing the J-space. They didn't help against random ablation, though, so the effect is specific to the J-space.

{{< figure src="./images/icl_retention_by_context.png" width="800px" align="center" caption="Retention (ablated / clean accuracy) by context, for J-space ablation (red) and the norm-matched random control (blue). Only the 'same problem, new numbers' context lifts J-space retention.">}}

Maybe this isn't the most surprising result. If the model has already seen the same problem worked out, it might only need to do some simple calculations, rather than figure out how to solve the problem from scratch.

This was a pretty rushed, quick extension that I did for fun, and there's probably a lot more to dig into here. I do want to work on it a bit more, and I have it listed in my [experiment ideas]({{< ref "/posts/experiment-ideas" >}}).

---

## References

[1] W. Gurnee, N. Sofroniew, et al., "Verbalizable Representations Form a Global Workspace in Language Models," *Transformer Circuits Thread*, 2026. [arXiv:2607.15495](https://arxiv.org/abs/2607.15495)

[2] N. Nanda, external commentary on *Verbalizable Representations Form a Global Workspace in Language Models*, published by Anthropic, 2026. Includes an independent replication on Qwen.
