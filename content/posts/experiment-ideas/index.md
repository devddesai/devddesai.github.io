---
title: 'Experiment Ideas'
date: 2026-11-13
author: ["Dev Desai"]
draft: false
weight: 1  # pins this post to the top of the home page
featured: true  # shown as a highlighted card in post lists (layouts/_default/list.html)
summary: "Mini experiments I've randomly thought about and hope to get to one day. Reach out if you want to work on one together!"
categories: ["ideas"]
tags: ["ideas", "research", "interpretability", "AI safety"]
showtoc: true
math: true
---

*This is a running list of mini experiments I want to try. I plan to work on these soon, but if any of them sound interesting to you, [reach out](mailto:devdesai@g.harvard.edu)! I'd love to work on them together.*

---

## J-Space

Most of these come from reading Anthropic's [*Verbalizable Representations Form a Global Workspace in Language Models*](https://arxiv.org/abs/2607.15495) and from my own [J-space experiments]({{< ref "/posts/jspace" >}}). If you haven't seen J-space before, that post has a quick intro.

### J-space and LoRA

**Question:** If you take the J-space of a LoRA post-trained model and swap it into the base model, does the base model behave like the post-trained one?

**Why it's interesting:** I think I read somewhere that LoRA is most effective when applied around the middle third of the model. The J-space also lives roughly in the middle of the model (in Qwen3-4B, I found the workspace band starting around 36% depth). That overlap makes me wonder whether a lot of what post-training changes shows up *in* the J-space.

**Rough plan:**
1. Take a base model and a LoRA post-trained version of it.
2. Compare their J-spaces on the same prompts. Which J-lens directions are most active in one model but not the other?
3. Take the top-$k$ J-space representations from the LoRA model and patch them into the base model's residual stream.
4. Check whether the patched base model gets close to the post-trained model's results.

### Routing around the J-space with context (the "textbook" experiment)

**Question:** Can the right context let a model bypass J-space ablation entirely?

**Why it's interesting:** I ran some preliminary tests on this, which are written up in the [follow-up section of my J-space post]({{< ref "/posts/jspace#follow-up-can-in-context-learning-route-around-the-ablation" >}}). The short version: textbook chapters and unrelated worked examples don't help, but showing the model *the same problem with different numbers* undoes the J-space damage. It looks like the procedure being present in context is what matters, not the "autocomplete" format.

**Status:** [Preliminary results]({{< ref "/posts/jspace#follow-up-can-in-context-learning-route-around-the-ablation" >}}), want to do more

### Does the J-space get crowded?

**Question:** If the model is doing a hard *internal* task at the same time as an *external* one, does the external task get worse? Or does the model give up on the internal task?

**Why it's interesting:** The paper shows that if you ask a model to count to 5 internally while solving a problem, the J-lens shows it actually counting, even though it never says the numbers out loud. If the J-space is a limited workspace, it should eventually get crowded.

**Rough plan:** Keep the external task fixed and steadily increase the difficulty of the internal task (counting higher, tracking more items, harder mental arithmetic). Measure both external-task accuracy and how faithfully the J-lens shows the internal task being carried out.

### Multi-token concepts and the oracle lens

**Question:** How much are we missing because the J-lens only sees single-token concepts?

**Why it's interesting:** The J-space is inherently one token, but a lot of ideas need multiple tokens to express. In my own experiments, I couldn't tell apart "chain of thought substitutes for the workspace" from "the ablation missed part of a broader space the J-lens can't see." The paper's appendix describes an **oracle lens** that seems to get around this. I had a conversation with a researcher at Anthropic who may or may not have said that the oracle lens seems *very* promising.

**Rough plan:** Wait for that to come out, then rerun my ablation experiments with it.

### What does math training do to the J-space?

**Question:** If you train a model heavily on math and then ablate its J-space, how much of the math ability survives? Is most of what the math training added represented *in* the J-space, or does it go through some other mechanism?

**Rough plan:** Compare a base model with a math-specialized fine-tune of the same model. Run the same J-space ablation on both and see whether the fine-tune's extra math accuracy is more or less fragile than the base model's.

---

## Looped LLMs and the J-lens

**Question:** How do J-lens probes behave on looped (recurrent-depth) LLMs, where the same block is applied many times?

**Why it's interesting:** The J-lens is built from an averaged Jacobian per *layer*. In a looped model, "layer" turns into "iteration." Does a workspace form over loop iterations the way it does over depth? Does it show up at a consistent iteration?

---

## Safety Cartridges

*I came up with this idea during a lecture by my professor, Boaz Barak, where he talked about [deliberative alignment](https://arxiv.org/abs/2412.16339), the approach OpenAI used to train o1.*

This is what I'm currently working on: comparing where a safety policy (like a model spec) should live. It could be **in context**, in a **compressed KV cache** (a [cartridge](https://arxiv.org/abs/2506.06266) or attention-matching compaction), or **in the weights** through post-training, as in deliberative alignment. I have a lot of general ideas here, and some of the questions I keep coming back to are:

- **Does post-training beat in-context learning if the in-context model is the teacher policy?** If you distill from the model with the spec in context, does the trained model learn something more *causal* than the in-context version, or is it just a copy of it?
- **Jailbreak robustness.** Intuitively, jailbreaking seems easy when the model spec is just sitting in context and harder when it's trained in. But if the in-context model is the teacher, why would the student end up more robust than the teacher?
- **Scaling laws for safety, cartridge vs. post-training.** With a fixed compute budget, what's better: $2x$ compute on post-training, or $x$ on post-training plus $x$ on a KV cartridge?
- **Attention matching** as a training-free alternative to cartridges. When does it keep the right things, and when does it copy the teacher's mistakes?
- **Stacking.** How does stacking different LoRA adapters compare with stacking different cartridges, especially when the policies conflict? LoRA adapters are matrices added on top of the weights, so stacking them probably works fine but might cause a small, noticeable hit to the model's capabilities. Stacked cartridges are still just context, so they shouldn't harm the model's capabilities.

