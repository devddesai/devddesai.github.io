---
title: 'Teaching a 3B Model to be Sheldon Cooper, and Crush a Physics Test'
date: 2026-09-28T18:47:40-06:00
author: ["Dev Desai", "Marina Lin", "Stefan Neuber"]
draft: true
categories: ["projects"]
tags: ["Post-training", "Reinforcement learning"]
showtoc: true
math: true
---

In my AI Safety class, taught by Boaz Barak, our first hw assignment was to take a small LLM and post train it to sound like a fictional character of our choice and improve its performance on a STEM benchmark. Our group picked Sheldon Cooper, the theoretical physicist from *The Big Bang Theory*. We chose Sheldon Cooper as our persona and physics as our STEM benchmark. Our motivation for this choice was that perhaps finetuning Sheldon's persona would automatically improve physics performance, since Sheldon is a physicist. The training data might have some reasoning styles that are useful for thinking through physics problems. 

Now that I'm writing this, I'm thinking it would cool to see if a Sheldon-persona LoRA and a physics LoRA share any overlapping representations? This would concretely demonstrate if the Sheldon persona elicits physics knowledge or reasoning ability. I'll add it to my long laundry list of future ideas.

Anyways, we used Qwen 2.5-3B as our base model, and post trained it in three stages:

1. **Supervised fine-tuning (SFT)** on real Sheldon dialogue
2. **RL from AI feedback (RLAIF)**, where an LLM judge scores how "Sheldon-like" a reply is
3. **RL with verifiable rewards (RLVR)** on physics multiple-choice questions with answer keys

The first two steps were aimed to learn the persona, while the third was to improve physics performance. Throughout, we did evals on both persona and physics to see if any step damaged (or helped) one or the other.

---

## Stage 1: Supervised Fine-Tuning

Firstly, we wanted to get the model to sound like Sheldon. To do this, we first wanted to perform supervised fine-tuning (SFT). The reason to do SFT before any RL is that SFT is a method which helps models learn newer tasks and domains, whilst RL helps a model output more of what it already knows to increase some reward - we'll get to that later.

So firstly, we needed to get some training data. We had two options - scrape transcripts from the show, or use a larger model to generate Sheldon-like dialogue. We decided to do the first, for 2 reasons. Firstly, we wanted the model to learn from the real Sheldon, not some model's imitation (which likely would be over the top). Mainly, we wanted to prevent what our TA refers to as "model cannibalism" - where a smaller model learns to cheaply imitate some nuanced statistics of the larger model's outputs rather than learning the richer representations, causing a mode collapse.

We scraped the transcripts and turned them into user-assistant pairs, where the user is a few lines of preceding dialogue, and the assistant is Sheldon's reply (filtered out super short replies, references to visual cues, and other junk). We also made sure to split the train/test data by episode, and not by line, as there is likely some correlation between intra-episode lines.

The largest effect of SFT was getting the model to *respond in the conversation* instead of explaining it like a helpful outside assistant:

>**User:** Saturday night at the comic book store, they're having a Mystic Warlords of Ka-ah tournament. First prize is five hundred dollars. If we team up, we'd be unstoppable.
>
>**Base Qwen:** That sounds like an exciting opportunity! ... If you and your friend decide to team up, you might want to consider the following: 1. *Team Strategy* ... (followed by a long explanation of how to win the tournament)
>
>**SFT:** I'm not sure I want to be part of an event that's so obviously based on Star Wars.

---

## Stage 2: RLAIF

For RLAIF, we used **GRPO** (Group Relative Policy Optimization). For each prompt, the policy samples several replies, a reward model scores them, and the policy is pushed toward replies that score above the *group average* for that same prompt. A KL penalty keeps the policy from drifting too far from where it started (the SFT model).

Our reward model was an LLM judge (GPT-5.6 Luna) with a rubric we called **J01**, scoring Sheldon-likeness from 0 (incompatible) to 100 (distinctive and convincing). The reward was just `score / 100`. There was no physics reward and no separate "quality" reward.

{{< figure src="./images/workflow_rlaif.png" width="650px" align="center" caption="RLAIF: continue the SFT model, using judge scores on its own sampled replies.">}}

**Picking the judge.** Before trusting a judge with the reward, we compared ten rubric prompts on three kinds of replies: real transcript replies, ordinary AI replies, and AI replies explicitly written to sound like Sheldon. J01 separated real Sheldon from ordinary AI replies with an AUROC of about **0.91** on 510 fully scored contexts.

**Training setup.** 3,000 generated everyday-conversation contexts (with no target reply), four sampled replies per prompt, one epoch, learning rate $5\times10^{-6}$, KL coefficient 0.04. The policy got no persona instruction in the prompt, so all of the Sheldon had to come from the weights.

It worked: the mean J01 score on the 511 held-out prompts went from **41.66 → 70.43**. Same prompt, SFT vs. RLAIF:

> **SFT:** Good night. And thank you for the gift of your time. *(J01: 72)*
>
> **RLAIF:** Good night. And thank you for the gift of your time. I'm sure you're aware that I have no need for your services. *(J01: 92)*

That example also shows a side effect. RLAIF replies got about **66% longer** (25.4 → 42.0 tokens on average), and hit the 256-token limit more often (12 → 31 replies). Length was never rewarded directly, but longer replies leave more room for persona cues, and possibly more room to impress the judge.

---

## Stage 3: RLVR on Physics

Now for the physics. All physics evaluation uses the same **488 MMLU physics questions** (high-school, college and conceptual physics). We never trained on MMLU. It was strictly the test set.

### The run that worked best (R2)

Our best physics result came from continuing the RLAIF model on **MMLU-Pro physics** questions. To avoid contamination, we removed 411 MMLU-sourced items, checked overlap with the test set and demonstrations, and cut each question down to four choices.

We also filtered for difficulty. If the model already gets a question right ~100% of the time (or never does), all GRPO rollouts get the same reward and there's no learning signal. So from 739 candidates, we kept 461 where the model's starting probability on the correct letter was between 0.25 and 0.95.

The reward was:

$$
r = \mathbb{1}[\text{correct final letter}] + 0.25 \cdot \mathbb{1}[\text{valid boxed answer}]
$$

The format bonus encourages the model to actually finish with a clean `\boxed{}` answer. Setup: eight rollouts per question, eight questions per update, 57 updates (one pass), learning rate $10^{-5}$, KL coefficient 0.04, on a single H100. No AI judge was involved here, since the answer key is the reward.

Generation was capped at 512 tokens, with up to 480 for a reasoning draft. If the draft didn't end in a valid box, the same model was asked to pick a letter A–D (without seeing the key).

### Results

| Metric | SFT | RLAIF | RLVR (R2) |
| --- | ---: | ---: | ---: |
| High-school physics (151) | 46.36% | 46.36% | **52.32%** |
| College physics (102) | **54.90%** | 44.12% | 48.04% |
| Conceptual physics (235) | 64.26% | **70.21%** | 68.51% |
| **Overall physics (488)** | 56.76% | 57.38% | **59.22%** |
| **Mean J01 Sheldon score** | 41.66 | **70.43** | 67.98 |

{{< figure src="./images/physics_accuracy.png" width="550px" align="center" caption="Zero-shot, 512-token boxed-answer evaluation on the same 488 MMLU physics questions.">}}

So R2 got **289/488**, up from 280 for RLAIF. That sounds nice, but it's more of a reshuffle than a clean gain: R2 fixed 30 answers and broke 21, for a net +9 (+1.84 percentage points). The 95% paired interval is **[−1.02, +4.71]** with exact paired $p = 0.26$. That's not statistically clear. Also, R2 is the *best* of several runs we evaluated on the same benchmark, so this should be read as exploratory.

On the persona side, R2's J01 score dropped by 2.45 points to 67.98 (95% interval [−4.85, +0.23]). That's the largest drop among our RLVR adapters, but the interval still includes zero.

{{< figure src="./images/persona_judge.png" width="600px" align="center" caption="Persona scores across all adapters. RLAIF gives the big jump; every RLVR run stays close to it.">}}

---

## What Didn't Work (and Why)

### Why were the physics gains so small?

The [Qwen2.5 Technical Report](https://arxiv.org/html/2412.15115v2) (Sections 3–4) describes a lot of scientific pretraining text, over a million SFT examples including math and logical reasoning (multiple choice too), and then DPO and GRPO. MMLU and MMLU-Pro are listed as *evaluations*, not training sets, but the model has clearly already been trained hard on the relevant skills.

Our guess is that a few hundred physics questions add very little new signal to a model that's already this post-trained. That's a hypothesis though: the report doesn't prove the model is saturated on physics. Also, our MMLU-Pro training pool leaned toward numerical problems, which may not transfer well to MMLU's mix (especially conceptual physics, which actually went *down*).

### Correct answers can be fragile

A correct final letter doesn't mean correct reasoning. On one wavelength question, RLAIF and R2 wrote the *exact same* wrong explanation (that gamma rays have the longest wavelength). RLAIF still picked the correct option D; R2 picked A. RLVR didn't "unlearn" anything here; the original correct answer was luck.

The flip side happened too: some R2 drafts had the right explanation and then picked an inconsistent letter. Since the verifier only checks the letter, it can't tell sound reasoning from a lucky guess.

### Formatting never really stuck

Despite the format bonus, **every** R2 eval answer needed the fallback "pick a letter" step. That's a formatting failure, not proof that there was no reasoning: plenty of drafts had real calculations in them. None of the 21 regressions hit the draft token limit, so truncation doesn't explain them either.

Training logs showed no clear upward trend in correctness and very little policy drift. Since each batch had different questions, that doesn't prove zero learning, but one pass over a small, mostly numerical set probably isn't much signal.

### A high persona score can hide a bad reply

In an earlier MMLU-Pro run (run E), one reply repeats *"I was in a state of profound joy"* over and over until it hits the 256-token limit. J01 gave it **92/100**.

> That's a judge weakness, not proof that training caused the repetition. But it's a good reminder that the judge is the same thing RLAIF was optimizing against.

J01 also preferred *generated* Sheldon-style replies over *real* transcript replies during calibration, and held-out examples helped us pick the rubric. Without an independent human evaluation, we can't rule out that RLAIF partly learned to game the judge, and the longer replies are at least consistent with that.

{{< figure src="./images/reply_length.png" width="600px" align="center" caption="Mean generated reply length. RLAIF made replies much longer even though length was never rewarded.">}}

---

## Earlier RLVR Attempts

R2 wasn't our first try. Before MMLU-Pro, we tried two other physics sources, all starting from the RLAIF model and tested on the same 488 MMLU questions:

- **OpenScience** (synthetic physics questions with answer keys): the 300-question, eight-rollout run fixed 31 answers and broke 30, for a net +1. Earlier letter-only versions gained 8 (300 questions, 4 rollouts) and 6 (500 questions, 16 rollouts). More data and more sampling didn't help.
- **ARC-Challenge** (real exam questions): most questions were already too easy. 215 of 308 had over 90% probability on the right answer before training, so there was little reward variation for GRPO to learn from. Filtering down to uncertain questions left very few. Higher learning rates ($5\times10^{-5}$) lost answers, and both ARC runs were sensitive to option order on shuffled-choice MMLU.

{{< figure src="./images/earlier_rlvr_accuracy.png" width="700px" align="center" caption="Paired physics changes vs. the starting RLAIF model, with 95% bootstrap intervals. Every interval includes zero.">}}

The MMLU-Pro follow-ups did better because harder questions gave more reward variation. Letter-only run E gained 9 answers (persona 70.06), a couple of other seeds/learning rates gained 7, and reasoning run R1 gained 6, before R2 changed the learning rate, batch size and format bonus together.

{{< figure src="./images/rlvr_all_runs.png" width="700px" align="center" caption="Every RLVR run we evaluated, as paired MMLU physics change vs. its matched RLAIF starting point (95% CI), grouped by evaluation protocol. The MMLU-Pro runs do best, but no interval sits clearly above zero.">}}

### One more try: a combined reward

Since R2 had the biggest persona drop, we also tried putting the judge back into the RLVR reward:

$$
r = \mathbb{1}[\text{correct}] + 0.25 \cdot \mathbb{1}[\text{boxed}] + 0.5 \cdot \frac{\text{J01(draft)}}{100}
$$

This only ran for 30 updates (deadline), but it kept the persona score near RLAIF (69.78) while gaining less physics (284/488). Neither change was statistically clear.

---

## Takeaways

- **SFT** on real dialogue is great for getting a model to stay in the conversation instead of acting like an assistant.
- **RLAIF** boosted the judge-measured persona a lot (41.66 → 70.43), but the judge has blind spots. A reply stuck in a loop scored 92/100, and replies got longer across the board. If your reward is an LLM, you should assume the policy will find its weak spots.
- **RLVR** on a few hundred physics questions barely moved an already heavily post-trained 3B model. Our best run was +1.84 pp with an interval that includes zero.
- A verifier that only checks the final letter can't distinguish good reasoning from a lucky guess, so "accuracy went up" and "the model learned physics" are different claims.

Overall, the persona and physics seemed mostly *compatible*: none of the RLVR runs clearly destroyed the Sheldon-ness. But we also never got a physics gain big enough to really test that tradeoff.
