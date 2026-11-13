---
title: 'Teaching a 3B Model to be Sheldon Cooper, and Crush a Physics Test'
date: 2026-10-03T18:47:40-06:00
author: ["Dev Desai", "Marina Lin", "Stefan Neuber"]
draft: false
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

### RL Motivation

The next step was to do some RL to further increase the Sheldon persona. To build some intuition, RL can be thought of as "distribution sharpening," where we push the model to increase the chances of it outputting something it already knows. For our example, the model might have both a Sheldon-like persona and a helpful assistant persona in its output distribution (since we performed LoRA SFT which adds on top of the current weights). Given some prompting and such, I'm sure you could get our SFT model to still speak like a helpful assistant. But, the goal of RL is to push the model towards the Sheldon distribution and "sharpen" it, meaning that it outputs Sheldon-ness more often. It's not actually learning anything new at this point - rather it's just being told which behavior is good and which not to do. 

There's two big names in RL training algorithms: PPO (Proximal Policy Optimization) and GRPO (Group Relative Policy Optimization). Roughly speaking, in PPO, you prompt the model and calculate the reward, which is some number that measures how aligned the model's output is to what we want. Then, another model (called a *critic*) estimates what reward the model usually gets for that prompt. Comparing the actual reward to that estimate tells you if the output was better or worse than expected, and then you can run the training step. GRPO aims to avoid using the critic (which is usually as big as the model itself) as it is expensive, and instead, you run the prompt $k$ times. Then you calculate the reward for each run, or *rollout*, find the average across, and then compute an *advantage* compared to the average for each rollout. The group average basically replaces the critic's estimate, so you don't need a whole extra model. Then you can do a training step, which pushes the model towards replies that score above the average. Also, you can (optionally, is a bit expensive) add a reverse KL divergence penalty (different from pretraining forward KL) which keeps the model from drifting too far from its initial distribution (can cause a mode collapse where the model overfits).

### Our Setup

For our RLAIF step, we used GRPO with $k = 4$ rollouts per prompt, plus the KL penalty I mentioned above. For the reward, we used an LLM judge (GPT-5.6 Luna) to score how Sheldon-like each reply was, from 0 (definitely not Sheldon) to 50 (could go either way) to 100 (a convincing Sheldon). We called this judge prompt **J01**, and the reward was just the score divided by 100. There was no physics reward at all in this stage.

{{< figure src="./images/workflow_rlaif.png" width="650px" align="center" caption="RLAIF: continue training the SFT model, using the judge's scores on its own sampled replies.">}}

### Picking the Judge

Since the judge is basically the entire reward, we wanted to make sure it could actually tell Sheldon apart from a normal assistant before training on it. We tried ten different judge prompts on three kinds of replies: real Sheldon lines from the show, normal AI replies, and AI replies that were told to sound like Sheldon. We went with J01, which separated real Sheldon from normal AI replies pretty well, with an AUROC of about 0.91. Roughly speaking, if you pick one real Sheldon reply and one normal AI reply, it scores the Sheldon one higher about 91% of the time.

### Results

For training data, we had GPT-5.6 Luna generate 3,000 everyday conversation snippets. Unlike SFT, there's no Sheldon reply to copy here - the model has to come up with its own replies, and the judge scores them. We also never told the model to act like Sheldon in the prompt, so all the Sheldon-ness had to come from the weights.

It worked pretty well! On the held-out prompts, the average judge score went from **41.66 → 70.43**. Here's the same prompt for both models:

> **SFT:** Good night. And thank you for the gift of your time. *(J01: 72)*
>
> **RLAIF:** Good night. And thank you for the gift of your time. I'm sure you're aware that I have no need for your services. *(J01: 92)*

One side effect was that replies got about **66% longer** (25 → 42 tokens on average). We never rewarded length, but longer replies give the model more room to throw in Sheldon-isms, and maybe more room to impress the judge. More on that later.

---

## Stage 3: RLVR on Physics

Now for the physics. This is the same distribution sharpening idea, but now the reward is just whether the model got the answer right. That's the "verifiable" part - we don't need a judge, since the answer key is the reward.

For testing, we used **488 physics questions from MMLU** (high school, college and conceptual physics). We never trained on MMLU, it was strictly our test set.

### Training Data

We trained on physics questions from **MMLU-Pro**, a harder version of MMLU. We removed any questions that originally came from MMLU (so nothing from the test set leaked in), and cut each question down to four choices to match MMLU's format.

We also filtered by difficulty, which goes back to how GRPO works. If the model always gets a question right (or always gets it wrong), every rollout gets the same reward, so none of them are above or below the average and there's nothing to learn. So out of 739 questions, we kept the 461 where the model's starting chance of picking the right answer was between 25% and 95%.

### Setup

The reward was:

$$
r = \mathbb{1}[\text{correct final letter}] + 0.25 \cdot \mathbb{1}[\text{valid boxed answer}]
$$

Basically, 1 point for the right answer, plus a small bonus for finishing with a clean `\boxed{}` answer. We used $k = 8$ rollouts per question and did one pass through the data on a single H100. The model could write up to ~480 tokens of reasoning, and if it didn't end with a boxed answer, we just asked it to pick a letter.

### Results

| Metric | SFT | RLAIF | RLVR (R2) |
| --- | ---: | ---: | ---: |
| High-school physics (151) | 46.36% | 46.36% | **52.32%** |
| College physics (102) | **54.90%** | 44.12% | 48.04% |
| Conceptual physics (235) | 64.26% | **70.21%** | 68.51% |
| **Overall physics (488)** | 56.76% | 57.38% | **59.22%** |
| **Mean J01 Sheldon score** | 41.66 | **70.43** | 67.98 |

{{< figure src="./images/physics_accuracy.png" width="550px" align="center" caption="Physics accuracy on the same 488 MMLU questions for each stage.">}}

Our best run (we called it R2) got **289/488** right, up from 280 for RLAIF. That sounds nice, but it's more of a reshuffle than a real gain: it fixed 30 answers and broke 21, for a net of +9 (about +1.8 percentage points). The confidence interval includes zero, so this could just be noise. Also, R2 was the best of several runs we tried, so take it with a grain of salt.

On the persona side, R2's Sheldon score dropped a bit, from 70.43 to 67.98, but that's also within noise.

{{< figure src="./images/persona_judge.png" width="600px" align="center" caption="Sheldon scores for every model. RLAIF gives the big jump, and none of the physics (RLVR) runs lose much of it.">}}

---

## What Didn't Work (and Why)

### Why were the physics gains so small?

My guess is that Qwen was already really good at this kind of stuff. According to the [Qwen2.5 Technical Report](https://arxiv.org/html/2412.15115v2), Qwen was pretrained on lots of scientific text, then post-trained on over a million SFT examples (including math and reasoning problems), and then trained with DPO and GRPO on top of that. Going back to the distribution sharpening idea, if Qwen's already been sharpened this much on these kinds of problems, there's probably not much left for a few hundred of our questions to sharpen. This is just a hypothesis though.

Also, our MMLU-Pro questions were mostly calculation-heavy, while MMLU has a lot of conceptual physics - and conceptual physics actually went *down* after RLVR.

### A right answer doesn't mean right reasoning

The verifier only checks the final letter, so it can't tell good reasoning from a lucky guess. On one question about wavelengths, RLAIF and R2 wrote the *exact same* wrong explanation (that gamma rays have the longest wavelength - they actually have the shortest). RLAIF still picked the right answer and R2 didn't. So R2 didn't really "unlearn" anything, RLAIF just got lucky. Some R2 answers also had the right reasoning, and then picked a letter that didn't match it.

Also, the formatting bonus never really worked: every single R2 answer still needed the "just pick a letter" fallback.

### The judge can be gamed

In one of our earlier runs, a reply just repeats *"I was in a state of profound joy"* over and over until it hits the token limit, and J01 gave it a **92/100**. That doesn't prove training caused the repetition, but it shows the judge has blind spots. J01 also liked AI-written Sheldon replies *more* than real Sheldon lines from the show.

Since RLAIF was optimizing against this exact judge, we can't rule out that part of the score jump came from the model learning to game the judge, rather than actually becoming more like Sheldon. The longer replies are at least consistent with that. To really check, we'd need humans to rate the replies, which we didn't do.

{{< figure src="./images/reply_length.png" width="600px" align="center" caption="Average reply length. RLAIF made replies a lot longer, even though length was never rewarded.">}}

---

## Earlier RLVR Attempts

R2 wasn't our first try. Before MMLU-Pro, we tried two other sources of physics questions:

- **OpenScience** (synthetic physics questions): barely moved anything. Our main run fixed 31 answers and broke 30, and using more data or more rollouts didn't help.
- **ARC-Challenge** (real science exam questions): most of the questions were too easy. 215 of the 308 questions were already answered correctly with over 90% probability before training, so there wasn't much for GRPO to learn from (same issue as the filtering above).

MMLU-Pro worked the best, probably because the harder questions gave GRPO more to work with. Here are all the runs we tried:

{{< figure src="./images/rlvr_all_runs.png" width="700px" align="center" caption="Every RLVR run we tried, compared to the RLAIF model it started from (with 95% confidence intervals). The MMLU-Pro runs do the best, but none of them are clearly above zero.">}}

### One more try: combining the rewards

Since R2's Sheldon score dropped a bit, we also tried adding the judge back into the physics reward, so the model gets rewarded for being right *and* sounding like Sheldon:

$$
r = \mathbb{1}[\text{correct}] + 0.25 \cdot \mathbb{1}[\text{boxed}] + 0.5 \cdot \frac{\text{J01(draft)}}{100}
$$

We only got to run this for 30 updates because of the deadline. It kept the Sheldon score close to RLAIF (69.78), but gained less physics (284/488). Neither change was big enough to be more than noise.

---

## Takeaways

- **SFT** on real dialogue was great for getting the model to actually talk like it's in the conversation, instead of acting like a helpful assistant.
- **RLAIF** made the model a lot more Sheldon-like according to the judge (41.66 → 70.43), but the judge has blind spots. If your reward is another LLM, you should assume the model will find its weak spots.
- **RLVR** on a few hundred physics questions barely moved a model that was already heavily post-trained.
- Checking only the final answer can't tell good reasoning from a lucky guess, so "accuracy went up" and "the model learned physics" are two different claims.

As for our original motivation, learning to be Sheldon didn't really make the model better at physics. SFT and RLAIF are within a point of each other in the table above, and compared to the base Qwen model, SFT was only about 1.6 points higher (on a slightly different physics eval), which is within noise. But the persona didn't hurt physics either, and none of the physics runs really destroyed the Sheldon-ness. I'd still love to try that LoRA overlap idea from the beginning, to see if the two actually share anything under the hood.
