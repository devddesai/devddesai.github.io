---
title: 'nanoFM: Trying to Turn nanoGPT into a Flow Matching Model'
date: 2026-09-18
author: ["Dev Desai"]
draft: false
categories: ["projects"]
tags: ["AI", "flow matching", "generative models", "language models", "diffusion"]
showtoc: true
math: true
---

*This is the experiment I teased at the top of my [stochastic interpolants post]({{< ref "/posts/flowFM" >}}). Spoiler: it didn't work, and I never finished it. But I learned a lot, and a few months later a paper came out doing a much better version of the same idea, so I wanted to write up what I tried. The code is [here](https://github.com/devddesai/nanoFM).*

---

## Motivation

In my [last post]({{< ref "/posts/flowFM" >}}), I went through how stochastic interpolants unify diffusion and flow matching: instead of learning a noisy denoising path, you learn a **velocity field** that carries samples from a base distribution $\rho_0$ to a target distribution $\rho_1$.

Flow matching dominates image generation, but language is discrete. Diffusion LLMs like LLaDA mostly work with *discrete* noise (masking tokens). I wanted to try the simplest possible continuous version: take Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT), keep the transformer, and swap autoregressive next-token prediction for flow matching on **token embeddings**.

---

## The Setup

### Flow matching on embeddings

Tokens are discrete, but their embeddings aren't. So the "data" $x_1$ is the sequence of token embeddings, and the base sample $x_0$ is Gaussian noise of the same shape:

$$
x_1 = \text{wte}(\text{tokens}) \in \mathbb{R}^{T \times d}, \quad x_0 \sim \mathcal{N}(0, I)
$$

I used the simplest interpolant, the linear (rectified flow) path, with one time $t \sim U(0, 1)$ shared across the whole sequence:

$$
x_t = (1 - t)\, x_0 + t\, x_1, \qquad \dot{x}_t = x_1 - x_0
$$

The model learns a velocity $\hat v_\theta(x_t, t)$ by regressing onto $x_1 - x_0$ with an MSE loss:

$$
\mathcal{L} = \mathbb{E}_{t, x_0, x_1} \left[ \| \hat v_\theta(x_t, t) - (x_1 - x_0) \|^2 \right]
$$

In code, the training step looks like this:

```python
x1 = raw_model.transformer.wte(X)   # target embeddings
x0 = torch.randn_like(x1)           # noise
t = torch.rand(b, device=device)    # random time per sequence
xt = (1 - t_expanded) * x0 + t_expanded * x1
v_target = x1 - x0                  # velocity target

v_pred, loss = model(xt, t, v_target)  # MSE inside forward
```

### Sampling

To generate, you start from noise and integrate the learned ODE $\dot x = \hat v_\theta(x, t)$ from $t=0$ to $t=1$ with 50 Euler steps. That gives a sequence of continuous vectors. To get tokens back, you project onto the embedding matrix and take the argmax:

```python
for i in range(steps):
    t = torch.full((B,), i * dt, device=device)
    v, _ = self.forward(xt, t)
    xt = xt + v * dt

logits = xt @ self.transformer.wte.weight.T   # [B, T, vocab]
tokens = torch.argmax(logits, dim=-1)
```

Unlike autoregressive generation, all positions are generated **in parallel** and refined together over the steps.

### Changes to nanoGPT

Most of the architecture changes came from DiT (Diffusion Transformers):

- **Bidirectional attention.** There's no reason to be causal when every position is denoised at once, so `CausalSelfAttention` became plain `SelfAttention` with `is_causal=False`.
- **Time conditioning with adaLN-Zero.** A sinusoidal `TimeEmbedder` (an MLP on Fourier features of $t$) produces a time embedding. Each block maps it to a shift, scale and gate for both the attention and MLP paths. The LayerNorms lose their own affine parameters, and the modulation layers are zero-initialized so each block starts out as the identity.
- **A velocity head instead of an LM head.** `lm_head` (tied to the embeddings) became `v_head`, a `Linear(n_embd, n_embd)` that outputs a velocity in embedding space rather than logits over the vocabulary.
- **No shifted targets.** `get_batch` just returns $x$, since there's no next-token target anymore.

```python
class Block(nn.Module):
    def forward(self, x, t_emb):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_Modulation(t_emb).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(self.ln_1(x, scale_msa, shift_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(self.ln_2(x, scale_mlp, shift_mlp))
        return x
```

I trained on character-level Shakespeare (65-character vocabulary, ~1M training tokens), starting from nanoGPT's small Shakespeare config (6 layers, 6 heads, 384-dimensional embeddings).

---

## What Happened

The model produced random characters.

<!-- TODO: paste a sample output and/or the loss curve here if I still have one -->

The frustrating part was that the loss went down, so on paper training looked fine. Looking back at the code, I think I know why.

### Embedding collapse

The target $x_1 = \text{wte}(\text{tokens})$ isn't fixed: it comes from the **same learnable embedding table** the model trains, and I never detached it. That means the MSE loss can be lowered not only by learning a better velocity field, but also by **changing the embeddings themselves**.

The easiest way to make the velocity easy to predict is to make all the embeddings tiny and nearly identical. Then $x_1 \approx 0$ for every token, the target velocity is just $-x_0$, and the model can read that straight off its input. Three things in my setup pushed in this direction:

1. The embeddings start at nanoGPT's init (std 0.02), while the noise has std 1, so $x_1$ is already tiny compared to $x_0$.
2. nanoGPT applies weight decay to all 2D parameters, which includes `wte`, so the embeddings are actively shrunk.
3. Nothing in the loss rewards the embeddings for staying *distinguishable*.

So the loss goes down, but at decoding time the argmax is choosing between 65 near-identical vectors, which gives essentially random characters. Saving the best-val-loss checkpoint would make this worse, since the most collapsed model also has the lowest loss.

This is a known failure mode. Earlier continuous diffusion language models like [Diffusion-LM](https://arxiv.org/abs/2205.14217) and [CDCD](https://arxiv.org/abs/2211.15089) add a cross-entropy/rounding term or normalize the embeddings to prevent exactly this.

### Smaller issues

- **Time scaling.** The DiT time embedder is designed for $t \in [0, 1000]$, but I fed it $t \in [0, 1]$, so the sinusoidal features barely vary over training time.
- **Dropout on the noisy input.** nanoGPT's dropout of 0.2 was applied to $x_t$ itself, adding even more noise on top of the noise.
- **Dot-product decoding.** Taking the argmax of a raw dot product favors embeddings with large norms. L2 or cosine distance would be a fairer way to round back to tokens.

---

## What I'd Try Next

If I come back to this, the fixes I'd try, roughly in order:

1. **Stop the embeddings from collapsing:** either freeze `wte` (or use fixed, normalized embeddings), or add a cross-entropy loss on the predicted clean sample $\hat x_1 = x_t + (1 - t)\, \hat v_\theta(x_t, t)$ and remove `wte` from weight decay.
2. Scale $t$ by 1000 before the time embedder and set dropout to 0.
3. Decode with cosine or L2 distance instead of a raw dot product.
4. Track how often $\hat x_1$ rounds to the correct character at several values of $t$. That's a much more honest metric than the MSE loss, which, as I found out, can go down while the model gets worse.

---

## ELF: Someone Did It (Properly)

In May 2026, a few months after I shelved this, Hu, Qiu, Lu, Zhao, Li, Kim, Andreas and He released [**ELF: Embedded Language Flows**](https://arxiv.org/abs/2605.10938) (NeurIPS 2026). It's very close in spirit to what I was going for: a diffusion language model that runs **continuous-time flow matching in an embedding space**, instead of over discrete tokens like most current diffusion LLMs.

Their abstract highlights a few things:

- ELF **stays in continuous embedding space until the very last step**, where a shared-weight network maps the output to discrete tokens.
- Because everything is continuous, image-diffusion tricks like **classifier-free guidance** carry over directly.
- It outperforms leading discrete and continuous diffusion language models, with better generation quality using fewer sampling steps.

Its central argument, that continuous diffusion language models can work well with only *minimal* adaptation to the discrete domain, is basically the bet I was making with nanoFM. The difference is that they figured out the adaptations that matter, especially how to get from continuous embeddings back to tokens without everything collapsing. That was exactly where my version broke. I'd recommend reading it if this idea interests you.

---

## References

[1] M. S. Albergo, N. M. Boffi, E. Vanden-Eijnden, "Stochastic Interpolants: A Unifying Framework for Flows and Diffusions," 2023. [arXiv:2303.08797](https://arxiv.org/abs/2303.08797)

[2] W. Peebles, S. Xie, "Scalable Diffusion Models with Transformers" (DiT), 2023. [arXiv:2212.09748](https://arxiv.org/abs/2212.09748)

[3] X. L. Li et al., "Diffusion-LM Improves Controllable Text Generation," 2022. [arXiv:2205.14217](https://arxiv.org/abs/2205.14217)

[4] S. Dieleman et al., "Continuous diffusion for categorical data" (CDCD), 2022. [arXiv:2211.15089](https://arxiv.org/abs/2211.15089)

[5] K. Hu, L. Qiu, Y. Lu, H. Zhao, T. Li, Y. Kim, J. Andreas, K. He, "ELF: Embedded Language Flows," NeurIPS 2026. [arXiv:2605.10938](https://arxiv.org/abs/2605.10938)

[6] A. Karpathy, nanoGPT. [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
