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

I used the simplest interpolant, the linear path, with one time $t \sim U(0, 1)$ shared across the whole sequence:

$$
x_t = (1 - t)\, x_0 + t\, x_1, \qquad \dot{x}_t = x_1 - x_0
$$

The model learns a velocity $\hat v_\theta(x_t, t)$ with an MSE loss against $x_1 - x_0$:

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

To generate, you start from noise and integrate the learned ODE $\dot x = \hat v_\theta(x, t)$ from $t=0$ to $t=1$ with 50 Euler steps. Then just unembed the final $x_1$ and take the argmax (or sample with some temperature) to get your sequence of tokens:

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

- **Bidirectional attention.** Since all the positions are denoised at once, we don't need to go in one direction. The model can see the entire sequence at once, so `CausalSelfAttention` just became `SelfAttention`.
- **Time conditioning.** A sinusoidal time embedder (with a small MLP) turns $t$ into a time embedding, and each block uses it to scale and shift its activations (DiT's adaLN-Zero).
- **A velocity head.** Instead of an LM head that outputs logits, the model just outputs a velocity.

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

The frustrating part was that the loss went down, so on paper training looked fine. Looking back, I think the main issue is that **my loss didn't have a real ground truth**.

I trained with MSE against the velocity $x_1 - x_0$ and treated that as the target. But $x_1 - x_0$ isn't actually ground truth. The real ground truth is the *tokens*. To get from tokens to $x_1$, we pass them through the embedding matrix (the "encoder"), and at the end we unembed back into tokens (the "decoder"). Those embeddings were being learned at the same time as the transformer, so nothing guaranteed they were the *right* encoder and decoder. The target the model was chasing kept moving along with it.

I actually saw this as a potential issue while I was building it, but I wasn't sure what the fix was. My worry with using a pretrained encoder/decoder was that its embedding space might not have the structure this kind of flow matching/velocity objective needs. So I figured I'd just train the embeddings and the transformer together and hope they'd find some good minimum.

This was my first big AI project, and looking back, that was a pretty naive thing to try. It didn't find that minimum. The MSE loss could still go down without the embeddings ever learning the structure needed to tell tokens apart, so when I unembedded at the end, I got random characters.

---

## What I'd Try Next

If I tried this again, I'd fix the ground-truth problem first: either use fixed or pretrained embeddings, or add a loss that ties things back to the actual tokens (like a cross-entropy loss on the unembedded $\hat x_1$), so the model can't lower the MSE without the embeddings staying meaningful.

---

## ELF: Someone Did It (Properly)

In May 2026, a few months after I forgot about this little project, Hu, Qiu, Lu, Zhao, Li, Kim, Andreas and He released [**ELF: Embedded Language Flows**](https://arxiv.org/abs/2605.10938) (NeurIPS 2026). To be honest, I haven't read it too closely yet, but it's very close in spirit to what I was going for: continuous-time flow matching in an embedding space instead of over discrete tokens.

One difference that jumped out is that they use a **pretrained encoder** (T5) for their embeddings, which could be exactly the fix for the problem with my version. I just haven't had time to revisit this yet.

---

## Final Thoughts

A couple of things I keep wondering about after this project:

**Does architecture research actually matter, given scaling laws?** The things that seem to matter most for language models are the number of parameters, the dataset size, and how much compute/training time you throw at it. So I wonder how this kind of architecture research will evolve. Will a flow-matching LM offer any *true* benefit over just scaling up a regular autoregressive transformer, or will it end up as a different way of spending the same compute?

**What do different interpolants buy you?** Here I used the linear interpolant. A big benefit of it, which I think is a really good one, is that you can generate the whole sequence in a few steps instead of needing one forward pass per token. The paths are close to straight, so you don't need many steps to follow them. But the [stochastic interpolants]({{< ref "/posts/flowFM" >}}) framework allows lots of other choices of interpolant, including ones that add noise along the path. I wonder what qualities those would have for generation, and whether they have other benefits that I'm not seeing.

---

## References

[1] M. S. Albergo, N. M. Boffi, E. Vanden-Eijnden, "Stochastic Interpolants: A Unifying Framework for Flows and Diffusions," 2023. [arXiv:2303.08797](https://arxiv.org/abs/2303.08797)

[2] W. Peebles, S. Xie, "Scalable Diffusion Models with Transformers" (DiT), 2023. [arXiv:2212.09748](https://arxiv.org/abs/2212.09748)

[3] K. Hu, L. Qiu, Y. Lu, H. Zhao, T. Li, Y. Kim, J. Andreas, K. He, "ELF: Embedded Language Flows," NeurIPS 2026. [arXiv:2605.10938](https://arxiv.org/abs/2605.10938)

[4] A. Karpathy, nanoGPT. [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
