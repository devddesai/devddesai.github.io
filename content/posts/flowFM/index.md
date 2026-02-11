---
title: 'Generalizable Diffuion via Stochastic Interpolants'
date: 2026-01-20T18:47:40-06:00
author: ["Dev Desai"]
draft: false
categories: ["notes"]
tags: ["AI", "diffusion", "generative models", "stochastics"]
showtoc: true
---

*Very soon, I will publish my project implementing a linear interpolant/flow matching to Andrej Karpathy's nanoGPT architecture to this page!*

Recently, I've been interested in novel generative techniques. We've seen the advent of diffusion over the past few years. Diffusion transformers are the gold standard for modern-day image generation, and there has been a lot of research on using diffusion for language modeling, sparked by the famous LLaDa model. Newer advancements include hybrid LLMs built on "Block Diffusion" to take advantage of diffusion's parallel token generation whilst utilizing autoregression's flexible-length generation and KV Caching.

Diffusion's core concept revolves around moving from one probability distribution to another by learning a noisy, diffusive path. Flow Matching was a framework which stated that instead of learning the noisy path and steps to take to move from a distribution to another, you can just learn the velocity field. This can be generalized using the ***Stochastic Interpolants*** framework [[1]](https://arxiv.org/abs/2303.08797), which aims to unify diffusion and flow matching.


---
## Diffusion and its Limitations

To understand the motivation behind stochastic interpolants, let's take a quick walk through the diffusion process. 

Let's say we have a target probability distribution, $\rho_{1}$ that represents the space that contains all the possible images/text we want to generate. There is no way we would know that entire distribution, otherwise we could just randomly sample from it to "generate." 

To try to approximate this distribution, diffusion aims to learn how to nudge Gaussian noise over many steps, eventually leading it to the target distribution. We start with a sample from $\rho_{1}$ and we slowly add Gaussian noise to it until it becomes pure Gaussian noise. The model then learns how to "undo" that in a denoising process. Eventually, we can just sample from $\rho_{0} \sim \mathcal{N}(0,1)$, and allow the trained model to iteratively remove Gaussian noise, eventually leaving us with a sample from $\rho_{1}$.

Some fragility of diffusion can be seen through these:

> 1. The base density, $\rho_0$, must be the standard normal. According to specific use cases, you might want to condition on a different base density. An example is if you wanted to take images of purple flowers and generate images of their yellow counterparts.
> 2. The noising and denoising process happens as $t$ approaches infinity. 
> 3. The sampling process is seen as the time reversal of some forward noising process. 

While diffusion has proved its worth, there is a more robust stochastic process we can define to capture diffusion's strengths while also accounting for its weaknesses.

--- 

## Stochastic Interpolants

A stochastic interpolant is a stochastic process defined by

$$ x_t = \alpha(t) x_0 + \beta (t) x_1 + \gamma(t)z$$

where $x_0$ is a sample from some base density $\rho_0$ (doesn't have to be Gaussian), $x_1$ is a sample from the target density, and $\rho_1$, $z \sim \mathcal{N}(0,1)$. The time-dependent coefficients satisfy the following boundary conditions:

 $$\alpha(0) = 1, \alpha(1) = 0$$
 $$\beta(0) = 0, \beta(1) = 1$$
 $$\gamma(0) = 0, \gamma(1) = 0$$

This framework allows us to define a whole family of generative models, including diffusion and flow matching, by choosing different $\alpha$, $\beta$, and $\gamma$ functions. For example, setting $\gamma(t) = 0$ gives us flow matching, while specific choices of these functions can recover diffusion processes.

Some key properties of stochastic interpolants include:
>1. Exact connection: $x_t$ interpolates between $x_0$ and $x_1$ as $t$ goes from 0 to 1.
>2. Ease of sampling: The sampling process can be designed to be more efficient than traditional diffusion methods.
>3. Flexibility in base distribution: The base distribution $\rho_0$ can be chosen to suit the specific application.

Through the following image, we can see how this framework can be powerful in unifying different generative modeling techniques through its flexible approach.

{{< figure src="./images/flowers.png" width="800px" align="center" caption="An illustration of how using different interpolant schedules can affect the latent state at different time steps. [1](https://arxiv.org/abs/2303.08797)">}}

We can see how choosing different $\alpha$, $\beta$, and $\gamma$ functions can lead to different latent states at various time steps. This flexibility allows for tailoring the generative process to specific needs, potentially improving efficiency and quality. Some key observations from the image above: 

- The base density is flexbile. Here, $\rho_0$ is white flowers, $\rho_1$ is purple flowers, and $x_t$ interpolates between the two! 

- Despite different noise schedules, the final sample at $t=1$ is always from the target distribution $\rho_1$. Even the first process which has zero noise at all still manages to reach the target distribution. What is even more fascinating to me is that in the first's intermediate states, the image is still discernable as a flower. This suggests that the model is learning a more than just the base and target distributions, and how to move between them, but a meaningful latent space. 

- We can see that different noise schedules lead to different intermediate states. This suggests that we can optimize the noise schedule for different tasks, such as faster sampling or better quality.

- The stochastic interpolant framework allows for a continuous transition between diffusion and flow matching, providing a unified view of generative modeling. The first process is pure flow matching, while the last is more akin to diffusion, as it is learning to denoise Gaussian noise.

{{< figure src="./images/measure.png" width="800px" align="center" caption="The intermediate distributions between two and three mode gaussian mixture based distribution. [1]">}}

This figure shows how different interpolant schedules affect the intermediate distributions when transitioning between two and three mode Gaussian mixture-based distributions. How can we characterize these intermediate distributions?

The time-dependent density $\rho(t, x)$ of the interpolant $x_t$ satisfies the transport equation

$$\partial_t \rho(t, x) + \nabla \cdot (b(t, x) \rho(t, x)) = 0, \quad b(t, x) = \mathbb{E} [\dot{x}_t | x_t = x].$$

Here, $b(t, x)$ is the drift field, representing the expected change in $x_t$ given its current state. The time-dependent density $\rho(t, x)$ also satisfies the family of Fokker-Planck equations. This ensures that the density correctly evolves over time.
$$\partial_t \rho(t, x) + \nabla \cdot ([b(t, x) + \epsilon(t) \nabla \log \rho(t, x)] \rho(t, x)) = \epsilon(t) \Delta \rho(t, x),$$

Great, but how do we practically use this to generate samples for our use? First, we train our model to minimize the square error of the drift ($\hat{b}$) and score ($\hat{s} \Leftrightarrow \nabla \log \rho$) functions.

$$\mathcal{L}_b(\hat{b}) = \int_{0}^{1} \mathbb{E} \left[ \|\hat{b}(t, x_t) - \dot{x}_t\|^2 \right] dt, \quad \mathcal{L}_s(\hat{s}) = \int_{0}^{1} \mathbb{E} \left[ \|\hat{s}(t, x_t) + \frac{1}{\gamma(t)} z\|^2 \right] dt$$ 

We can then just solve the corresponding ODE or SDE to generate samples from the target distribution!

$$\dot{x}(t) = b(t, x(t)), \quad dX_t = (b(t, X_t) + \epsilon(t) \nabla \log \rho(t, X_t)) dt + \sqrt{2\epsilon(t)} dW_t$$

---

## Commentary
Stochastic interpolants provide a powerful and flexible framework for generative modeling by unifying diffusion and flow matching. I believe exploring this flexibility can lead to more efficient and effective generative models. We have seen its potential in image generation, but I am interested in seeing its application in language modeling or other domains! We have seen diffusion LLMs perform well, but perhaps stochastic interpolants can provide a more efficient approach, and I aim to explore this in future work.

---

## References
[1] https://arxiv.org/abs/2303.08797

[2] https://www.imsi.institute/videos/generative-modeling-with-stochastic-interpolants/
