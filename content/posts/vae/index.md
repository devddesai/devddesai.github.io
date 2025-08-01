---
title: 'An Overview of Autoencoders'
date: 2025-07-22T18:47:40-06:00
author: ["Dev Desai"]
draft: false
categories: ["notes"]
tags: ["machine learning", "python", "deep learning"]
showtoc: true
math: true
---
For my 2-D Ising Model project, I was reading up on *Denoising Diffusion Probabilistic Models* (DDPMs), which are generative models used in machine learning – especially for generating images. To better understand how DDPMs work, I decided to explore some of the foundational models they build upon, which led me to autoencoders, and specifically, variational autoencoders. 

In this post, I’ll give a brief overview of autoencoders and introduce two variants: the **variational autoencoder (VAE)** and the **sparse autoencoder (SAE)**.


## Autoencoders

Firstly, what exactly is an autoencoder? Simply put, an autoencoder is a neural network which compresses data ($X$) into a lower dimensional representation, called the *latent space (Z)* and reconstructs it back to its original form. 

It consists of two main components:
- **Encoder**: Maps the input data $x$ to a latent vector $z$
- **Decoder**: Maps the latent vector $z$ to a reconstructed vector $\hat x$

where $\mathbf x$, $\hat{\mathbf x}$ $\in X$ and $\mathbf z\in Z$.

{{< figure src="./images/autoencoder.png" width="500px" align="center" caption="A basic autoencoder architecture: the encoder compresses the input into a latent space, and the decoder reconstructs the input from this representation. Note, in this image, there is only one hidden layer, but deep autoencoders may have multiple in the encoder and decoder.">}}


The goal is for $\hat{x}$ to be as close as possible to $x$, hence when training the model, our loss function will typically be the L2 Reconstruction Loss. 

$$\mathcal{L}_{\text{recon}} = \| \mathbf{x} - \hat{\mathbf{x}} \|^2$$

Note that this model is not generative but rather aims to retain important information and reconstruct the compressed input vector. Next, let's explore sparse autoencoders (SAEs), which have a small twist from the standard autoencoder to promote feature selection.

---

## Sparse Autoencoders

I first heard of sparse autoencoders when my friend introduced a project on the mechanistic interpretability of large language models. In theory, a sparse autoencoder could be stuck inside of a neural network as a probe to help us see "features" that the model is learning. Before we dive into what SAE's are, here's a bit more about the motivation behind them.

Neural networks are highly complex, non-linear functions which rely on a concept called *polysemanticity* to model such complex datasets without blowing up their parameter space and running into the *curse of dimensionality* (overfitting too many variables, being computationally difficult to train a high dimension model). 

>*Polysemanticity* refers to how individual neurons do not represent singular features in the model. Instead, each neuron can activate in response to many different, seemingly unrelated patterns. 

It's the combination of these overlapping activations that allows the network to represent rich, complex models. This same strength that neural networks have is also what makes it difficult to interpret what it is actually "thinking." Sparse autoencoders aim to aid in this.

There are two key ideas that differ an SAE from an AE which could help us interpret a neural network:

>1. The latent space should be large (we can have more neurons in the latent layer than the input layer).
>2. We add a *sparsity* term in the loss function to encourage the activations of neurons to be close to 0.

$$ \mathcal{L} = \mathcal{L}_{\text{recon}} + \beta \cdot \mathcal{L}_{\text{sparsity}} $$

where $\beta$ is a hyperparameter describing the strength of sparsity constraints. Of course, there are many different functions we can use for $\mathcal{L}_{\text{recon}}$ and $\mathcal{L}_{\text{sparsity}}$. Some examples are L2 reconstruction loss or cross entropy loss, and  L1 regularization/LASSO regression.

SAE's do not aim to "compress" the input data and reconstruct it. These two design choices work together to force the network to represent features more explicitly. A large latent space gives the model enough capacity to represent many possible patterns in individual neurons, while the sparsity constraint ensures that only a few neurons activate for some input. This encourages neurons to be more distinct, and in theory, more interpretable, as they don't have complex overlapping relationships.

---
## Variational Autoencoders

Let's say we trained an autoencoder on the MNIST dataset (handwritten numbers). We can now have a way to compress handwritten digits. If we encoded input digit images and then stored the latent embeddings only, we would be able to decode those latent embeddings to generate handwritten digits.

But, these images would be essentially the same as our training data. What if we wanted to generate completely new unique handwritten digits? For instance, let's say we have a latent embedding of a training image 7. If we find a vector in the latent space close to $z$, by taking $z + \epsilon$, and pass it through the decoder, we should end up with a new handwritten 7 that is slightly different than the original one, right?

Actually, the answer is no. Regular autoencoders learn arbitrary latent spaces, and sampling from the latent space results in uninterpretable results. The key idea in variational autoencoders is that, instead of encoding an input into a single point in the latent space, a VAE encodes it to a distribution in the latent space.

ELBO


Sources: 
Image 1: https://www.geeksforgeeks.org/machine-learning/auto-encoders/ 