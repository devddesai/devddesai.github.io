---
title: 'An Overview of Autoencoders'
date: 2025-07-22T18:47:40-06:00
author: ["Dev Desai"]
draft: false
categories: ["notes"]
tags: ["machine learning", "python", "deep learning"]
showtoc: true
---
Hey guys! For my 2-D Ising Model project, I was reading up on *Denoising Diffusion Probabilistic Models* (DDPMs), which are generative models used in machine learning – especially for generating images. To better understand how DDPMs work, I decided to explore some of the foundational models they build upon, which led me to autoencoders. 

In this post, I’ll give a brief overview of autoencoders and introduce two important variants: the **variational autoencoder (VAE)** and the **sparse autoencoder (SAE)**.


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

Neural networks are highly complex, non-linear functions which rely on a concept called *polysemanticity* to model such complex datasets without blowing up their parameter space. 

>*Polysemanticity* refers to how individual neurons do not represent singular features in the model. Instead, each neuron can activate in response to many different, seemingly unrelated patterns. 

It's the combination of these overlapping activations that allows the network to represent rich, complex models. This same strength that neural networks have is also what makes it difficult to interpret what it is actually "thinking." Sparse autoencoders aim to aid in this.

There are two key ideas that differ an SAE from an AE which could help us interpret a neural network:

>1. The latent space should be large (we can have more neurons in the latent layer than the input layer).
>2. We add a sparsity term in the loss function to encourage the activations of neurons to be close to 0.

SAE's do not aim to "compress" the input data and reconstruct it. Rather, their purpose is to try to understand what the model could be "thinking". Typically, we want to keep the dimensionality of the latent space lower to help with the curse of dimensionality (overfitting, being computationally difficult to train a high dimensional model), and neural networks work due to the idea of 

Sources: 
Image 1: https://www.geeksforgeeks.org/machine-learning/auto-encoders/ 

