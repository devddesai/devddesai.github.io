---
title: 'Annotated S4'
date: 2025-07-22T18:47:40-06:00
author: ["Dev Desai"]
draft: true
categories: ["notes"]
tags: ["machine learning", "python", "deep learning"]
showtoc: true
math: true
---

## Part 1: State Space Models

We want a model which takes in an input $u(t)$ and maps it onto $x(t)$, the latent space. $x(t)$ stores the information from the sequence, and $u(t)$ is the current signal. It does this by saying that $x(t)$ changes wrt to $x(t)$ and $u(t)$, giving us the equation $x'(t) = \textbf{A}x(t) + \textbf{B}u(t).$ 
- SSMs are based on control/signal processing theory, where linear models approximate reality well. This and also the fact that linear equations are much easier to solve than non linear equations, which often don't have closed form solutions, is why we assume a linear form.

- skip connections are just adding some linear transformation of the input to the output. they are good as it makes the model not have to take that into account. easy math, so they excluded it to focus on the actual model.