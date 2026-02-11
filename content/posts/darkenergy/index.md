---
title: 'Temporal Evolution of Dark Energy'
date: 2026-01-20T18:47:40-06:00
author: ["Dev Desai", "Gautam Krishnan", "William Brown"]
draft: false
categories: ["notes"]
tags: ["AI", "diffusion", "generative models", "stochastics"]
showtoc: true
---

Throughout my senior year at UT, I've been taking astronomogy and cosmology courses out of personal interest. As you may know, dark energy is supposedly responsible for the accelerated expansion of the universe. In this blog post, I will discuss my team's research on the temporal evolution of dark energy through analysis of the DESI (Dark Energy Spectroscopic Instrument) and DES (Dark Energy Survey) datasets.


# Background

Before the discovery of dark energy, it was believed that the expansion of the universe would eventually slow down due to gravity. Scientists aimed to measure the rate of this deceleration through the deceleration parameter: 

$$q = -\frac{a \ddot{a}}{\dot{a}^2}$$

where $a$ is the scale factor of the universe. After doing some algebra that I won't go into here (check out Barbara Ryden's *Introduction to Cosmology,* great textbook), we can see that there would be 3 fates of the universe depending on the value of of the mass density of the universe, $\Omega_m$:

1. If $\Omega_m > 1$, the universe would eventually stop expanding and recollapse in a "Big Crunch."
2. If $\Omega_m = 1$, the universe would expand and approach zero as $t \to \infty$
3. If $\Omega_m < 1$, the universe would expand forever at a decreasing rate.

Under this model, $q =  \frac{1}{2} \Omega_m$, so the universe should always be decelerating with $q>0$. However, in 1998, two independent teams of astronomers studying distance measurements of Type Ia supernovae found that the expansion of the universe was actually accelerating, with $q < 0$. This was a groundbreaking discovery that led to the introduction of dark energy as a new component of the universe to explain this acceleration.

> Note: Type Ia supernovae are a class of supernovae that occur when a white dwarf star accretes enough matter and crosses the Chandrasekhar limit (which is the critical mass at which a white dwarf can no longer support itself against gravitational collapse), and explodes in a supernova. Because we know the chandrasekhar limit is about 1.4 solar masses, we also know the intrinsic brightness of Type Ia supernovae, so we can use the observed brightness of Type Ia supernovae to determine their distance, making them "standard candles" for measuring cosmic distances.

This discovery forced the deceleration paramter to be redefined as:

$$q = \frac{1}{2} \Omega_m - \Omega_\Lambda$$

where $\Omega_\Lambda$ is the density parameter for dark energy. With this new model, if $\Omega_\Lambda$ is large enough, it can lead to $q < 0$, which explains the observed acceleration of the universe. This led to the $\Lambda$CDM model, which is the current standard model of cosmology. In this model, the dark energy density is considered to be constant over time. However, the question of whether dark energy is truly constant or if it evolves over time remains an open question in cosmology. 

## Motivation

Some theories suggest that dark energy could be dynamic and change over time, such as quintessence, or it could be a manifestation of modified gravity. From quintessence emerges the CPL (Chevallier-Polarski-Linder) parametrization, which models the equation of state of dark energy as:

$$w(z) = w_0 + w_a \frac{z}{1+z}$$

where $w(z)$ is the equation of state parameter of dark energy, $w_0$ is its present-day value, and $w_a$ describes how it evolves with redshift $z$. If $w(z) = -1$ for all $z$, then dark energy behaves like a cosmological constant. However, if $w(z)$ deviates from -1, it could indicate that dark energy is dynamic and evolving with redshift. Because higher redshifts correspond to earlier times in the universe, analyzing the behavior of $w(z)$ at different redshifts can provide insights into the temporal evolution of dark energy.

# Analysis

To understand whether $w$ evolves with $z$, we can fit observational data to the CPL parametrization. If $w_0 \to -1$ and $w_a \to 0$, then the $\Lambda$CDM must hold true, as $w(z) = -1$. But if the parameters deviate from (-1, 0), we will have evidence to support temporally evolving dark energy. 

## First Methodology

In the DESI dataset, we can find `chain1`, `chain2`, `chain3`, `chain4` files, which contain **EDIT HERE** Monte Carlo simulations of the universe with different parametrizations, and compare the fidelity of those simulations with current astronomical surveys. In this, they tested the CPL parametrizations, and had a respective $\chi^2$ value, allowing us to see which parametriztion best fits the universe. 

We plotted a heatmap of the data to see this. **EDIT HERE, ADD AN INTERACTIVE VISUAL?**

We can see that the $\Lambda$CDM model is far from the best fit paramtrization. Yet, the best fit occurs at the edge of the dataset, meaning the true optimized parametrization could be elsewhere. 

## Second Methodology

We turned to the DES dataset, where we found Distance Modulus, $\mu$, of Type Ia supernovae and their associated redshift values, $z$. $\mu$ and $z$ follow a certain relationship according to the parametrization of the equation of state, $w$. If we use the CPL parametrization, this yields the following relation:

**INSERT EQUATION HERE**

From here, we can run a regression to find which parametrization best fits our $(z, \mu)$ data. Things get a bit hairy as this is not a linear relationship, and has an integral in it. Here is an interactive visual allowing us to visualize the parameter space and its behavior on $\mu$. 

**insert visual**

Since this is not a clean, linear regression, but rather follows a non-linear relation with an integral in it, I decided to utilize particle swarm optimization, as the parameter space isn't too large. 

>Particle Swarm Optimization, or PSO, is an optimization algorithm which is useful when **insert help**. It initializes some $n$ particles, or parametrization candidates, at random locations with random velocities in the parameter space. It calculates the loss function for each particle, and decides who the best candidate is. At the next step, the particles move according to their velocity, and the velocity is then updated with a weighted sum of the particle's previous momentum and the velocity to reach best candidate's location. After a series of steps, the particles should converge to the optimal candidate.

In order to calculate the $\mu_{\text{pred}}$, I utilized `Scipy`'s `quadrature` integrator. View an animation of the PSO algorithm at work here!

**insert visualization**

From this, we derive that the optimal parametrization of $(w_0, w_a)$ is **insert**, leading us to believe that the CPL parametrization is more accurate at describing the nature of dark energy than the $\Lambda$CDM model. To test the confidence of our solution,

 **insert**.


# Discussion

The nature of dark energy is an open question in cosmology. **Insert stuff about how we need to continue working on this to see if there are any other factors which might contribute etc**