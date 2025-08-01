---
title: 'Kinematic Correlations of Charmed Hadrons'
date: 2025-07-24T18:47:40-06:00
author: ["Dev Desai"]
draft: false
categories: ["project"]
tags: ["pythia", "python", "particle physics"]
showtoc: true
math: true
---

In my junior year, I took an independent research course under Dr. Deepa Thomas whose work specializes in High Energy Physics and Quark Gluon Plasma. A common research method in this field is to use a Monte Carlo event generator, such as `Pythia`, to simulate particle collision data at extremely high energy. This post will cover my analysis of the kinematic correlations of charmed hadrons. If you haven't checked out my crash course on particle physics, check it out [here]({{< ref "posts/particle_physics/index.md" >}}), as this post assumes a basic knowledge.

---

### Motivation
Charm quarks are only created during high energy collisions, and are almost exclusively formed during the inital hard scattering, helping us understand QCD dynamics. But, charm quarks are difficult to observe directly due to **color confinement?**, so we want to study charmed hadron, or hadrons which contain at least one charm quark, in hopes to understand more about the production mechanisms behind charm quarks.


