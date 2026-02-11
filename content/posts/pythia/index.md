---
title: 'Kinematic Correlations of Charmed Hadrons'
date: 2025-07-24
author: ["Dev Desai"]
draft: false
categories: ["projects"]
tags: ["particle physics"]
showtoc: true
math: true
---

*A full repository of this project can be found [here](https://github.com/devddesai/Pythia-Studies).* 

In my junior year, I took an independent research course under Dr. Deepa Thomas whose work specializes in High Energy Physics and Quark Gluon Plasma. A common research method in this field is to use a Monte Carlo event generator, such as `Pythia`, to simulate particle collision data at extremely high energy. This post will cover my analysis of the kinematic correlations of charmed hadrons. *This post assumes a basic knowledge of experimental particle physics. I will soon post a crash course blog on this topic!*

<!-- If you haven't checked out my crash course on particle physics, check it out _link_ as this post assumes a basic knowledge. -->

---

### Motivation
Charm quarks are only created during high energy collisions, and are almost exclusively formed during the inital hard scattering, helping us understand QCD/QGP dynamics. But, charm quarks are difficult to observe directly due to color confinement, and they hadronize rapidly. Hence, I will focus my analysis on charm hadrons, or hadrons which contain at least one charm quark, in hopes to understand more about their underlying production mechanisms.

# Setup

In this study, I utilized `pythia` to simulate particle collisions. Some key notes about `pythia`:

- Models the full evolution of an event:
    1. Hard scattering: primary parton interaction (e.g., charm quark
    production).
    2. Parton showering: emission of gluons and quarks (initial/final
    state radiation).
    3. Hadronization: transition of partons into observable hadrons. • Decay: of unstable hadrons into final-state particles.
- Highly configurable: allows tuning of energy, cuts (like pTHatMin), and filtering for specific processes or particles.

We will focus on the final state particles when doing our final analysis, as in the real world, we do not have access to the full lifetime of the particles, but rather, just what hits the detector. But, we can use each simulated particle's lifetime to understand what we see in the final state.

### Data Structure

In order to do a robust study on charmed hadrons, we need a large dataset of these particles, which can quickly grow in filesize. In newer versions of pythia, there is not a simple method to write the full details of a simulation into a file. `hepMC` is a universal data structure which has many formats such as binary and ASCII, but there are quite a few dependency issues with more efficient formats such as binary, and using ASCII or txt formats can bloat a file. 

We can use `ROOT`, CERN's particle physics framework - not only will I utilize this for my analysis later on, but it also contains a `TParticle` class, optimized to store the lifetime of a particle, and a `Tree` class which is optimized for storing `ROOT` objects. By creating a custom data structure using these classes, we are able to reduce filesize by ~20%, which is useful as our dataset later on is ~ 1-10 GB. This also allows for easy reproducability with low dependency issues.

When I created this ROOT data strcuture, I wanted to test that it was indeed storing the same information as the hepMC methods. I found 2 discrepancies: my structure had a very large number of PDG 90 counts. (PDG number is a unique identifier for different types of particles, but there is no such thing as PDG 90). This led me to believe that PDG 90 is some internal pythia object that can safely be ignored. Furthermore, in the hepMC class, I found an abnormally high number of PDG 21 (gluons), which wasn't present in the ROOT structure. I believe that these two discrepancies are correlated, as gluons are sort of an intermediate particle that aren't physically detected. This also highlighted an important note: pythia is not accurate to the intermediate quark/gluon level.

The solution to this is to simply filter out these unphysical PDG 90 artifacts and gluons, as they won't be needed in the analysis anyways. After this filtering, both data structures were consistent!

### Phase Space Cuts - Maximize Charm Quark Yield

Again, we need a dataset of particle collisions rich with charmed quarks. Yet, there only ~3.3% of proton-proton collisions contain charm quarks; this is because they are heavy flavor particles which require a high energy collision. This only occurs if the protons collide more head on with each other and have a certain energy/momentum transfer. The histogram below shows this phenomena:

{{< figure src="./images/pt0.png" width="600px" align="center" caption="A histogram visualizing the number of charm hadrons in an event/collision.">}} 

In order to get a charm quark rich dataset, one could just simulate collisions and throw them out if they don't contain a charm quark. This wastes 97% of compute though - instead, pythia has a parameter called `pThatMin`, which essentially cuts the simulation if the initial transverse momentum transfer is not high enough. If we choose a cutoff too high, this also leads to unnecessarily slow simulations, so let's find a good balance.

I ran small scale simulations with different phase space cuts to optimize charm quark yield and compute time. Here are the findings:

{{< figure src="./images/charmperevent.png" width="600px" align="center" caption="A histogram visualizing the number of charm hadrons in an event given different phase space cutoffs.">}} 

A clear relationship is seen - as we increase the cutoff, there is a higher yield of charm quarks. But again, choosing a value too high may be overkill. 

{{< figure src="./images/ptsummary.png" width="600px" align="center" caption="A histogram visualizing the average number of charm hadrons in an event given different phase space cutoffs.">}} 

Here, we can see that around 10 GeV/c, the charm quark yield starts to plateau, so I will use this for all future work. Lastly, we can also look at a similar analysis with D0 mesons, as these are the most popular charm quarks - we see consistent results with this.

{{< figure src="./images/ptD.png" width="600px" align="center" caption="A histogram visualizing the average number of D mesons in an event given different phase space cutoffs. Results are consistent with all charm quarks">}} 


Next, I simulated 10,000 events for this analysis, although ~1 million events would be ideal. I simply didn't have the compute or storage at the time for such a high-scale analysis.

# Charm Multiplicity Analysis

The first thing I did was to analyze the multiplicity of charm quarks per event. I plotted a histogram of this, as below.

{{< figure src="./images/mult1.png" width="600px" align="center" caption="A histogram visualizing the average number of charm quarks produced in each event">}} 

We see something odd here: according to literature, charm quarks, and thereby hadrons, are most commonly produced in pairs. But, we see that there are more events with 3 charm hadrons than 2! To understand this, let's refresh on the charm quark production processes:

> **Pair Production:** $gg → cc \text{ or } qq → cc$
> - Produces a pair charm quarks → 2 charm hadrons (leading to
> even count)
> - Is the most common production mechanism

> **Flavor Excitation:** $gc → gc \text{ or } qc → qc$
> - Produces 1 charm quark → 1 charm hadron (leading to odd
> count)
> - Rarely happens

> **Gluon Splitting:** $g → cc$
> - Produces a pair charm quarks → 2 charm hadrons (leading to even count)

According to these three production mechanisms, the data should be comprised of mostly even numbered charm quark events. The issue lies with a specific decay process of the D* meson. D* mesons are a higher energy state of the D meson. These usually decay into more stable particles before it hits the detector. Hence, if we count all charmed hadrons, we might be double counting a charm quark as it is present in both a D* meson and its decayed meson. The solution to this is to filter out all non-final state particles, as these won't decay into other particles.


{{< figure src="./images/Dstardecay.png" width="600px" align="center" caption="A histogram visualizing the average number of final-state charm quarks produced in each event">}} 

We can now clearly see that events with an odd count of charmed hadrons are much less likely than their even counterparts. Let's integrate this and normalize it to convert it into a PDF.

{{< figure src="./images/charmPDF.png" width="600px" align="center" caption="A log-scale PDF of final-state charm hadron multiplicity.">}} 

Here, I had a bin size of 2, so that instead of seeing the seasonal even-odd pattern, we can see that there is a decaying exponential relation between the number of hadrons and their probability. One might also ask, why not count the total number of quark quarks instead of looking at the hadrons? Well first, it is good to base our analysis on the hadrons, as we can then compare that analysis to real accelerator data. Secondly, it leads to massive double counting similar to with the D* phenomena. We would be counting the same charm quark in every step of its lifetime. Just to visualize this, I counted the number of charm quarks into a histogram:

{{< figure src="./images/quarkmult.png" width="600px" align="center" caption="A histogram of charm multiplicity.">}} 

# Kinematic Correlations

In order to extract meaningful relationships in the data, I decided to only use events which had exactly 2 final state charm hadrons, as these should be correlated according to the three charm quark production processes as mentioned above. This section will have a lot of graphs, so please bear with me!

## Two Final-State Charm Hadrons Correlations

### Transverse Momentum (pT)

{{< figure src="./images/transverse.png" width="600px" align="center">}} 

Through this figure (log-scale), we can see that most charm hadrons have a pT around 2-3 GeV/c. This makes sense, as that means on average, the two charm quarks contain around 50% of all the minimum transverse moementum (due to our phase space cutoff at 10 GeV/c). We can also see that there is a sharp cutoff as we get to higher pT values - this is due to the rarity of these events as well as the cutoff at 10 GeV/c. I wouldn't bet that if we put the phase space cut to 20 GeV, we would see a higher count of charm hadrons with higher pT though - that extra momentum might just go into created even heavier flavor quarks such as the beauty quark.

### Rapidity (y) and Pseudorapidity ($\eta$)

{{< figure src="./images/rapidity.png" width="600px" align="center">}} 

{{< figure src="./images/eta.png" width="600px" align="center">}} 

We can see that both these figures look very similar, as expected due to their definitions. These distributions relate to the polar angle with respect to the beamline
- On average, the particles are centered around the pT plane with y = 0. (Polar angle of 90)
- Fewer particles are going in the direction of the beamline (very positive or negative values)
- There is a dip in $\eta$ at 0 due to the way it is calculated (from the $\text{cos}( \theta)$)

### Phi ($\phi$)

{{< figure src="./images/phi.png" width="600px" align="center">}} 

The random spread indicates that there is no bias towards a specific direction in the pT plane.

### Delta-Phi ($\Delta \phi$)

{{< figure src="./images/dphi2.png" width="600px" align="center">}} 

$\Delta \phi$ is the difference between the two azimuthal angles of the charm hadrons. Although we don't expect a bias, or pattern, in the direction of $\phi$, we do expect there to be strong correlations between the two hadrons, especially because we only picked events which have 2 charmed hadrons which should be correlated. We expect a peak at 0 and a broad peak on the away side. The two hadrons should be going opposite from each other. Let's explore this further and see if we can reconcile this further! 

## Kinematic Correlations With All Particles

I first experimented by doing this same exact analysis with all events instead of only choosing those which contained two final-state charm hadrons. I won't put all the plots here, because unfortunately, I got the exact same behavior for all the correlations! Everything except the $\Delta \phi$ made sense.

One interesting I tried though, is to use D0 mesons as triggers, and then plot the $\Delta \phi$ between those D0 and ALL other particles. For some reason, this yielded the behavior I expected the charmed hadrons to have. 

{{< figure src="./images/dphi all.png" width="600px" align="center">}} 

Here, we can clearly see that there is a correlation at 0 and $\pi$. The reason we expect these peaks:

> **Sharp peak at 0** $\to$ in gluon splitting, there is 1 gluon which splits into 2 charm quarks. These quarks are close to collinear, leading to front to front peak being sharp
>
> **Broad peak at $\pi$** $\to$ in pair production, there is 2 gluons which become 2 charm quarks. These quarks are opposite in phi, leading to back to back correlations. Due to fragmentation and other interactions, the peak smears.



Notice that the y axis starts at 185 counts, which tells us that there is a lot of backgorund noise which comprises that 185 count, but there is a small signal - which I would expect to be the charmed hadrons. But, when we isolated the charm hadrons, we didn't see that signal. So it must be coming from something else!

---

Unfortunately, my semester project had concluded here for PHY 371C - at a cliffhanger! I hope to one day return to these findings and understand where this discrepancy comes from. If you have any ideas, please feel free to reach out to me!