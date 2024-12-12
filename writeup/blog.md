---
title: "Straightening Flows"
pubDate: 2024-12-11
description: "Blog about straightening flows for image generation."
author: "Timothy Qian and Evan Kim"
tags: ["diffusion", "image-generation", "flow-matching"]
---

## Overview

We provide a brief overview of diffusion and flow matching and their relation. As elucidated in [citation], they are much more closely related than most think. As a continuation of this, we explore the procedure of “Reflow” on diffusion models, which is meant to speed up generation by straightening the flows.

We’ll begin this blog post with a discussion of Diffusion models and Flow Matching models and show that they are just two different formulations. It is well known that you can train a diffusion trajectory with the flow matching objective if you choose the right scheduling, but it turns out that after a bit of reparametrization, it’s actually possible to initialize a Flow Matching model from a diffusion model and vice-versa (without any additional training!)\footnote{at least, when the latent distribution is a gaussian}. Following this, we’ll detail our experiments on “Half-Straightening”, where we test out how running reflow on different parts of the trajectory affects sample quality and sample speed.

### Diffusion

Diffusion models are a class of generative models that have gained attention for their ability to generate high-quality images, audio, and other types of data. Diffusion models are inspired by the idea of gradually transforming a simpler distribution into a more complex one through Gaussian noise.

Diffusion models have a forward and a backward process. The forward process is given by

$$z_t = \alpha_t x_t + \sigma_t\epsilon_t, \text{where } \epsilon\sim \mathcal N(0, I).$$

$\alpha_t, \sigma_t$ are the noise schedule of the diffusion process. A variance-preserving noise schedule satisfies $\alpha_t^2 + \sigma_t^2$ = 1. $z_1$ has a distribution similar to clean data, $z_0$ has a distribution similar to Gaussian noise.

### Flow Matching

Flow matching models have gained traction recently for their relatively simple formulation, flexibility to multiple latent distributions, and their promise of fast inference. On the engineering side, they’ve proven to be much more scalable, with models like Stable Diffusion 3 and Flux providing tremendous results.

#### Continuous Normalizing Flows

We define a _probability flow path_ as a function $p_t(x) : [0,1] \times \mathbb R^d \rightarrow \mathbb R_{\ge  0}$ -- a probability density function over the vector space $\mathbb R^d$ which evolves over time $t \in [0,1]$. The idea is that we want to model a mapping from some easy-to-sample distribution $\pi_0$ (usually a multivariate gaussian) to a desired distribution $\pi_1$ (the probability distribution of plausible images). Now, to model this probability flow path, we use a _flow_, which is a time-dependent map $\phi : [0,1] \times \mathbb R^d \rightarrow \mathbb R^d$. That is, if u have a sample $x_0 \sim \pi_0$, then you can determine a sample from the probability distribution $p_t$ by taking

$$x_t = \phi_t(x_0).$$

The two are related by

$$p_t(x) =  p_0(\phi_t^{-1} (x) \det \left[\frac{\partial \phi_t^{-1} (x)}{\partial x}\right],$$

The jacobian term is there to properly scale the mapping. Now, the last remaining question is, how do we represent the function $\phi$? Of course, we could just try to learn the function $\phi_t(x)$, but it turns out in many cases it's much easier to learn its derivative -- analogous to adding noise levels in diffusion. In this case, we are now learning a velocity field $v : [0,1] \times \mathbb R^d \rightarrow \mathbb R^d$, which is related to $\phi$ by

$$\frac{\partial \phi_t(x)}{\partial t} = v_t(x_t), x_t = \phi_t(x).$$

Here we can already see the direct connections to diffusion models which predict noise (an offset) from a given sample and timestep.

#### Flow Matching Objective

Flow matching is a generalized framework with a simple foundation. Building off of the velocity representation of flows, we create a learning objective that directly matches these velocities,

$$\mathcal L_{\text{FM}}(\theta) = \mathbb E_{t, p_t(x)}\| v_{\theta , t}(x) - u_t(x)\|^2.$$

#### Connecting Flow Matching to Diffusion

Include the derivation in Lee et. al appendix

## Sampling

## Training

## Results

## References
