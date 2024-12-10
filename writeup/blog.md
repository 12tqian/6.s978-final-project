---
title: 'Straightening Flows'
pubDate: 2024-12-11
description: 'Blog about straightening flows for image generation.'
author: 'Timothy Qian and Evan Kim'
tags: ['diffusion', 'image-generation', 'flow-matching']
---

## Overview

We provide a brief overview of diffusion and flow matching

### Diffusion

Diffusion models are a class of generative models that have gained attention for their ability to generate high-quality images, audio, and other types of data. Diffusion models are inspirred by the idea of gradually transforming a simpler distribution into a more complex one through Gaussian noise. 

Diffusion models have a forward and a backward process. The forward process is given by 

$$z_t = \alpha_t x_t + \sigma_t\epsilon_t, \text{where } \epsilon\sim \mathcal N(0, I).$$

$\alpha_t, \sigma_t$ are the noise schedule of the diffusion process. A variance-preserving noise schedule satisfies $\alpha_t^2 + \sigma_t^2$ = 1. $z_1$ has a distribution similar to clean data, $z_0$ has a distribtuion similar to Gaussian noise. 



### Flow Matching


## Sampling

## Training


## References
