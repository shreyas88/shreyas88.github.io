---
title: Optimizers
date: 2023-06-15 00:00:00 Z
layout: post
---

## Adagrad

https://d2l.ai/chapter_optimization/adagrad.html

* Sparsity issue: The gradients in different directions can be updated at different rates depending on the gradient vector. For example, we could have a direction where the gradient magnitude is very large and we are making great progress, while another direction where we are making slow progress(feature sparse). The adagrad algorithm tries to adjust for this effect by scaling the learning rate by gradient squared in that specific direction

* Ill-conditioning : Ill conditioned problems are where optimization number(ratio of largest eigen-value/smallest eigen value) is very large. This makes optimization difficult because the eigenvalue space is distorted and it is difficult to converge with same learning rates. It turns out that by dividing by gradients magnitude is a good way to scale the eigenspace as it is a good proxy for hessian diagonals

* Adagrad decreases the learning rate dynamically on a per-coordinate basis.
* It uses the magnitude of the gradient as a means of adjusting how quickly progress is achieved - coordinates with large gradients are compensated with a smaller learning rate.
* Computing the exact second derivative is typically infeasible in deep learning problems due to memory and computational constraints. The gradient can be a useful proxy.
* If the optimization problem has a rather uneven structure Adagrad can help mitigate the distortion.
* Adagrad is particularly effective for sparse features where the learning rate needs to decrease more slowly for infrequently occurring terms.
* On deep learning problems Adagrad can sometimes be too aggressive in reducing learning rates. We will discuss strategies for mitigating this in the context of Section 12.10.

## RMSProp

* This is a modified version of adagrad, the problem with the adagrad being that the square gradient term can become very large over optimization iterations which can make convergence slower. RMSprop accounts for this by using the exponential moving average of square gradients so it forgets the less recent terms in the gradient square calculation.
* RMSProp is very similar to Adagrad insofar as both use the square of the gradient to scale coefficients.
* RMSProp shares with momentum the leaky averaging. However, RMSProp uses the technique to adjust the coefficient-wise preconditioner.
* The learning rate needs to be scheduled by the experimenter in practice.
* The coefficient γ determines how long the history is when adjusting the per-coordinate scale.

## Adam

* In the Adam optimization algorithm, momentum and scale components play important roles in the update rule for the model's parameters.
* The momentum component helps to accelerate the optimization process by allowing the update to incorporate information from past gradients. Specifically, the momentum component adds a fraction of the previous update to the current update, which helps to smooth out the optimization process and prevent oscillations. This can be especially useful when the optimization surface is noisy or has uneven curvature.
* The scale component, on the other hand, helps to normalize the update by dividing the update by an estimate of the variance of the gradients. This helps to stabilize the optimization process and prevent the updates from becoming too large or too small.
