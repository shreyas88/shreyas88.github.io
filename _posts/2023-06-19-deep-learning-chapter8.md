---
title: Deep learning book(Goodfellow) Chapter 8 Optimization
date: 2023-06-19 00:00:00 Z
layout: post
---

## Empirical risk minimization

Optimization problem formulation:
* Empirical minimization : Loss function expectation approximated as average of loss over batched samples(empirical loss)
Surrogate loss
* 1-0 loss not differentiable, surrogate loss function log likelihood minimizes expected classification error

## Gradient for log likelihood 

$$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})=\mathbb{E}_{\mathbf{x}, \mathrm{y} \sim \hat{p}_{\mathrm{data}}} \nabla_{\boldsymbol{\theta}} \log p_{\text {model }}(\boldsymbol{x}, y ; \boldsymbol{\theta})$$

Practical considerations of batch size
* Can be estimated empirically using batched data, however standard error is proportional to $$ \alpha / \sqrt(n) $$ hence increasing the number of samples has a sublinear effect on the error compared to the computation.
* Small batch also has a regularization effect
* Batch size is usually scaled up to better utilize multi GPU/CPU. Batch size however is limited by memory available 
Accurate estimation of gradient requires independently drawn samples in the minibatch - consider using shuffling if data loading process has an inherent order


## Challenges in Neural Network Optimization

### Ill-Conditioning
	$$f\left(\boldsymbol{x}^{(0)}\right)-\epsilon \boldsymbol{g}^{\top} \boldsymbol{g}+\frac{1}{2} \epsilon^2 \boldsymbol{g}^{\top} \boldsymbol{H} \boldsymbol{g}$$

* Loss function approximated by taylor series expansion. When the third time dominates, moving in the direction can actually increase the loss function
* General problem with optimization


### Local minima
* Non convexity imposes challenges since the model might not converge on a global minima
* Neural networks tend to have a large number of local minimas. Due to Model identifiability problem, generally due to weight space symmetry(hidden inputs rearrangement), relu(scale input weights by $$\alpha$$ and outputs by $$1/\alpha$$ have same outputs
* Local minima only a problem if they are significantly degraded performance compared to the global minima.
* Consensus in practitioner community today is that local minima is not a problem for neural networks since they empirically are able to achieve low cost value. It is possible to rule out local minima if the gradient norm is not decreasing over training epochs

### Plateaus, Saddle Points and Other Flat Regions

* For high dimensional non-convex functions critical points are more likely to be saddle points.
* Motivated from probability standpoint since saddle point requires second order derivative in different eigenvalues to be mixed sign(positive and sign) rather than all positive(minima) or all negative(maxima), the expected ratio of the number of saddle points to local minima grows exponentially with n 
* Empirically gradient descent is shown to escape saddle points

### Cliffs and exploding gradients
Gradients can explode near cliff structures and gradient clipping can prevent this from happening

### Long term dependencies
* Deep computational graphs with repeated params like recurrent neural networks can run into problems for eg eigendecomposition
$$\boldsymbol{W}=\boldsymbol{V} \operatorname{diag}(\boldsymbol{\lambda}) \boldsymbol{V}^{-1}$$
	$$W^t$$ will cause eigenvalues < 1 to vanish and > 1 to explode.
* Vanishing gradients can halt the optimization steps while exploding gradients can cause instability in the optimization process

### Inexact gradients
Sampling techniques can give noisy or biased estimate of gradient 
