---
title: Deep learning book(Goodfellow) Chapter 7 Regularization
date: 2023-06-19 00:00:00 Z
layout: post
---

* Sometimes these constraints and penalties are designed to encode specific kinds of prior knowledge. Other times, these constraints and penalties are designed to express a generic preference for a simpler model class in order to promote generalization.
* Sometimes penalties and constraints are necessary to make an underdetermined problem determined
* To understand the effect of applying the regularization on the overall optimization result. We use the quadratic taylor expansion around the optimal point $$ w^* $$

	$$ \tilde{w} = (H + \alpha * I)^{-1} H w^* $$

Applying eigendecomposition to hessian matrix 

$$ \tilde{w} = Q ( \Lambda  + \alpha * I)^ {-1} \Lambda Q^T w^* $$

From the above, the scale factor = $$ \lambda_{i} / \lambda_{i}+\alpha $$ we can observe that 
* eigenvectors whose eigenvalues >> alpha will not get impacted much while directions with low eigenvalues(low sensitivity to the gradients) will see a significant impact.
* Hence the regularization has a stronger shrinking effect towards the origin in the eigen-direction in which the cost function exhibit low variance.


L1 regularization sparsity and feature selection. 
* Regularization equivalent to bayesian maximum posterior inference (MAP) from a probabilistic perspective.
* L2 regularization equivalent to applying a gaussian prior, while L1 regularization equivalent to applying isotropic laplace distribution


## Regularization
* Helps in undetermined problems 
* Regularization also used for undetermined problems such as when $$ X X^T $$ matrix is singular due to collinearity in linear regression
* Even problems with non closed form solutions such as gradient descent with logistic regression can have undetermined solutions eg if $$W$$ weight acts as a decision boundary, then $$ 2*W $$ will also serve as a decision boundary and the gradient descent can continue multiplying the weights till it hits numerical overflow. Regularization helps convergence by enforcing a prior on weights


## Noise Robustness
* Injecting noise at weight level
* Makes the optimization favor more stable points in the landscape ie minima surrounded by flat regions so that small fluctuations have a minimal effect  on the optima
* Injecting noise at output node
* Label smoothing: eg softmax with k outputs can be $$ 1-\epsilon $$ and $$ \epsilon/(k-1) $$    [nice to read paper ]
* Semi supervised learning and multitask learning 
* Brief discussion about generalizations not noteworthy
* Early stopping
* We keep a running copy of the best set of weight parameters and update it whenever we encounter a lower validation loss. Implicit “hyperparam” selection of the number of training steps
* How to use the validation set data for training purpose? Use multi pass - for first pass holdout validation dataset for early stopping and then merge the validation dataset into training set in the second pass
* Could use the same number of fixed training steps or epoch in the second pass or just continue the first pass(with validation set added) training till some heuristic hit eg validation loss falls below training loss
* Equivalent effect to L2 regularization - assume gradient bounded the final weight param bounded to a certain volume space. 
* Early stopping number of iterations inversely proportional/equivalent to inverse of weight decay
* Parameter sparsity : convention L1 norm weight decay. Represental sparsity works by enforcing hidden layer activation norm penalty in the loss 

## Dropout
* Dropout provides an inexpensive approximation to training and evaluating a bagged ensemble of exponentially many neural networks
* Main difference between bagging and dropout is the param sharing which makes dropout computationally feasible
* Every time a mini batch example is loaded in memory, we sample a binary mask for every node in the network. 
Training minimizes $$ E_{\mu} J(\theta, \mu) $$ where $$ \mu $$ is the binary mask vector corresponding to each input and hidden vector
* Dropping hidden units(as opposed to input pixel values) is powerful and systematic way to destroy information as the hidden unit usually contain higher level semantic information
* Inference time weight scaling multiplication by dropout probability is shown to work empirically 
* Some approaches like dropout boosting show that dropout have independent merit due to the bagging effect beyond the noise robustness effect
* Dropout also has a biological interpretation in that hidden units must learn to adapt independent of other connecting neurons thus causing them to learn features that are good in many contexts
* Additive vs multiplicative noise: Dropout employs multiplicative noise which doesn’t allow for pathological solutions for noise robustness, refer to this great explanation 

## Adversarial training
* Small perturbations in data shouldn’t cause output discontinuities. Adversarial training enforces a local constancy prior and improves test performances 
* Primary cause of these issues is excessive linearity, If we change each input by $$ \epsilon $$, then a linear function with weights w change by as much as $$ epsilon * ||w||_{1} $$ which can be a very large amount if it is high dimensional.
* Common assumption is that class specific manifold are not connected and lie on separate manifolds thus perturbing the inputs shouldn’t lead to output class changes  
