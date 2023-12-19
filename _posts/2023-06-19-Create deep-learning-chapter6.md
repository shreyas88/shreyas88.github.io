---
title: Deep learning book(Goodfellow) Chapter 6
date: 2023-06-19 00:00:00 Z
layout: post
---

## Solving XOR Function Using Neural Networks

Understanding how neural networks operate to solve complex functions can provide great insight into their inner workings. The XOR function is a common example used in this context.

## Understanding the XOR Function

Consider a simple XOR function where x1=1, x2=1 which should yield 0. This scenario can't be solved using a linear transformation, posing a real challenge for basic computational models.

## Role of the Neural Network in XOR

A neural network maps the input to a hidden non-linear transform where h1=1, h2=0 would yield 1 and h1=1, h2=1 would yield 0. This outcome can now be described by the linear mapping layer.

## Implementing Cost Functions

Cost functions form a critical part of the training process. Generally, the most common approach is to use the Maximum Likelihood Estimator (MLE), which works by minimizing the cross entropy between the data generation empirical training distribution and the model distribution.


In mathematical terms, the cost function can be written as:

$$ J(\theta) = - E_{x,y\sim p(data)} \log{p_{model}(y|x)} $$


## Advantages of Using Log Likelihood

There are several pros to using the log likelihood approach:

There is no need to design a custom loss function; the loss function follows naturally from the prediction function. For linear regression, we can simply take the output to be a Gaussian distribution output. In this case, log likelihood simply gives us the mean square error loss.

The formula is:

$$ p_{model} (y|x) = \mathcal{N}(y; f(x;\theta), \mathcal{I}) $$

## Interaction of Cost Function and Output Unit

The interaction between the cost function and the output unit is particularly interesting. The exponential saturating feature in sigmoid is "undone" by using the log term in the log likelihood loss equation.

The equation is:

$$  \mathcal{J}(\theta) = Softplus((1-2y)z) $$

Where:

* softplus(x) = log(1+exp(x))
* z = w^T x + b 

The softplus function is a smoothed version of the max(x, 0) function whose gradient only saturates when x < 0.

The equation only becomes negative when we already have the correct answer, i.e., y=0 and z << 0 and vice versa. This characteristic can be crucial in understanding how the cost function and output unit interact in a neural network.



The log function can help undo the exponential in the output function commonly used for neural networks, which can assist with the vanishing gradient problem.
