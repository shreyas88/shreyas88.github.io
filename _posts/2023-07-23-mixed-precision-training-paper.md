---
title: Mixed precision training
date: 2023-07-23 00:00:00 Z
layout: post
---

* "Mixed Precision Training" by [Micikevicius et al. (2017)](https://arxiv.org/pdf/1710.03740.pdf) presents a technique for training deep neural networks using a mix of lower-precision (e.g., float16) and higher-precision (e.g., float32) arithmetic. 
This approach accelerates training and reduces memory requirements without sacrificing model accuracy.
Key contributions of the paper include:
* Introducing a loss-scaling technique to prevent gradient underflow, which can occur due to the limited dynamic range of lower-precision data types.
* Demonstrating that using higher-precision weight updates and maintaining a master copy of weights in higher precision is crucial for maintaining training stability and accuracy.
* Showing that certain operations, such as layer normalization, should be performed in higher precision to avoid numerical instability.
* Providing empirical evidence that mixed precision training can achieve the same accuracy as full-precision training across a variety of deep learning tasks, including image classification, speech recognition, and neural machine translation.
* Establishing that mixed precision training leads to significant speed improvements and memory savings on GPUs, specifically NVIDIA's Volta architecture, which supports hardware-accelerated mixed precision arithmetic.
*  Great way to see this in action is the popular [Pytorch nano GPT implementation](https://github.com/karpathy/nanoGPT/blob/master/train.py) where GradScaler Pytorch uses similar techniques including loss-scaling(just follow scaler object)
