---
title: Work in progress Notes CIFAR10 resnet exploration
date: 2023-02-04 00:00:00 Z
layout: post
---

Study notes blog post are based on the blog post [How to train your resnet series](https://myrtle.ai/learn/how-to-train-your-resnet-5-hyperparameters/)


## Learning rate schedule
![learn-rate]({{ site.baseurl }}/images/learning-rate-cifar10.png "Learning rate schedule(picewise)")

When we try to increase the learning rate, it could lead to degradation in two ways
* Increasing the learning rate could destabilize the training due to second order curvature effects beginning to dominate. This is because in presence of higher learning rate, we are more likely to experience the destabilizing effect of a larger step size since first order optimization techniques like gradient descent do not take into account the second order gradient effects. In the literature it is described as ill-conditioning of the optimization problem.
* Higher learning rates could lead to the phenomena of catastrophic forgetting. Although this is more common in the context of the multiple task setting, it could also happen for a single task setting. Intuitively, if the learning rate is high, the model might “forget” the batches it saw early on.

## Tuning learning rate and batch size

![batch-size]({{ site.baseurl }}/images/batch-size-cifar10.png "Loss vs learning rate for batch sizes")

The above graph shows:
* The optimal validation loss occurs at higher learning rates as we increase the batch size ie validation loss is minimized at learning factor=1 at batch size = 128 and 4 at batch size=512
* This follows the general advice behind the linear scaling relationship between batch size and learning rate ie increasing batch size by k we should also be scaling the learning rate by k. 
* As a side note, the above linear scaling relationship is mentioned in the paper [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677). In a nutshell, the author exploited this linear scaling phenomena to achieve linear speedups by using distributed training in the multi-GPU setting and preserving model convergence by linearly scaling the batch size and learning rate relationship. 

We can see from the below graph that the learning rate has a drastic effect on the forgetting phenomenon. When learning rate is very high, we get a comparable validation loss with 1) half dataset configuration + no augmentations and 2) full dataset + augmentations which is unexpected 

![learn-rate-half]({{ site.baseurl }}/images/learn-rate-half-dataset.png "Val/train Loss half dataset with learn rate scale=1 vs epoch")
![learn-rate-half]({{ site.baseurl }}/images/learn-rate-full.png "Val/train Loss full dataset with learn rate scale=1 vs epoch")


## Weight decay in presence of batch normalization


How to interpret Weight decay in presence of the batch normalization?

Weight decay is used as a regularization technique to lower the weight norm and prevent overfitting. However, in the example case where the convolution layer is followed by the batch normalization layer, weights are rescaled by the batch norm layer. In this case, loss function is independent of the weight norm. However, the blog post linked shows why weight decay serves an important function in the optimization process. 

Essentially the weight update equation can be split into a weight decay portion and weight update through gradient descent step. Imagine if we rescale and increase the weights by 2x, the effective gradient will be 2x smaller. Additionally, in the 2x scaled regime, in order to maintain a similar optimization trajectory the gradient needs to be increased by 2x. 

In presence of batch normalization the loss function is scale invariant wrt to the weights
As a qualitative argument, weight decay acts as a control mechanism in presence of batch norm and maintains the weights:gradient ratio.   
