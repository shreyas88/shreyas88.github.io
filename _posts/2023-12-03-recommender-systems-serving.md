---
title: Serving latency considerations for embedding recommender systems
date: 2023-12-03 00:00:00 Z
layout: post
---

Modern recommendation deep learning models have below main components 
- Embedding ID tables
- MLP and other interaction/dense layers.

When reporting the model parameters usually the number of parameters are combined together however since they have different performance characteristics, it is important to consider the parameter increase separately. Memory and parameter increase in MLP/dense layers would impact performance differently than a corresponding increase in embedding table size

We will assume the inference takes place on the CPU. We also assume that we are operating in the large model regime where the model resides in DRAM. This typically starts to be true once we exceed CPU cache capacity which is roughly of the order of ~100MB’s 


## Summary
Below points summarize understanding of the system- 

- Embedding lookup is **O( #batchsize )** Note the lack of params in this equation, in general embedding lookups performance should be largely independent of the embedding table raw memory footprint - this should be largely the case except step function changes ie if the embedding table no longer fits in DRAM and goes out to disk. 

- MLP dense layer should scale however in proportion to **O( #param / #batchsize )**. In contrast to embedding parameters, notice that MLP dense layer performance degrades with the number of parameters. Additionally, increasing the batch size should help amortize the performance cost since parameters fetched from memory can be re-used across the batch-size which is in contrast to embedding lookups.

Modern recommender systems are generally dominated by embedding params, LLM on the other hand are dominated by MLP dense layers. Depending on the above factors, one might dominate over the other. 

## Horizontal vs vertical scaling

- Horizontal scaling is scaling the fleet by adding more host machines to the overall fleet
- Vertical scaling is keeping the number of hosts constant and instead scaling the host capacity ie buy more expensive grade hardware(memory, disk, CPU etc)

In general horizontal scaling is considered a preferred way of scaling and reduces cost of serving as it is relatively straightforward to allocate and free more machines as required incrementally. 

On the other hand, vertical scaling requires justifying buyer dedicated expensive hardware for serving and would drive up the cost of serving. However, there might be reasons for going with vertical scaling:
- Single host doesn’t provides a reasonable SLA for latency without scaling host vertically
- When model can no longer fit on a single host machine

## Breakdown

- Embedding tables(latency bound): Typically held in large cardinality array-like data structure and typically residing in DRAM due to large size(assume it is small enough to fit in DRAM memory which should be the case for current planned embeddings < 4GB) .
   - Note that access pattern for O(#batchsize * k) where k is a small constant depending on the use-case. We DO NOT read the entire embedding table and filter the required rows on each query which would kill your performance and instead read only the necessary ID’s from the lookup table stored in memory
   - Note that embedding lookup is expected to be independent of the number of rows in the embedding table, in general scaling embedding dimensions should be cheap.
- Hash table: The hash maps the entity ID to associative embedding structure ID. Performance characteristic should be similar to embedding table, the only difference is that it is a hash table so there would be a slight overhead in terms of how hash buckets are implemented(imbalance could cause extra data fetches) 
- MLP and crossing layers(bandwidth bound): Increase in dense layer params however including interactions etc will directly impact the latency since we would need to load the entire set of parameters on each serving request.
  - For instance, 1 inference request for a model, requires cycling through the entire MLP parameter set which implies we will be memory bound as opposed to compute bound(CPU >>> memory). The utilization can be improved if we have bigger batch sizes and we can amortize this memory cost across a large batch size
  - In general given memory bandwidth bound this could severely impact the performance — ie 100M params*4bytes/param @ 1GBps memory bandwidth requires ~1/2sec to simply load the weights into memory for each request which would obviously kill the performance. Based on the above extra MB’s added to MLP/dense layer is typically more expensive than equivalent increase in embedding table capacity in memory regime

Additionally memory pressure in embedding and dense layer scales differently as we increase the QPS/batch size
- MLP and dense layers(model parameters) memory transfer cost can be amortized by increasing the batch size subject to CPU vectorization/compute capacity 
- In general embedding scales linearly QPS ie higher QPS generally increases the data that needs to be transferred can’t be amortized over batch as the embedding ID will in general be distinct across the batch dimension
