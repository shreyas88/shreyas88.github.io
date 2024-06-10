---
title: Speculative Decoding
date: 2024-06-10 22:16:00 Z
---


Preliminaries

LLM inference is a serialized process where each new token is generated conditioned on the previously accumulated tokens. This implies that n iterations of LLM inference requires n steps and these steps can’t be parallelized as the input of the current step depends on the output tokens from the previous step. 

Token generation step consists of sampling from a probability distribution over the set of all possible tokens in the vocabulary. Normally this part is abstracted from the end user and we observe the generated tokens directly but for this discussion we would be operating on the probability distribution over tokens hence it is important to get the notation.


Motivation

Observation1: The LLM inference process is inherently bottlenecked by the memory due to the auto regressive (generate one token at a time) nature. In simple terms it means that the wallclock time is dominated by data transfers(model weights, kv cache) as opposed to performing the actual matrix multiplies on GPU. This implies we can perform additional parallel computations on GPU per memory access without impacting the wallclock time.  If you want to understand this tradeoff further please refer to this fantastic blog https://horace.io/brrr_intro.html , additionally for deeper understanding for LLM inference characteristics 

Observation 2: Some tokens are easier to predict for the LLM than other tokens. Eg for code generation maybe curly braces after if statement, generation of stop words, conjunctions and other easier to predict words. In theory, it should be possible for a much smaller model to predict those easier tokens and offload some computation from a larger model.

Speculative decoding technique exploits the above observations to enable inference speedup. It consists of using a faster, smaller approximate model(M_q) to predict K lookahead tokens in parallel to the main larger, slower baseline model(M_p). The name of the technique comes from these lookahead tokens which are speculative in nature i.e. can be rejected if they don’t match the verification step and generation restarts from the previously accepted point. The idea is inspired by speculative execution which is commonly employed in modern CPU processors where the processor is typically predicting branches speculatively to better overlap computation and memory access. 

Overview

The basic idea is that in the amount of time it takes to generate a single auto-regressive token on the larger model M_p, we can generate multiple such tokens on the smaller model M_q(draft model). We then “check” these generated lookahead tokens and only accept them if it matches some criteria. 
Note that it takes the same amount of time to generate 1 additional token and K fast forwarded additional tokens from the base model in parallel due to the memory bound nature of LLM inference as discussed earlier.  In this step we effectively fast forward the baseline model on the speculative tokens generated earlier i.e. {token1}, {token1, token2}, {token1, token2, token3} , … etc  to enable the checking against the draft tokens.

Details
![Screenshot 2024-06-10 at 3.20.15 PM.png](/uploads/Screenshot%202024-06-10%20at%203.20.15%E2%80%AFPM.png)

Let's break it down step wise

If q(x) <= p(x) then we accept the token since the base model is more likely to generate this token and generally the speculative token stream emitted from the draft model is aligned with the base model token stream

If q(x) > p(x) 
In this case we roughly want to reject tokens based on deviation/error roughly speaking if q(x) is only slightly higher than p(x) then we should probably accept the token since the error is fairly low. On the other extreme if p(x) = 0 i.e. base model doesn’t emit the token x,  then we want to reject this token since the spec token stream is misaligned with the base token stream. 

This can be accomplished if we sample probabilistically (q(x)-p(x))/q(x)


In a single speculative decoding iteration, we have the inputs previously generated tokens called prefix tokens, base model and draft model. 

We generate tokens k speculative tokens from the draft model spec_tokens
In parallel, 
Consider the k sequences - prefix + spec_tokens[:i]
For each of the sequences, we extract the corresponding probability distribution from the base model by prefix + spec_tokens[:i]. This corresponds to one inference generation for the base model.
    -  We apply the accept/reject criterias discussed in the above section and find the first of the k token which is rejected(or none if all are accepted)
    - From the rejected token, we append back the tokens upto the rejected token to the input prefix tokens for next epoch of speculative decoding. 
- Additionally, in order to ensure forward progress for each epoch(in case we fail to get any token accepted) we generate one additional sample from the adjusted probability distribution  p'(x) = \text{norm}(\max(0, p_{n+1}(x) - q_{n+1}(x)))

Very roughly speaking, algorithmically we are trying to sample a new frontier of the token space which lies further away from the frontier of the draft token space. This is done by upweighting the probability mass of tokens in vocab space which are more likely to be generated by the base model when compared to the support draft probability distribution.


https://github.com/vllm-project/vllm/pull/2607
