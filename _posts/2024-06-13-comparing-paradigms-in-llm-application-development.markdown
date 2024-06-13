---
title: Comparing paradigms in LLM application development
date: 2024-06-13 23:07:00 Z
---

When it comes to LLM application development design patterns, how do you evaluate various strategies and come up with the best approach for your problem.  Note that some of these techniques are not exclusive and can be used in combination with each other. 


| Paradigm                       | Advantages                                                                                                                                                                                                                                                                                                      | Disadvantages                                                                                 |
|--------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| Prompt engineering             | 1. Give model few shot examples and clear/detailed instructions<br>2. Usually the first technique to try out, great for building problem intuition<br>3. Low engineering cost, fast iteration loops<br>4. Break down the problem down into steps e.g. Chain of thoughts, react framework                        | 1. Prompt tokens are expensive, feeding more into the context is expensive and may not make sense<br>2. Not effective in learning new output structure, formatting, programming language etc                        |
|                       |    |   |                                                                                                                                                                                                                                                                                                                                                                               |                       |    |   |     
|                       |    |   |             
| Retrieval augmented Generation aka RAG | 1. Currently the only scalable way to ingest new knowledge from a bunch of documents given a pretrained model<br>2. Getting the context aka "what the model needs to know"<br>3. Interpretability: It's straightforward to implement interpretability since model answers based on the retrieved context<br>4. Decomposing the problem into generation and retrieval can lead to independent iteration | 1. Medium engineering cost i.e. requires tuning<br>2. RAG pipeline can be expensive to maintain<br>3. Multi-step evaluation is needed (generation & retrieval/ranking) |
|                       |    |   |         
|                       |    |   |         
|                       |    |   |         
|                       |    |   |         
| Fine tuning                    | 1. Generally specifying how the model needs to behave i.e. domain style, formatting etc<br>2. Effective at teaching model output structure, new programming language, syntax etc<br>3. More efficient during inference since model can follow instructions and style better (reduced number of context tokens)<br>4. Reduced reliance on extensive prompt tuning | 1. Does not teach the model new knowledge, only emphasizes knowledge that already exists in pretraining<br>2. Supervised dataset creation can be expensive<br>3. Infra cost, hyperparameter tuning cost             |
| Agents                         | 1. Ability to solve more complex problems problems requiring long range context and real world interaction<br>2. Planning and breaking down complex tasks into simpler tasks<br>3.Using complementary LLM’s strength eg use code generation LLM  + general purpose LLM + LLM fine tuned for tool use<br>4.Incorporate multi turn human feedback into the loop and switch between human dialogue and tool use.                                                                                                                                                                                                                                                                   | 1. Reliability with current generation of LLM is still an issue 2. Context management and pollution is an issue and a hard problem |

References: 
[A Survey of Techniques for Maximizing LLM Performance
](https://www.youtube.com/watch?v=ahnGLM-RC1Y&ab_channel=OpenAI)<br>
[Mastering LLMs: A Conference For Developers & Data Scientists](https://maven.com/parlance-labs/fine-tuning)