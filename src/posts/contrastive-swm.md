---
layout: post.njk
title: "Contrastive Learning of Structured World Models"
date: 2026-08-11
description: 'Analysis of the paper titled "Contrastive Learning of Structured World Models" and the C-SWM as a method for learning abstract state representations from observations alone.'    
code: "https://github.com/estiftole/c-swm"
banner: "/assets/images/post-images/c-swm simplified.png"
---

## Introduction
A world model is an internal representation of the environment that allows you to navigate and achieve goals. 

It is a crucial aspect of human cognition, and has inspired a wide array of works exploring how to build systems with this ability.  

Learning the right latent space is the most challenging part of building world models. Methods that rely on visual reconstruction as the primarily objective (e.g. variational autoencoders) run the risk of learning visual features that aren't relevant for abstract reasoning. Predicting the world at pixel-level detail is wasteful. 

Several solutions have been proposed to tackle this problem. The paper titled <a class="reference" href="https://arxiv.org/abs/1911.12247">Contrastive Learning of Structured World Models</a> is one such example. It introduces a class of models called Contrastively-trained Structured World Models (C-SWM) that learn abstract object-factorized state representations to model the environment. 

## Intuition
Every environment is made of objects. Therefore, the state of a given environment can be defined compositionally as the states of the objects within it.

The state $s$ of the environment at time $t$ can be expressed as:
$$s_t = \{s_t^{(1)}, s_t^{(2)}, \dots, s_t^{(K)}\}$$
Where $s_t^{(k)}$ is the state of object $k$ at $t$.  

Similarily, the latent representation of the state can be expressed as:
$$z_t = \{z_t^{(1)}, z_t^{(2)}, \dots, z_t^{(K)}\}$$

## C-SWM Architecture
Here's a simplified diagram of the C-SWM architecture:  

<img src="{{ '/assets/images/post-images/c-swm simplified.png' | url }}" alt="c-swm simplified">

The state $s_t$ is passed through an encoder module that returns an abstract state representation $z_t^k$ for each object. 

That abstract representation is then coupled with the action $a_t$ and passed to the transition model $T$; a model that predicts how an environment transitions from one state to another.
$$
\Delta z_t = T(z_t, a_t)
$$

The transition model returns a $\Delta z_t$ that represents the change in the state due to the action. The predicted abstract representation of the next state $s_{t+1}$ can be obtained by adding it to the current latent state $z_t$:
$$\hat{z}_{t+1} = z_t + \Delta z_t$$
The training objective is to make the predicted abstract state $\hat{z}_{t+1}$ as close as possible to the actual abstract state $z_{t+1}$ while pushing it away from negative samples $\tilde{z}_{t}$.

Therefore the loss used to train C-SWMs can be represented as:
$$
\mathcal{L} = d(z_t + \Delta z_t, z_{t+1}) + \max(0, \gamma-d(\tilde{z}_{t}, z_{t+1}))
$$
where $d(z_t + \Delta z_t, z_{t+1})$ is the difference between the predicted and actual next states, $d(\tilde{z}_{t}, z_{t+1})$ is the difference between a negative sample and actual next state, and $\gamma$ is a hyperparameter.

### **Encoder**
The encoder module is composed of a CNN-based model for object extraction $E_{ext}$ and an MLP-based object encoder $E_{enc}$.
The object extractor takes an image as input and returns $K$ feature maps where each feature map $m_t^k$ can be interpreted as belonging to one particular object. $E_{enc}$ then takes one feature map $m_t^k$ at a time and returns the corresponding abstract state representation $z_t^k$.  
### **Transition Model**
The transition model $T$ takes the latent state $z_t$ and action $a_t$, and returns the change in the state $\Delta z_t$:
$$
\Delta z_t = T(z_t, a_t) = GNN(\{(z_t, a_t)\}_{k=1}^K)
$$
Implementing the transition model as a GNN provides permutation invariance (i.e. the order of objects doesn't matter, which would be a problem when using a standard MLP), and is a natural choice when modeling the environment through local interactions between objects.  
### **GNN**
In the GNN, an edge function $f_{edge}$ computes directional messages (representing interactions) between every pair of object slots $i$ and $j$.
$$
e_t^{(i,j)} = f_{edge}\left(\left[z_t^i, z_t^j\right]\right)
$$
A node function $f_{node}$ then computes the predicted state delta of each object $i$ using its abstract state $z_t^i$, the action performed on it $a_t^i$, and an aggregate of its interactions with other objects $\sum_{i\neq j} e_t^{(i,j)}$:
$$
\Delta z_t^i = f_{node}\left(\left[z_t^i, a_t^i, \sum_{i\neq j} e_t^{(i,j)}\right]\right)
$$

Finally, the contrastive loss for multiple objects:
$$
\mathcal{L} = H + \max(0, \gamma-\tilde{H})
$$
where $H$ and $\tilde{H}$ are positive and negative sample losses respectively:
$$
H = \frac{1}{K}\overset{K}{\underset{k=1}{\sum}} d(z_t^k + T(z_t^k, a_t^k), z_{t+1}^k)
$$
$$
\tilde{H} = \frac{1}{K}\overset{K}{\underset{k=1}{\sum}} d( \tilde{z}_{t}^k, z_{t+1}^k)
$$

This loss function teaches the model to do three things:
1. discover objects only from observations,
2. compute relevant abstract representations of states, and
3. predict state transitions caused by actions.

 A more complete representation of the C-SWM architecture is in this image:
 
 <img src="{{ '/assets/images/post-images/c-swm architecture.jpg' | url }}" alt="c-swm architecture">
 <a class="small-reference">From the paper</a>
 
## Implementation
The authors published the <a class="reference" href="https://github.com/tkipf/c-swm">code of their implementation</a> along with their paper. Unfortunately, their implementation relies on now out-dated packages and functions (such as the `gym` package which is now `gymnasium`, with the atari games being branched off in another package `ale-py`), so I had to do some refactoring to get it to run in the modern ecosystem. 

You can also find <a class="reference" href="https://github.com/estiftole/c-swm">my updated implementation.</a>
## Evaluation
The authors of the paper evaluated this architecture on two grid world environments, two atari games (pong and space invaders), and 3-body physics simulation.

- **Hits@1 (Hits at Rank 1)**: The percentage of evaluation samples where the ground-truth target state ranks among the top $K$ nearest neighbors to the predicted state. $K$ here is 1, so it measures strict accuracy.  

- **MRR (Mean Reciprocal Rank)**: The average position of the true target state in the ranked list across the entire test set. The closer it is to 100, the higher the rank.

Compared to approaches that rely on a pixel-level reconstruction loss (AE/VAE) and approaches that don’t use object-factorization (Physics-As-Inverse-Graphics model), the authors found C-SWM-based models perform better on multi-step abstract prediction.

<img src="{{ '/assets/images/post-images/table1.jpeg' | url }}" alt="authors' table">
<a class="small-reference">Table from the paper</a>

For my evaluations, I compared the C-SWM to a version of the model that was trained on reconstruction loss with a decoder. Performance was compared on two Atari games (Breakout and Centipede). 

<img src="{{ '/assets/images/post-images/table2.jpeg' | url }}" alt="my table">

Results mimic that of the original paper, with the C-SWM model performing better than the reconstruction-based version.

## Limitations
The authors mentioned the architecture struggled with identifying multiple instances of the same object (instance disambiguation), and the fact that the formulation doesn’t take into account environments affected by randomness/stochasticity. 

The most significant limitation in my eyes is the fact that it requires manually defining the number of object slots. Performance seems to vary a lot with the number of $K$ slots you define when building the model.  

From the paper:
> "While for Space Invaders, a large number of object slots (K=5) appears to be beneficial, C-SWM achieves best results with only a single object slot in Atari Pong. This suggests that one should determine the optimal value of K based on performance on a validation set if it is not known a-priori." 

The authors describe a potential solution (iterative object encoding) to overcome that problem. It would allow the model to assign empty slots, so as long as we provide the model with more slots than needed, it’ll be able to use the optimal number of slots by itself.

They left it for future work, and so I decided to do the same. But I’ll be sure to explore it in the future. I think it would be interesting to see if it would work.

## Conclusion and Relevance
2019 is ancient news in this field, but the ideas behind C-SWMs, and especially SWMs, are more relevant than ever. Model-based methods have several advantages over model-free ones, and they provide superior generalization to novel scenes. But the most exciting prospect in my opinion is that of **analogical reasoning**. 

Similar to states in SWMs, analogies are defined by their members. It may be possible to build a system that uses a SWM to draw analogies between scenarios that seem very different on the surface but share the same underlying dynamics. 

These systems might learn to spot patterns across all sorts of domains that were previously unseen; enabling accurate **far transfer**, and perhaps even a universal form of transfer. I think this is more than enough reason to further explore Structured World Models.
