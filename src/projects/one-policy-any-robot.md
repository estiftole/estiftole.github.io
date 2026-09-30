---
layout: post.njk
title: "Building a Universal Policy that Controls Any Robot"
date: 2026-09-15
code: "https://github.com/estiftole/cross-embodiment"
description: Exploration of cross-embodiment policies; models that can adapt to and effectively control unseen robot morphologies.
banner: "/assets/images/one-policy-any-robot/urma_banner.gif"
summary: "I trained cross-embodiment policies that learn to adapt to unseen robot bodies at test-time and control them effectively."
---

## Introduction
Simulated environments are great because they allow you to train robots to perform actions without having to train them in the real world, which can be expensive, time-consuming, and could easily damage the robot. Simulations provide safety and full control over the environment. 

<a class="reference" href="https://arxiv.org/abs/1703.06907">Domain randomization</a> is a standard component in training control policies today because it develops the robustness of models against variations in things like mass, friction, and/or actuator strength. However, many control policies today are trained in environments where the structure of the robot itself remains unchanged. That's fine if you want to train a specialized policy to control that *specific* robot. But it means you'll have to train a new policy from scratch for every new robot.

Instead, what if you trained a single policy that can adapt to any new robot? What if it could do so with little-to-no fine-tuning?

An embodiment-agnostic (cross-embodiment) policy would have superior sample efficiency over training a new policy from scratch, or even fine-tuning on data from the target environment. This idea has inspired many approaches and architectures over the years, and this is my attempt at describing the most interesting ones I've encountered.

## Early Experiments

(Skip to "APPROACHES" for actual cross-embodiment architectures.)

Before experimenting with advanced algorithms, I played around with simple environments. My goal was to see if I could train a policy that adapts to varying limb sizes. So, I started with training simple PPO models on the default `Cartpole`, `BipedalWalker`, and `Pusher` environments from the gymnasium library, but with two modifications; I randomized certain variables (like limb length, mass), and concatenated morphology information into the observation vector.

*The "Naive" policy is one trained on the default environment parameters, and the "Randomized" policy is trained on randomized environment parameters.*

### Cartpole
<video src="{{ '/assets/videos/one-policy-any-robot/cartpole-video.mp4' | url }}" class="post-asset" controls >
</video>

### BipedalWalker
<video src="{{ '/assets/videos/one-policy-any-robot/bipedalwalker-video.mp4' | url }}" class="post-asset" controls >
</video>

I was surprised to see that the gap in robustness in the BipedalWalker environment was less significant than I imagined. Of course, there was a gap; the policy trained in the modified environment was more robust to variations and its movements were much less awkward. But it often took significant variations to fault the vanilla policy. 

### Pusher
<video src="{{ '/assets/videos/one-policy-any-robot/pusher-video.mp4' | url }}" class="post-asset" controls>
</video>

_The arm itself has no collision by design. The robot can only interact with the cylinder by using the gripper._

The robustness gap was most visible in the `Pusher` environment. The policy trained on randomized forearm lengths had a much higher success rate than the vanilla policy, which appeared to succeed only when the forearm length was within a small margin of the length it was trained on. 

After playing around with simpler enviroments, I moved on to training actual cross-embodiment policies i.e. varying the number of joints instead of only mass and length. I reimplemented two architectures, NerveNet and URMA, to represent the performance of GNNs and Transformers respectively.

## Approaches
Arguably, the most important part of building a cross-embodiment architecture is deciding how your model adapts to the varying number of limbs and joints. You could still train a policy to adapt to varying limb sizes and joint conditions even if it only accepts a fixed number of joints (like I did in early experiments). But it would be limited to that specific number. You can't train it to control both bipeds *and* quadrupeds for example.

The two most popular approaches to solve this problem are Graph Neural Networks and Transformers. GNNs dominated early on because, for one, they existed before transformers. But also because modelling a robot as a graph is intuitive. <a class="reference" href="https://openreview.net/forum?id=S1sqHMZCb">NerveNet</a> is one of the earlier cross-embodiment architectures that utilitized GNNs.

GNNs are less compute-intensive than Transformers because nodes only communicate with their neighbors. But this becomes a problem when scaling to larger and more complex morphologies. If two limbs or joints that are far apart need to coordinate, messages between them would get washed out because they'd have to hop over many nodes in-between. 

Transformers don't have this problem because they're basically fully-connected GNNs; every node directly attends to every other node, so messages don't get washed out. This comes at extra computational cost since *all* the nodes are connected (even ones that don't need to be). Their robustness, however, still makes them favored over GNNs.

**But you don't have to entirely rely on a Transformer as the core action generator**. For example, the <a class="reference" href="https://arxiv.org/abs/2409.06366">Universal Robot Morphology Architecture (URMA)</a> only uses self-attention to "compress" variable-length morphology information into a fixed-length latent vector. The modules that come after this encoder are not attention-based, so computation is not only cheaper, but also faster than using a Transformer all the way through.

## NerveNet

<img src="{{ '/assets/images/one-policy-any-robot/nervenet-architecture.jpeg' | url }}" class="post-asset">
<a class="small-reference" href="https://openreview.net/forum?id=S1sqHMZCb">NerveNet: Learning Structured Policy with Graph Neural Networks</a>

NerveNet's architecture is pleasantly straightforward. Intialize a node for each joint in the robot, encode joint information and pass it thorough to its respective node, propagate messages for a few layers, pass the final hidden states through an action decoder, and in the end you get the motor commands for each joint.

NerveNet was originally made with only forward locomotion in mind. For my goal of having the robot move to an arbitrary target location, a slight modification was necessary. I settled for adding a global node that's connected to every node in the graph, and passing the goal information to that global node.

<img src="{{ '/assets/images/one-policy-any-robot/global-nervenet.png' | url }}" class="post-asset">

## URMA
The reason I picked URMA over other transformer-based architectures like <a class="reference" href="https://arxiv.org/abs/2203.11931">MetaMorph</a> is because it uses self-attention only for the most relevant operations. A single Multi-head Attention Layer is used for encoding observations of varying length (like joint and feet observations). But once it converts that information into a fixed sized latent vector, it uses standard MLPs for both the **Core Network** and the **Action Decoder.** This makes computation far cheaper than relying on a Transformer end-to-end.

The diagram used in the paper may seem a bit intimidating; it certainly was for me. But it's a lot simpler once you break it into modules. I'll describe these modules with my own simplified diagrams before I show you what the author's looks like.

### Joint and Feet Encoder
This module is responsible for compressing variable-length morphology information into a fixed-size latent vector. 

<img src="{{ '/assets/images/one-policy-any-robot/encoder.png' | url }}" class="post-asset">
<a class="small-reference">Encoder Module</a>

In the diagram above, **Description** means static information about a joint or a foot (like its local position and rotation axis), and **Observation** represents dynamic information (like its velocity or current angle). Each of these two inputs passes through its own encoder before their latent representations are multiplied. Then that product is passed into a single Multi-head Attention Layer to produce a fixed-size latent vector.

URMA has two versions of this module, one for encoding information about joints and another for feet.
    
### Core Network
This module is responsible for generating the action latent vector, which is essentially the "global action strategy."

<img src="{{ '/assets/images/one-policy-any-robot/core-network.png' | url }}" class="post-asset">
<a class="small-reference">Core Network Module</a>

It's a standard MLP. It takes in the latent vectors of joints and feet (which it gets from the attention encoders), and a **General Observation** vector (which contains aggregated information about the robot as a whole along with the target location), and it gives out an **Action Latent Vector**.

### Universal Decoder
This module is responsible for turning the abstract Action Latent Vector or "global action strategy" into low-level motor commands. 

<img src="{{ '/assets/images/one-policy-any-robot/decoder.png' | url }}" class="post-asset">
<a class="small-reference">Universal Decoder Module</a>

It has a really interesting design choice because it doesn't output the value of the action directly. This decoder gives out a mean value $\mu$ and a standard deviation value $\sigma$. The action $a$ is then calculated as:
$$
a = \mu + \sigma \cdot \epsilon
$$
The variable $\epsilon$ is a noise vector and where the randomness comes from during sampling. This is a common sampling method used in continuous control policies where you want more exploration early on in training. As the model gets better, you expect the value of $\sigma$ to get lower as the "confidence" of the model increases. Then you can completely drop the random sampling and just use $a = \mu$ during deployment.

### Putting it all together
Finally, put the three modules together and you get the complete **Universal Robot Morphology Architecture**; the best of Transformers without prohibitive computational cost:

<img src="{{ '/assets/images/one-policy-any-robot/urma-architecture.jpeg' | url }}" class="post-asset">
<a class="small-reference" href="https://openreview.net/forum?id=S1sqHMZCb">One Policy to Run Them All: an End-to-end Learning Approach to Multi-Embodiment Locomotion</a>

## Environment
I considered using the robots from `gymnasium-robotics`, but those models were clearly not meant to be modified. I figured it would be easier to build custom bare-bones models for on-the-fly modification. I setup an environment with 4 model variants: Biped, Biped with wheels, Quadruped, and Quadruped with wheels. The environment has functions to modify the dimensions of the torsos, legs, and wheels. 


```python
reward = (
    progress_reward
    + healthy_reward
    + ctrl_cost
)
```

The reward constitutes of a reward for moving toward the target (`progress_reward`), a reward for staying upright (`healthy_reward`), and a control cost (`ctrl_cost`) to penalize erratic and aggressive movements. Each term has its own weight, the highest one being that of the progress reward. Finding the right reward parameters has been arguably the most time-consuming part of this project. I will update this post with a breakdown of the performance of the two architectures once training is complete.

## Things I Left Out
### VLAs
Vision-Language-Action models have emerged recently as a promising path towards general cross-embodiment control going beyond simple tasks like locomotion. I have a lot of thoughts on them, but I believe that is beyond the scope of this project, so I'll explore them at some point in the future.

<iframe class="yt-link"
src="https://www.youtube.com/embed/6dme3JYj3Hg">
</iframe>

### Simplicity of Environment
My environment only had two distinct embodiments; significantly simpler than the embodiments explored in many cross-embodiment works. But as I was working on this project I realised that the number of practical embodiments deployed in the real-world is limited. I didn't see the purposed in building models to control the kinds of models used in the MetaMorph project:

<iframe class="yt-link"
src="https://www.youtube.com/embed/mGXtjLxyAkQ">
</iframe>

That doesn't take away from the fact that developing models to control different variants is crucial for robust deployments in unseen environments, even if the *category* of embodiment is the same. Imagine deploying a robot on another for autonomous exploration. It would fail if it was training for specific environmental conditions. What if it breaks its leg? What if it has to add mass onto itself to move to another location? I predict that cross-embodiment policies are going to be the industry standard for building robots in the future.
