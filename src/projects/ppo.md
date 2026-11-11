---
layout: post.njk
title: "An Overview of Proximal Policy Optimization"
date: 2026-10-18
description: "Intuitive breakdown of the PPO algorithm"
code: "https://github.com/estiftole/ppo"
banner: "/assets/images/reinforcement-learning/ppo_clipping.jpeg"
summary: "I reimplemented the PPO reinforcement learning algorithm in pytorch and used it to train a Breakout agent."
---

# Introduction
Let's say you want to build an agent that can maximize its score in a given environment - could be a game, a robot control task, whatever. When the agent performs an action $a_t$, it gets feedback in the form of a new state $s_t$, and (optionally) a reward $r_t$. The agent then takes this new information, performs a new action $a_{t+1}$, gets a new state $s_{t+1}$, and the loop goes on. 

Say you want your agent to learn the optimal **Policy** just from interacting with the environment. To do that, you'll have to answer one important question: _How do you determine how good an action is?_ That's the central problem all **Reinforcement Learning** algorithms are trying to solve.

The **Advantage** of action $a_t$, denoted by $A_t$, can be thought of as a measure of how good the action is. You'll likely encounter many different ways of calculating it, but the idea behind it is the same. You want your agent to learn to select actions with high Advantages more often than actions with low Advantages.

# The Value Function

How do you determine whether an action has a "high" or "low" Advantage? "High" and "low" are defined relatively, so we need to be clear about what we're comparing.

In PPO the Advantage of an action is essentially how much better it is than the average action. Meaning you compare the quality of the action to the average quality of _all_ actions you could take in that state, which is what we call the **Value** of that state.

More formally, the Value of a state $V(s_t)$ is defined as the **expected cumulative reward** that our agent can get starting from that state. Bit of a mouthful, so let's break it down.

## Value of a State

Remember how we said the agent gets a new state _and_ a reward when it performs an action? Imagine we take the sum of the rewards of every action in our agent's **trajectory**. That's $r_t+r_{t+1}+...+r_T$ where $r_t$ is the reward of action $a_t$, and $T$ is the total length of the episode. That sum is the **cumulative reward** $G_t$. 

(In practice we use a **discount factor** $\gamma$ (usually $0.9$) to assign a lower weight to rewards further out in the future.)

$$
G_t = r_t+\gamma  r_{t+1}+...+\gamma^{T-1}  r_T
$$

Now imagine calculating the cumulative rewards for _every possible trajectory_ starting from the current state and taking the average. That value is the **expected cumulative reward**.

$$
V(s_t) = \mathbb{E} \bigl[G_t | s_t \bigr]
$$

However, calculating something like this is prohibitive when we're dealing with a _huge_ number of possible trajectories. Instead, we train a **Value Function** to learn to _estimate_ the Value of the state. That estimated value is denoted by $V_{\theta}(s_t)$. The loss for the value function is then calculated as:

$$
L^{VF} = (V_{\theta}(s_t) - G_t)^2
$$

# The Policy Network

Cool. Now we know how to train our Value Function to estimate state values. But the Value Function never outputs actions, it just estimates the **Value** of a given state. Selecting actions is the responsibility of the **Policy Network**. So, what does its loss function look like?

If you look at the final Policy Network's loss function in the paper, it can seem really daunting. So, it's important that you get an intuition of what it's doing to understand it better.

## The Policy's Objective

In the introduction, we said that we want our Policy, denoted by $\pi_{\theta}$, to select actions with high Advantages more often and actions with low Advantages less often. That means we want to increase (or decrease) the probability of selecting an action based on its Advantage.

In its most basic form, we want to maximize:
$$ 
\pi_{\theta}(a_t|s_t) \hat{A_t} 
$$
where $\pi_{\theta}(a_t|s_t)$ is the probability of action $a_t$ getting selected at state $s_t$ under the policy $\pi_{\theta}$, and $\hat{A_t}$ is the estimated Advantage. If the Advantage $\hat{A_t}$ is positive, we'll _increase_ $\pi_{\theta}(a_t|s_t)$. If the Advantage is negative, we'll do the opposite and _decrease_ $\pi_{\theta}(a_t|s_t)$.

We want to increase/decrease the likelihood of the action being selected relative to the old weights of the model. So, we modify our objective to be like:

$$ 
\frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} \hat{A_t} 
$$

The formula's getting a bit cramped. Let's use $r_t(\theta)$ to represent the ratio and simplify the loss function:
$$ 
r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
\\ r_t(\theta) \hat{A_t} 
$$

This is nice, but training our Policy Network to maximize this as it is can lead to excessively large updates. To prevent that from happening, one more modification is required. We will want to **clip our loss**:
$$
clip(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A_t}
$$

This simply keeps the value of $r_t(\theta)$ between $1-\epsilon$ and $1+\epsilon$. The value for $\epsilon$ is most commonly set to $0.2$, so that means $r_t(\theta)$ is never allowed to get above $1.2$ or below $0.8$. But we probably only want to limit how big the reward can be. If we set a limit to how big the _punishment_ can be, then we'd be punishing the policy equally for actions that are "kinda" bad and actions that are _really_ bad.

To get that balance, PPO takes both the clipped and unclipped losses, and takes the minimum:
$$
min(r_t(\theta) \hat{A_t}, \; clip(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A_t})
$$


<img src="{{ '/assets/images/reinforcement-learning/ppo_clipping.jpeg' | url }}" class="post-asset">

For a positive Advantage $\hat{A_t} > 0$, the reward can only get as large as $1+\epsilon$. For a negative Advantage $\hat{A_t} < 0$, the baseline "punishment" is $1-\epsilon$, but the loss can get infinitely large as it's not constrained at all. So, the final clipped loss function $L^{CLIP}$ for our Policy Network looks like:

$$
L^{CLIP} = min(r_t(\theta) \hat{A_t}, \; clip(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A_t})
$$

We're doing great so far. But there's just one more number we need to calculate - the Advantage.

## Advantage of an Action

To get the Advantage of an action, we simply take the difference between the value of the state $V(s_t)$ and the quality of the action, denoted by $Q(s_t,a_t)$. 
$$
\hat{A_t} = Q(s_t,a_t)-V(s_t)
$$

This is where the Math gets a little heavy. 
There are two ways you could calculate the quality of the action $Q(s_t,a_t)$. The **Monte Carlo** approach is basically just equating it with the cumulative reward $G_t$ of the trajectory starting from state $s_t$:

$$
Q(s_t,a_t) = G_t = r_t+\gamma  r_{t+1}+...+\gamma^{T-1}  r_T \\
\hat{A_t} = G_t - V(s_t)
$$

The other approach (known as **Bootstrapping**) involves summing the reward of that state with the Value of the state it _resulted_ in, with a discount factor of course:

$$
Q(s_t,a_t) = r_t + V(s_{t+1}) \\
\hat{A_t} = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

The PPO algorithm uses something called the **generalized advantage estimation** that combines Bootstrapping and Monte Carlo to get a balance between **bias** and **variance**. It computes the Advantage as:

$$
\sigma_{t} = r_t + \gamma V(s_{t+1}) - V(s_t)
\\
\hat{A_t} = \sum_{k=0}^{T-t}(\gamma \lambda)^k \sigma_{t+k}
$$

where $\lambda$ is a coefficient that determines how close the estimation is to the Monte Carlo or the Boostrapping approaches. If it's $0$, $\hat{A_t}$ reduces to the Bootstrapped Advantage $r_t + \gamma V(s_{t+1}) - V(s_t)$. If it's $1$, it reduces to the Monte Carlo Advantage.

I skipped quite a bit of Math here for simplicity sake, so my fault if you're a bit confused. But what matters is that you get a general understanding of what the formulas are doing.

# Putting It All Together

Here’s what a standard PPO training loop might look like:

First, we let our Policy perform a bunch of actions in the environment. For each state timestep t, we sample an action at from the Policy Network and record the estimated Value $V_{\theta}(s_t)$ of the state that we get from the Value Function.

After our agent finishes an episode, we go through each timestep t and calculate the cumulative reward $G$ from that timestep until the last timestep $T$. We take the squared difference between that cumulative reward and estimated value we recorded earlier to get the loss for our value function:

$$
L^{VF} = (V_{\theta}(s_t) - G_t)^2
$$

Then we calculate the Advantage of the action taken in that state with the generalized advantage estimator:

$$
\sigma_{t} = r_t + \gamma V(s_{t+1}) - V(s_t)
\\
\hat{A_t} = \sum_{k=0}^{T-t}(\gamma \lambda)^k \sigma_{t+k}
$$

We then take that advantage, the probability of selecting the action $a_t$ by the model before and after the update, and we plug those values into the PPO clipped loss:

$$ 
r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
\\ L^{CLIP} = min(r_t(\theta) \hat{A_t}, \; clip(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A_t})
$$

And that's it! We perform updates for a few epochs on one episode before starting a brand-new episode and starting the loop all over again. Run this for a few hundred thousand iterations and—depending on the environment—your agent will get pretty good at racking up rewards.

I have made a number of oversimplifications in this post, so don’t take it as a full in-depth tutorial. It’s meant to make the PPO algorithm less intimidating and strip away its mystique. If you want to really dig in, I recommend [this video](https://www.youtube.com/watch?v=8jtAzxUwDj0) in addition to doing external reading on PPO; there’s plenty of good resources out there.

If you want to dig into code, you can find my implementation of PPO <a class="reference" href="https://github.com/estiftole/ppo">here.</a>
