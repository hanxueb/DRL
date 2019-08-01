
Implementation(Proof Of Concept) of Parameterized Action Space Proximal Policy Optimization, based on papers:

Hybrid Actor-Critic Reinforcement Learning in Parameterized Action Space
https://arxiv.org/pdf/1903.01344.pdf

Hierarchical Approaches for Reinforcement Learning in Parameterized Action Space
https://arxiv.org/abs/1810.09656

Proximal Policy Optimization Algorithms
https://arxiv.org/abs/1707.06347

Parameterized Action Space Proximal Policy Optimization illustration (each node is a neural network)
![Parameterized Action Space](https://github.com/hanxueb/DRL/blob/master/PAPPO_POC/PAPPO.png)

can support flexible layers based on configuration 

1) learnerBase.py: Common operations of different algorithms.
2) learnerPAPPO.py: PAPPO implementation with tensorflow eager execution, build network based on modtype(model type class) and modname(model name) and train. 
3) policyPAPPO.py:  policy network of PAPPO
4) experience_buffer.py: Data structure for saving experience item(State, action, next state, reward...)

To add:
1) A3C, SDQN, DDPG, D4PG algorithms.
2) Advantages calculation as in PPO Paper
