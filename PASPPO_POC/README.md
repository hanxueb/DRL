
Implementation(Proof Of Concept) of Parameterized Action Space Proximal Policy Optimization, based on papers:
Hierarchical Approaches for Reinforcement Learning in Parameterized Action Space
https://arxiv.org/abs/1810.09656

Hybrid Actor-Critic Reinforcement Learning in Parameterized Action Space
https://arxiv.org/pdf/1903.01344.pdf

Proximal Policy Optimization Algorithms
https://arxiv.org/abs/1707.06347

Parameterized Action Space Proximal Policy Optimization illustration (each node is a neural network)
![Parameterized Action Space](https://github.com/hanxueb/DRL/blob/master/PASPPO_POC/PASPPO.png)

Number of layers, input, output and hidden layers can be configured.

1) learnerBase.py: Base class of algorithms, to extend with new algorithms easily.
2) learnerPASPPO.py: PAPPO implementation with tensorflow eager execution, build network based on modtype(model type class) and modname(model name) and train. 
3) policy.py:  Base class of policy networks, to extend with new policy networks easily.
4) policyPASPPO.py:  Policy network of PAPPO.
5) experience_buffer.py: Data structure for saving experience item(State, action, next state, reward...).
