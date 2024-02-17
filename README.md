# Actively Learning Reinforcement Learning
This code is for the numerical example used in our paper "Actively Learning Reinforcement Learning: A Stochastic Optimal Control Approach" found her [https://arxiv.org/pdf/2309.10831.pdf](https://arxiv.org/pdf/2309.10831.pdf).

### Implementation (reproducing the results in the paper)
We use the notebook main.ipynb to implement, in a step-by-step and user-friendly fashion, the learning algorithm as explained in our paper above. We also conclude it by a closed-loop performance comparison, between the reinforcement learning controller and a certainty equivalence LQR controller.

### Adjusting the state-space example and/or the reinforcement learning algorithm
The code can be adopted to different examples by adjusting the dynamic model defined in Example_system.py. The reinforcement learning algorithm (here it is the DDPG) can be changed as well through changing/replacing DeterministicPolicyGradient.py by the intended algorithm.

### The extended Kalman filter and AD
Example systems can be defined without explicitly stating their state/output dynamics' jacobians; the extended Kalman filter code ExtendedKF.py itself implements automatic differentiation and can get these jacobians easily.


