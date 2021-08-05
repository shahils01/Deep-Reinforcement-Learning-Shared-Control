# Deep-Reinforcement-Learning-Shared-Control

Shared control of mobile robots integrates manual input with auxiliary autonomous controllers to improve the overallsystem performance. However, prior work that seeks to find the optimal shared control ratio needs an accurate human model,which is usually challenging to obtain. In this paper, we develop an extended Twin Delayed DDPG (TD3X) based shared controlframework that learns to assist a human operator in teleoperating mobile robots optimally. The robot’s states, shared control ratio in the previous time step, and human’s control input are used as inputs to the reinforcement learning (RL) agent, which then outputs the optimal shared control ratio between human input and autonomous controllers without knowing the human model. We develop noisy softmax policies to make the TD3X algorithm feasible under the constraint of a shared control ratio. Furthermore, to accelerate the training process and protect the robot, we develop a navigation demonstration policy and a safety guard. A neuralnetwork (NN) structure is developed to maintain the correlation of sensor readings among heterogeneous input data and improve the learning speed. We also develop an extended DAGGER (DAGGERX) human agent for training the RL agent to reduce humanworkload. Robot simulations and experiments with humans-in-the-loop are conducted. Results show that the DAGGERX humanagent can simulate real human inputs in the worst-case scenarios with a mean square error of 0.012. Compared to the original TD3 agent, the TD3X based shared control system decreased the average collision number from 387.3 to 44.4 and increasedthe maximum average return from 1043 to 1187 with a faster converge speed. In the human subject tests, participants’ averageperceived workload was significantly lower in shared control than exclusively manual control (26.90 vs. 40.07, p=0.013).

# Install

Install all the dependent package based on 
https://www.processon.com/view/link/5e399ebae4b03e660716b193

The SRC folder contain all the required packages. Aditionally, Install turtlebot the packages.

The Gym folder contains TD3_v1, TD3_v2, ddpg_v2, deepq_v2, and human_exp.

TD3_v1 is the basic version of the TD3 algorithm.

TD3_v2 is the extended version of TD3 algorithm.

ddpg_v2 and deepq_v2 are the baseline Deep Deterministic Q-Learning and Deep Q-Learning algothms used for comparision with our algorithm.

# API for Learning Algorithm

try.py for the training of different learning algorithm for 10 different seeds

plot.py for the plot of reward, average reward for last x expisode, and reward in evaluation loop

# APT for Human Subject Experiment

agent_human_familar.py used to familar with the control

agent_human_experiment.py for the real experiment

sign.py for the sign

analysis.py for data analysis

# Workflow

roscore

rosparam load my_turtlebot2_goa.yaml

Then use any algorithm you want to use
