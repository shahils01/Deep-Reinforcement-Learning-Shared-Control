# Deep-Reinforcement-Learning-Shared-Control

Shared control of mobile robots integrates manual input with auxiliary autonomous controllers to improve the overall system performance. We develop an extended Twin Delayed DDPG (TD3X) based shared controlframework that learns to assist a human operator in teleoperating mobile robots optimally. The robot’s states, shared control ratio in the previous time step, and human’s control input are used as inputs to the reinforcement learning (RL) agent, which then outputs the optimal shared control ratio between human input and autonomous controllers without knowing the human model. We develop noisy softmax policies to make the TD3X algorithm feasible under the constraint of a shared control ratio. Furthermore, to accelerate the training process and protect the robot, we develop a navigation demonstration policy and a safety guard. A neuralnetwork (NN) structure is developed to maintain the correlation of sensor readings among heterogeneous input data and improve the learning speed. We also develop an extended DAGGER (DAGGERX) human agent for training the RL agent to reduce humanworkload. 

# Install

The SRC folder contain all the required packages. Aditionally, Install turtlebot the packages.

The Gym folder contains human_exp which includes the human agent and TD3X based shared control learning algorithms. 

# Workflow

roscore

rosparam load my_turtlebot2_goa.yaml

Then use any algorithm you want to use
