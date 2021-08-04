#!/usr/bin/env python3
import gym
import rospy
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
import rospkg
import os
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from TD3 import td3
from std_msgs.msg import Int64

#####################  hyper parameters  ####################
MAX_EP_STEPS = 1500
LR_A = 0.0001   # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.99     # reward discount
TAU = 0.001      # soft replacement
MEMORY_CAPACITY = int(1e6)
BATCH_SIZE = 128


###############################  Data Collection  ####################################
def main():
    map_dir = "/home/i2rlab/shahil_files/shahil_RL_ws_new/src/turtlebot/turtlebot_gazebo/worlds/"
    #os.rename(map_dir+"house_train.world",map_dir+"maze.world")
    rospy.init_node('turtlebot2_human', anonymous=True, log_level=rospy.WARN)
    env = StartOpenAI_ROS_Environment(
            'MyTurtleBot2HumanModel-v1')
    #os.rename(map_dir+"maze.world",map_dir+"house_train.world")
    s_dim = env.observation_space.shape[0]+4
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high
    #print('a_bound test', a_bound)
    sign_talker = rospy.Publisher('/sign', Int64, queue_size=1)
    ####################### load agent #######################################
    #TD3 = td3(a_dim, s_dim, a_bound, GAMMA, TAU, MEMORY_CAPACITY, BATCH_SIZE, LR_A, LR_C)
    #TD3.loader()

    d = [0.5,1,0.5,1]+list(6*np.ones([180]))
    #A = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    #TGREEN =  '\033[32m' # Green Text
    #ENDC = '\033[m' # reset to the defaults

    #################### Start ####################
    i=0
    while i <2:
        s = env.reset()
        #print("env printing: ",env)
        t1 = time.time()
        min_distance_ob = s[-2]
        theta_bt_uh_ob = s[-1]
        #a_old = A[0]
        a_old = np.array([1,0,0,0])
        s = np.hstack((a_old,np.array(s[3:-2])/d))
        for j in range(MAX_EP_STEPS):
            ################ Mode #########################################
            #if i%2==0:
                #a = TD3.choose_action(s,1)
                #a = [1,0,0,0]
                #if a[1]<=0.9 and min_distance_ob<=0.7 and abs(theta_bt_uh_ob)<1.3:
                    #print (TGREEN + "Going to Crash!" , ENDC)
                    #a = A[1]
            '''else:
                a = A[0]'''

            #x_test = env.state_msg.pose.position.x
            #print('x_test',env.collect_data)

            a = np.array([1,0,0,0])
            s_, r, done, info = env.step(a)
            sign_talker.publish(int(r))
            if r <-200:
                done = 0
            min_distance_ob = s_[-2]
            theta_bt_uh_ob = s_[-1]
            a_old = a
            s=np.hstack((a_old,np.array(s_[3:-2])/d))
            if done or j == MAX_EP_STEPS-1:
                t = time.time()-t1
                if j>3:
                    i=i+1
                break

if __name__ == '__main__':
    main()
