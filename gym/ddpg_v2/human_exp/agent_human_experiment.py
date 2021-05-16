#!/usr/bin/env python3
import gym
import rospy
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
import rospkg
import os
import time
import tensorflow as tf
import numpy as np
from TD3 import td3
from std_msgs.msg import Int64
# from DDPG import DDPG
#####################  hyper parameters  ####################
MAX_EP_STEPS = 1000
LR_A = 0.0001   # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.99     # reward discount
TAU = 0.001      # soft replacement
MEMORY_CAPACITY = int(1e6)
BATCH_SIZE = 128
###############################  Data Collection  ####################################
def main():
    name = input('Please input your exp number:')
    folder = "Human_exp_" + str(name)
    os.system("mkdir "+folder)

    ####################### inite environment ########################################
    map_dir = "/home/i2rlab/RL_ws/src/turtlebot/turtlebot_gazebo/worlds/"
    os.rename(map_dir+"maze_exp.world",map_dir+"maze.world")
    rospy.init_node('turtlebot2_human', anonymous=True, log_level=rospy.WARN)
    env = StartOpenAI_ROS_Environment(
            'MyTurtleBot2HumanModel-v1')
    os.rename(map_dir+"maze.world",map_dir+"maze_exp.world")
    s_dim = env.observation_space.shape[0]+4
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high
    sign_talker = rospy.Publisher('/sign', Int64, queue_size=1)
    ####################### load agent #######################################
    TD3 = td3(a_dim, s_dim, a_bound, GAMMA, TAU, MEMORY_CAPACITY, BATCH_SIZE, LR_A, LR_C)
    TD3.loader()
    # ddpg = DDPG(a_dim, s_dim, a_bound, GAMMA, TAU, MEMORY_CAPACITY, BATCH_SIZE, LR_A, LR_C)
    # with ddpg.sess.as_default():
    #     with ddpg.g.as_default():
    #         ddpg.loader()
    d = [0.5,1,0.5,1]+list(6*np.ones([180]))
    A = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    TGREEN =  '\033[32m' # Green Text
    ENDC = '\033[m' # reset to the defaults

    #################### Initialize Data #################################
    n_h = 0          ############## number of eps of pure human control #################
    n_s_TD3 = 0          ############## number of eps of shared control #################
    # n_s_ddpg = 0

    n_crash_h = 0    ############## number of crash in pure human control #################
    n_crash_s_TD3 = 0    ############## number of crash in shared control #################
    # n_crash_s_ddpg = 0

    n_success_h = 0  ############## number of success of pure human control #################
    n_success_s_TD3 = 0  ############## number of success of shared control #################
    # n_success_s_ddpg = 0

    t_h = 0          ############## total time of pure human control #################
    t_s_TD3 = 0          ############## total time of shared control #################
    # t_s_ddpg = 0
    #################### SART #####################
    SA_h = 0
    SA_s_TD3 = 0
    # SA_s_ddpg = 0
    #################### workload #################
    WL_h = 0
    WL_s_TD3 = 0
    # WL_s_ddpg = 0
    #################### Start ####################
    i=0
    total_ep = 2
    sequence_ep = [0,1]
    #np.random.shuffle(sequence_ep)
    pureh_ep = sequence_ep[0:int(total_ep/2)]
    # ddpg_ep = sequence_ep[int(total_ep/3):int(2*total_ep/3)]
    td3_ep = sequence_ep[int(total_ep/2):]
    n_d = 0
    while i < total_ep:
        s = env.reset()
        t1 = time.time()
        min_distance_ob = s[-2]
        theta_bt_uh_ob = s[-1]
        a_old = A[0]
        s = np.hstack((a_old,np.array(s[3:-2])/d))
        done = 0
        while not done:
            ################ Mode #########################################
            if i in td3_ep:
                a = TD3.choose_action(s,1)
                if a[1]<=0.9 and min_distance_ob<=0.7 and abs(theta_bt_uh_ob)<1.3:
                    print (TGREEN + "Going to Crash!" , ENDC)
                    a = A[1]
            elif i in pureh_ep:
                a = A[0]

            s_, r, done, info = env.step(a)
            sign_talker.publish(int(r))
            if r <-400 and done:
                done = 0
                if i in td3_ep:
                    n_crash_s_TD3+=1
                elif i in pureh_ep:
                    n_crash_h+=1
            print("n_crash_s_TD3:",n_crash_s_TD3)
            print("n_crash_h:",n_crash_h)
            print("i:",i)
            print("n_d:",n_d)
            min_distance_ob = s_[-2]
            theta_bt_uh_ob = s_[-1]
            a_old = a
            s=np.hstack((a_old,np.array(s_[3:-2])/d))
            if done:
                t=0
                n_d=n_d+1
                if n_d %5==0:
                    instability = int(input('How changeable is the situation (1: stable and straightforward -- 7: changing suddenly):'))
                    variability = int(input('How many variables are changing within the situation: (1: very few -- 7: large number):'))
                    complexity = int(input('How complicated is the situation (1: simple -- 7: complex):'))
                    arousal = int(input('How aroused are you in the situation (1: low degree of alertness -- 7: alert and ready for activity):'))
                    spare = int(input('How much mental capacity do you have to spare in the situation (1: Nothing to spare -- 7: sufficient):'))
                    concentration = int(input('How much are you concentrating on the situation (1: focus on only one thing -- 7: many aspect):'))
                    attention = int(input('How much is your attention divide in the situation (1: focus on only one thing -- 7: many aspect):'))
                    quantity = int(input('How much information have your gained about the situation (1: little -- 7: much):'))
                    quality = int(input('How good information have you been accessible and usable (1: poor -- 7: good):'))
                    famlilarity = int(input('How familar are you with the situation (1: New situation -- 7: a great deal of relevant experience):'))
                    SA = quantity + quality + famlilarity - ((instability + variability + complexity) - (arousal + spare +concentration + attention))
                    if i in td3_ep:
                        SA_s_TD3 += SA
                        WL_s_TD3 += float(input('Please input your workload (from TLX):'))
                        t_s_TD3 += t
                        n_s_TD3 += 1
                        if r>500:
                            n_success_s_TD3+=1
                    elif i in pureh_ep:
                        SA_h += SA
                        WL_h += float(input('Please input your workload (from TLX):'))
                        t_h += t
                        n_h += 1
                        if r>500:
                            n_success_h+=1
                    i=i+1
                break


    np.savetxt('data.dat',np.array([[n_s_TD3,t_s_TD3,n_crash_s_TD3,n_success_s_TD3, SA_s_TD3, WL_s_TD3], [n_h,t_h,n_crash_h,n_success_h, SA_h, WL_h]]))
    #,[n_s_ddpg,t_s_ddpg,n_crash_s_ddpg,n_success_s_ddpg, SA_s_ddpg, WL_s_ddpg]]))
    ########### shared_TD3: number of eps, total time, time of crash, time of success, situation awareness, workload ###########
    ########### human: number of eps, total time, time of crash, time of success, situation awareness, workload ###########
    ########### shared_TD3_ddpg: number of eps, total time, time of crash, time of success, situation awareness, workload ###########
    os.system('cp -r data.dat '+folder+'/')



if __name__ == '__main__':
    main()
