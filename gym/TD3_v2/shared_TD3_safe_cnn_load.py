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

#####################  hyper parameters  ####################
MAX_EPISODES = 2001
MAX_EP_STEPS = 1000
LR_A = 0.0001
LR_C = 0.001
TAU = 0.0005
GAMMA = 0.99     # reward discount
MEMORY_CAPACITY = int(5e5)
BATCH_SIZE = 128
desired_norm_factor=(1/30)**(1/int(3e5))
EVAL_FREQUENCY = 10000
n_start_train_eval = 5e4
TGREEN =  '\033[32m' # Green Text
ENDC = '\033[m' # reset to the defaults
###############################  Data Collection  ####################################
def main():
    rospy.init_node('turtlebot2_human', anonymous=True, log_level=rospy.WARN)
    env = StartOpenAI_ROS_Environment(
            'MyTurtleBot2HumanModel-v0')
    s_dim = env.observation_space.shape[0]+4
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high
    d = [0.5,1,0.5,1]+list(6*np.ones([180]))
    A = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    TD3 = td3(a_dim, s_dim, a_bound, GAMMA, TAU, MEMORY_CAPACITY, BATCH_SIZE, LR_A, LR_C)

    TD3.backuploader()
    reward_list = list(np.load('reward1.npy'))
    eval_reward_list = list(np.load('eval_reward.npy'))
    TD3.memory = np.load('memory1.npy')
    TD3.pointer = int(np.load('parameters1.npy')[1])
    highest_reward = int(np.load('parameters1.npy')[2])
    highest_step = int(np.load('parameters1.npy')[7])
    #TD3.loader()
    var = float(np.load('parameters1.npy')[0])#7  # control exploration
    corrent_norm = float(np.load('parameters1.npy')[3])
    desired_norm = float(np.load('parameters1.npy')[4])
    n_crash = int(np.load('parameters1.npy')[5])
    n_prevent = int(np.load('parameters1.npy')[6])
    for i in range(len(reward_list),len(reward_list)+1000):
        if TD3.pointer % EVAL_FREQUENCY==0 and TD3.pointer>=n_start_train_eval:
            avg_reward = 0.
            for k in range(10):
                s = env.reset()
                min_distance_ob = s[-2]
                theta_bt_uh_ob = s[-1]
                a_old = A[0]
                s = np.hstack((a_old,np.array(s[3:-2])/d))
                for j in range(MAX_EP_STEPS):
                    a = TD3.choose_action(s,1)
                    if a[1]<=0.9 and min_distance_ob<=0.7 and abs(theta_bt_uh_ob)<1.3:
                        print (TGREEN + "Going to Crash!" , ENDC)
                        n_prevent +=1
                        #TD3.store_transition(s, a, -500, np.zeros(s_dim), 1)
                        a = A[1]

                    s_, r, done, info = env.step(a)
                    if done and r < -200:
                        n_crash +=1
                    min_distance_ob = s_[-2]
                    theta_bt_uh_ob = s_[-1]
                    r -= 50*np.linalg.norm(a_old-a)**2
                    print('actual reward:',r)
                    a_old = a
                    s=np.hstack((a_old,np.array(s_[3:-2])/d))
                    avg_reward += r
                    if done:
                        break
            avg_reward /= 10
            eval_reward_list.append([TD3.pointer,avg_reward])
            np.save('eval_reward.npy',eval_reward_list)
            del k
            del j
            del avg_reward
        s = env.reset()
        min_distance_ob = s[-2]
        theta_bt_uh_ob = s[-1]
        a_old = A[0]
        s = np.hstack((a_old,np.array(s[3:-2])/d))
        ep_reward = 0
        for j in range(MAX_EP_STEPS):
            pre = 0
            if TD3.pointer<40000:
                if min_distance_ob>1:# and (np.argmax(a_old)==2 or np.argmax(a_old)==3)) or abs(theta_bt_uh_ob)> 1.3:
                    n=0
                    a = A[0].copy()*10
                # elif min_distance_ob>1 and (np.argmax(a_old)==0 or np.argmax(a_old)==1) or abs(theta_bt_uh_ob)> 1.3:
                #     n=0
                #     a = A[0].copy()*10
                elif min_distance_ob<=0.7:
                    n=1
                    a = A[1].copy()*10
                else:
                    if theta_bt_uh_ob>np.pi*0.4:
                        n=0
                        a = A[0].copy()*10
                    else:
                        n = np.argmax(a_old)
                        if n ==2:
                            a = A[2].copy()*10
                        elif n==3:
                            a = A[3].copy()*10
                        else:
                            n=np.random.choice([2,3])
                            a = A[n].copy()*10
                a_original = A[n]
                print("a_original:",a_original)
                a = a+(np.random.normal(0, var,4))
                #a = np.clip(a+(np.random.normal(0, var,4)),-10,10)
                a = np.exp(a)/np.sum(np.exp(a), axis=0)
                corrent_norm = corrent_norm*0.99+0.01*np.linalg.norm(a-a_original)
                if var<=30:
                    if corrent_norm< desired_norm:
                        var=var*0.99+var*1.1*0.01
                if corrent_norm>desired_norm:
                    var=var*0.99+var/1.1*0.01
                print("corrent_norm:",corrent_norm,"desired_norm:",desired_norm,"var:",var)
                if a[1]<=0.9 and min_distance_ob<=0.7 and abs(theta_bt_uh_ob)<1.3:
                    print (TGREEN + "Going to Crash!" , ENDC)
                    n_prevent +=1
                    TD3.store_transition(s, a, -500, np.zeros(s_dim), 1)
                    a = A[1]
            else:
                a = TD3.choose_action(s,2)
                a_original = np.exp(a)/np.sum(np.exp(a), axis=0)
                print("a_original:",a_original)
                a = a+(np.random.normal(0, var,4))
                #a = np.clip(a+(np.random.normal(0, var,4)),-10,10)
                a = np.exp(a)/np.sum(np.exp(a), axis=0)
                corrent_norm = corrent_norm*0.99+0.01*np.linalg.norm(a-a_original)
                if var<=30:
                    if corrent_norm< desired_norm:
                        var=var*0.99+var*1.1*0.01
                if corrent_norm>desired_norm:
                    var=var*0.99+var/1.1*0.01
                print("corrent_norm:",corrent_norm,"desired_norm:",desired_norm,"var:",var)

                if a[1]<=0.9 and min_distance_ob<=0.7 and abs(theta_bt_uh_ob)<1.3:
                    print (TGREEN + "Going to Crash!" , ENDC)
                    n_prevent +=1
                    TD3.store_transition(s, a, -500, np.zeros(s_dim), 1)
                    a = A[1]
                    if highest_reward<600:
                        pre = 1

            s_, r, done, info = env.step(a)
            r -= 50*np.linalg.norm(a_old-a)**2
            print('actual reward:',r)
            if done and r < -200:
                n_crash +=1
            min_distance_ob = s_[-2]
            theta_bt_uh_ob = s_[-1]
            a_old = a
            s_ = np.hstack((a_old,np.array(s_[3:-2])/d))
            TD3.store_transition(s, a, r, s_, done)
            if pre == 1:
                done = 1

            if TD3.pointer > n_start_train_eval:
                if desired_norm>0.01:
                    desired_norm*=desired_norm_factor
                TD3.learn()

            s = s_
            ep_reward += r
            print("number of crash:",n_crash,"number of prevent crash:",n_prevent)
            if done or j == MAX_EP_STEPS-1:
                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var,"highest_step:",highest_step )
                # if ep_reward > -300:RENDER = True
                if j>3:
                    reward_list.append(ep_reward)
                break
            if TD3.pointer % EVAL_FREQUENCY==0 and TD3.pointer>=n_start_train_eval:
                break
            if TD3.pointer > 5e5:
                break

        np.save('reward1.npy',reward_list)
        if i%50 ==0:
            np.save('memory1.npy',TD3.memory)
            print('memory saved')
            np.save('parameters1.npy',np.array([var,TD3.pointer,highest_reward,corrent_norm,desired_norm,n_crash,n_prevent,highest_step]))
            print('parameters saved')
            print('***try to save***')
            TD3.backupsaver()
        if i > 100:
            mean100 = np.mean(reward_list[-100:])
            if mean100 > highest_reward:
                highest_reward = mean100
                highest_step = TD3.pointer
                #print('***try to save***')
                #TD3.saver()
                if mean100>800:
                    print("mean100_reward:",mean100)
                    print("highest_reward:",highest_reward)
                    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                    np.save('best_reward.npy',reward_list[-100:])
                    np.save('memory1.npy',TD3.memory)
                    print('memory saved')
                    np.save('parameters1.npy',np.array([var,TD3.pointer,highest_reward,corrent_norm,desired_norm,n_crash,n_prevent,highest_step]))
                    print('parameters saved')
                    TD3.saver()
                    #break

            print("mean100_reward:",mean100)
            print("highest_reward:",highest_reward)
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        if TD3.pointer > 5e5:
            np.save('parameters1.npy',np.array([var,TD3.pointer,highest_reward,corrent_norm,desired_norm,n_crash,n_prevent,highest_step]))
            break

if __name__ == '__main__':
    main()
