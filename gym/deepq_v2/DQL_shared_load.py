import gym
import rospy
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
from RL_brain import DQN
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


rospy.init_node('share_training', anonymous=True, log_level=rospy.WARN)
env = StartOpenAI_ROS_Environment(
        "MyTurtleBot2HumanModel-v0")
MEMORY_SIZE = int(5e5)
ACTION_SPACE = 4
Obs_Space = env.observation_space.shape[0]+4
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=config)

DQN = DQN(
    n_actions=ACTION_SPACE, n_features=Obs_Space, memory_size=MEMORY_SIZE, batch_size= 128,
    e_greedy_increment=(0.99/0.7)**(1/int(3e5)), sess=sess, output_graph=True,learning_rate=0.001)

sess.run(tf.global_variables_initializer())


def train(RL):
    MAX_EPISODES = 10000
    MAX_EP_STEPS = 1000
    RL.backuploader()
    RL.memory = np.load('memory1.npy')
    RL.memory_counter = int(np.load('parameters1.npy')[0])
    RL.learn_step_counter = int(np.load('parameters1.npy')[1])
    RL.epsilon = float(np.load('parameters1.npy')[2])
    eval_reward_list = list(np.load('eval_reward.npy')) ###########need to load #################
    EVAL_FREQUENCY = 10000
    reward_list = list(np.load('reward1.npy'))
    highest_reward = float(np.load('parameters1.npy')[3])
    highest_step = int(np.load('parameters1.npy')[-1])
    d = [0.5,1,0.5,1]+list(6*np.ones([180]))
    A = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    TGREEN =  '\033[32m' # Green Text
    ENDC = '\033[m' # reset to the defaults
    n_crash = int(np.load('parameters1.npy')[4])
    n_prevent = int(np.load('parameters1.npy')[5])
    crash_state = []
    for i in range(len(reward_list),len(reward_list)+1500):
        if RL.memory_counter % EVAL_FREQUENCY==0 and RL.memory_counter>=50000:
            avg_reward = 0.
            for k in range(10):
                s = env.reset()
                min_distance_ob = s[-2]
                theta_bt_uh_ob = s[-1]
                a_old = A[0]
                s = np.hstack((a_old,np.array(s[3:-2])/d))
                for j in range(MAX_EP_STEPS):
                    a = A[RL.choose_action(s,1)]
                    # if a[1]<=0.9 and min_distance_ob<=0.7 and abs(theta_bt_uh_ob)<1.3:
                    #     print (TGREEN + "Going to Crash!" , ENDC)
                    #     n_prevent +=1
                    #     #RL.store_transition(s, a, -500, np.zeros(s_dim), 1)
                    #     a = A[1]

                    s_, r, done, info = env.step(a)
                    if done and r < -200:
                        n_crash +=1
                        crash_state.append([min_distance_ob,theta_bt_uh_ob]+list(a))
                        np.save('crash.npy',crash_state)
                    min_distance_ob = s_[-2]
                    theta_bt_uh_ob = s[-1]
                    r -= 50*np.linalg.norm(a_old-a)**2
                    print('actual reward:',r)
                    a_old = a
                    s=np.hstack((a_old,np.array(s_[3:-2])/d))
                    avg_reward += r
                    if done:
                        break
            avg_reward /= 10
            eval_reward_list.append([RL.memory_counter,avg_reward])
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
            if RL.memory_counter<0:
                if min_distance_ob>1.35:
                    a_index = 0
                elif min_distance_ob<=0.7:
                    a_index = 1
                else:
                    if theta_bt_uh_ob>np.pi*0.4:
                        a_index = 0
                    else:
                        n = np.argmax(a_old)
                        if n ==2:
                            a_index = 2
                        elif n==3:
                            a_index = 3
                        else:
                            a_index = np.random.choice([2,3])

            else:
                a_index = RL.choose_action(s,2)


            # if a_index!=1 and min_distance_ob<=0.7 and abs(theta_bt_uh_ob)<1.3:
            #     print (TGREEN + "Going to Crash!" , ENDC)
            #     n_prevent +=1
            #     RL.store_transition(s, a_index, -500, np.zeros(Obs_Space), 1)
            #     a_index = 1
            #     if highest_reward<600:
            #         pre = 1
            a = A[a_index]
            s_, r, done, info = env.step(a)
            r -= 50*np.linalg.norm(a_old-a)**2
            print('actual reward:',r)
            if done and r < -200:
                n_crash +=1
                crash_state.append([min_distance_ob,theta_bt_uh_ob]+list(a))
                np.save('crash.npy',crash_state)
            min_distance_ob = s_[-2]
            theta_bt_uh_ob = s_[-1]
            a_old = a
            s_ = np.hstack((a_old,np.array(s_[3:-2])/d))
            RL.store_transition(s, a_index, r, s_, done)
            if pre == 1:
                done = 1
            if RL.memory_counter > 5e4:
                RL.learn()
            s = s_
            ep_reward += r
            print("number of crash:",n_crash,"number of prevent crash:",n_prevent)
            if done or j == MAX_EP_STEPS-1:
                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                print('Episode:', i, ' Reward: %i' % int(ep_reward),"highest_step:",highest_step )
                # if ep_reward > -300:RENDER = True
                if j>3:
                    reward_list.append(ep_reward)
                break
            if RL.memory_counter % EVAL_FREQUENCY==0 and RL.memory_counter>=50000:
                break
            if RL.memory_counter > 5e5:
                break

        np.save('reward1.npy',reward_list)
        if i%50 ==0:
            np.save('memory1.npy',RL.memory)
            print('memory saved')
            np.save('parameters1.npy',np.array([RL.memory_counter,RL.learn_step_counter,RL.epsilon,highest_reward,n_crash,n_prevent,highest_step]))
            print('parameters saved')
            print('***try to save***')
            RL.backupsaver()
        if i > 100:
            mean100 = np.mean(reward_list[-100:])
            if mean100 > highest_reward:
                highest_reward = mean100
                highest_step = RL.memory_counter
                #print('***try to save***')
                #RL.saver()
                if mean100>600:
                    print("mean100_reward:",mean100)
                    print("highest_reward:",highest_reward)
                    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                    np.save('memory1.npy',RL.memory)
                    print('memory saved')
                    np.save('parameters1.npy',np.array([RL.memory_counter,RL.learn_step_counter,RL.epsilon,highest_reward,n_crash,n_prevent,highest_step]))
                    print('parameters saved')
                    RL.saver()
                    #break
            print("mean100_reward:",mean100)
            print("highest_reward:",highest_reward)
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        if RL.memory_counter > 5e5:
            np.save('parameters1.npy',np.array([RL.memory_counter,RL.learn_step_counter,RL.epsilon,highest_reward,n_crash,n_prevent,highest_step]))
            break
if __name__ == '__main__':
    train(DQN)
