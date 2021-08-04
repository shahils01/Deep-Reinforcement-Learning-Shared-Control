#!/usr/bin/env python3
import os
import time
import tensorflow as tf
import numpy as np

#####################  hyper parameters  ####################

MAX_EPISODES = 10
n_iteration = 24
#####################  Net Work Building  ###################

class Human_model(object):
    """docstring for Human_model"""
    def __init__(self):
        super(Human_model, self).__init__()
        self.sess = tf.Session()
        self.S = tf.placeholder(tf.float32, [None, 13], 'S')
        self.a_real = tf.placeholder(tf.float32, [None, 2], 'a_real')
        self.keep_prob = tf.placeholder(tf.float32)
        self.a_predict = self.build_c(self.S,self.keep_prob)
        self.loss = tf.losses.mean_squared_error(labels=self.a_real, predictions=self.a_predict)
        self.train = tf.train.AdamOptimizer(0.001).minimize(self.loss)
        #self.loader()
        self.train_set = np.loadtxt('scaled_house_data_annotated_2.dat')
        self.initsess = tf.global_variables_initializer()


    def build_c(self,S,keep_prob):
        net1 = tf.layers.dense(S, 256, activation=tf.nn.relu, name='l1', trainable=True, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
        drop_out1 = tf.nn.dropout(net1, keep_prob)
        net2 = tf.layers.dense(drop_out1,256, activation=tf.nn.relu,name='l2',trainable=True, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
        drop_out2 = tf.nn.dropout(net2, keep_prob)
        net3 = tf.layers.dense(drop_out2,128, activation=tf.nn.relu,name='l3',trainable=True, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
        drop_out3 = tf.nn.dropout(net3, keep_prob)
        a = tf.layers.dense(drop_out3,2,trainable=True,activation=tf.tanh)
        return tf.multiply(a, [0.5,1], name='scaled_a')

    def choose_action(self,s):
        a = self.sess.run(self.a_predict,feed_dict={self.S:s,self.keep_prob:1})
        return a[0]

    def train_net(self,j):
        #self.sess.run(self.initsess)
        data_n = int(5000+j*2000)
        print(data_n)
        self.train_set = np.loadtxt('scaled_house_data_annotated_2.dat')[:-3000,:]
        self.text_set = np.loadtxt('scaled_house_data_annotated_2.dat')[-3000:,:]
        train_set_x = self.train_set[:,0:-2]
        train_set_y = self.train_set[:,-2:]
        text_set_x = self.text_set[:,0:-2]
        text_set_y = self.text_set[:,-2:]
        MEMORY_CAPACITY=train_set_x.shape[0]
        BATCH_SIZE = 2046
        self.MSE = 1
        for i in range(10000):
            index = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
            bs = train_set_x[index,:]
            ba = train_set_y[index,:]
            self.sess.run(self.train,feed_dict={self.S:bs,self.a_real:ba,self.keep_prob:0.5})
            if i % 100 == 0:
                cost = self.sess.run(self.loss,feed_dict={self.S:text_set_x,self.a_real:text_set_y,self.keep_prob:1})
                #costlist.append(cost)
                print("after %i iteration, MSE: %f" %(i, cost))
                if cost < self.MSE:
                    self.MSE = cost

    def loader(self,j):
        loader= tf.train.Saver()
        loader.restore(self.sess,tf.train.latest_checkpoint('Human_model_exp_'+str(j)))

    def saver(self,j):
        saver = tf.train.Saver()
        saver.save(self.sess,"Human_model_exp_40"+"/net")
        print("*****net_saved******")

############################Collect Data####################################
def remove_zero_rows(X):
    # X is a scipy sparse matrix. We want to remove all zero rows from it
    nonzero_row_indice, _ = X.nonzero()
    unique_nonzero_indice = np.unique(nonzero_row_indice)
    return X[unique_nonzero_indice]

def main():
    Human = Human_model()
    success_rate = np.zeros([1,n_iteration])
    MSE = np.zeros([10,n_iteration])
    time_data = np.zeros([MAX_EPISODES,n_iteration])
    for i in range(10):
        Human.sess.run(Human.initsess)
        for j in range(n_iteration):
            Human.train_net(j)
            Human.saver(j)
            MSE[i,j]=Human.MSE
            np.savetxt("MSE3.dat",MSE)





if __name__ == '__main__':
    main()
