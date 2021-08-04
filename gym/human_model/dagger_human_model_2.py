#!/usr/bin/env python3
import os
import time
import tensorflow as tf
import numpy as np

#####################  hyper parameters  ####################

MAX_EPISODES = 10
n_iteration = 24
#n_iteration = 1

MAX_EP_STEPS = 1500
LR_A = 0.0001   # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.99     # reward discount
TAU = 0.001      # soft replacement
MEMORY_CAPACITY = int(1e6)
BATCH_SIZE = 128

#####################  Net Work Building  ###################

class Human_model(object):
    """docstring for Human_model"""
    def __init__(self):
        super(Human_model, self).__init__()
        self.sess = tf.Session()
        self.S1 = tf.placeholder(tf.float32, [None, 5], 'S1')
        #self.S2 = tf.placeholder(tf.float32, [None, 198, 1], 'S2')
        self.S2 = tf.placeholder(tf.float32, [None, 180], 'S2')
        self.a_real = tf.placeholder(tf.float32, [None, 2], 'a_real')
        self.keep_prob = tf.placeholder(tf.float32)
        self.a_predict = self.build_c(self.S1,self.S2,self.keep_prob)
        #self.visuavalize = self.build_c2(self.S1,self.S2,self.keep_prob)
        self.loss = tf.losses.mean_squared_error(labels=self.a_real, predictions=self.a_predict)
        self.train = tf.train.AdamOptimizer(0.001).minimize(self.loss)
        self.loader()
        #self.data = np.loadtxt('scaled_house_data_2.dat')
        #self.data_2 = np.loadtxt('data3_old.dat')
        #self.data = np.loadtxt('scaled_house_data_annotated_2.dat')             #loading the data set which includes lidar data
        #self.data_2 = np.loadtxt('scaled_house_data_2.dat')
        self.initsess = tf.global_variables_initializer()


    def build_c(self,S1,S2,keep_prob):
        '''orth_init = tf.initializers.orthogonal(gain=np.sqrt(2))
        net1 = tf.compat.v1.layers.conv1d(S2, filters=32, kernel_size=19, strides=1, padding='valid', activation=tf.nn.relu, trainable=True,kernel_initializer=orth_init,name='net1')
        net2 = tf.compat.v1.layers.conv1d(net1, filters=32, kernel_size=8, strides=4, padding='valid', activation=tf.nn.relu, trainable=True,kernel_initializer=orth_init,name='net2')
        net3 = tf.compat.v1.layers.conv1d(net2, filters=64, kernel_size=4, strides=4, padding='valid', activation=tf.nn.relu, trainable=True,kernel_initializer=orth_init,name='net3')
        net4 = tf.compat.v1.layers.conv1d(net3, filters=64, kernel_size=3, strides=2, padding='valid', activation=tf.nn.relu, trainable=True,kernel_initializer=orth_init,name='net4')
        net4_flat = tf.reshape(net4,[-1,5*64])
        net5 = tf.layers.dense(net4_flat, 256, activation=tf.nn.relu, trainable=True,kernel_initializer=orth_init,name='net5')
        net6 = tf.layers.dense(net5, 128,activation=tf.nn.relu, trainable=True, name='net6')
        net7 = tf.layers.dense(net6, 64,activation=tf.nn.relu, trainable=True, name='net7')'''
        #net7 = tf.contrib.layers.layer_norm(net6, center=True, scale=True)
        #net8 = tf.nn.relu(net7)
        #net9_input = tf.concat([S1, net8], 1)

        net1 = tf.layers.dense(S2, 256, activation=tf.nn.relu, name='net1', trainable=True, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
        net2 = tf.layers.dense(net1, 512, activation=tf.nn.relu, name='net2', trainable=True, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
        net3 = tf.layers.dense(net2, 512, activation=tf.nn.relu, name='net3', trainable=True, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
        net4 = tf.layers.dense(net3, 256, activation=tf.nn.relu, name='net4', trainable=True, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
        net5 = tf.layers.dense(net4, 128, activation=tf.nn.relu, name='net5', trainable=True, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
        net6 = tf.layers.dense(net5, 64,trainable=True,activation=tf.nn.relu, name='net6')
        #net5_input = tf.concat([S1, net4], 1)

        net7 = tf.layers.dense(S1, 256, activation=tf.nn.relu, name='net7', trainable=True, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
        #drop_out5 = tf.nn.dropout(net5, keep_prob)
        net8 = tf.layers.dense(net7,256, activation=tf.nn.relu,name='net8',trainable=True, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
        #drop_out6 = tf.nn.dropout(net6, keep_prob)
        net9 = tf.layers.dense(net8,32, activation=tf.nn.relu,name='net9',trainable=True, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
        #drop_out7 = tf.nn.dropout(net7, keep_prob)

        net10_input = tf.concat([net9, net6], 1)
        net10 = tf.layers.dense(net10_input, 256, activation=tf.nn.relu, name='net10', trainable=True, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
        net11 = tf.layers.dense(net10, 128, activation=tf.nn.relu, name='net11', trainable=True, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
        #net13 = tf.layers.dense(net12, 2,trainable=True,activation=tf.nn.relu, name='net13')

        a = tf.layers.dense(net11,2,trainable=True,activation=tf.tanh)
        return tf.multiply(a, [0.5,1], name='scaled_a')

    '''def choose_action(self,s):
        a = self.sess.run(self.a_predict,feed_dict={self.S:s,self.keep_prob:1})
        return a[0]'''

    def train_net(self,j):
        data_n = int(5000+j*2000)
        print(data_n)
        self.data = np.loadtxt('scaled_house_data_1.dat')             #loading the data set which includes lidar data
        self.data_2 = np.loadtxt('scaled_house_data_2.dat')
        self.data_3 = np.loadtxt('scaled_house_data_2.dat')
        #self.data_2 = np.loadtxt('data3_old.dat')

        #self.train_set_1 = np.concatenate((self.data[:20094,:5],self.data[:20094,-26:-3]),axis=1)         #reducing the weights for the lidar data
        #self.train_set_1 = np.concatenate((self.train_set_1,self.data[:20094,5:28]),axis=1)           #reducing the weights for the lidar data
        #self.train_set_1 = np.concatenate((self.train_set_1,self.data[:20094,-2:]),axis=1)                  #adding the oputput part to the training data

        #self.train_set_1 = np.concatenate((self.data[:-2046,:-3],self.data[:-2046,5:23]),axis=1)
        self.train_set_1 = np.concatenate((self.data[:-2046,:-3],self.data[:-2046,-2:]),axis=1)
        self.train_set_2 = np.concatenate((self.data_2[:-2046,:-3],self.data_2[:-2046,-2:]),axis=1)
        self.train_set_3 = np.concatenate((self.data_3[:,:-3],self.data_3[:,-2:]),axis=1)

        #self.train_set = self.data[:-2046,:]

        #self.train_set_2 = np.concatenate((self.data[21094:31501,:5],self.data[21094:31501,-26:-3]),axis=1)       #This is the repetation of the same thing as above but for data with no obstracles
        #self.train_set_2 = np.concatenate((self.train_set_2,self.data[21094:31501,5:28]),axis=1)
        #self.train_set_2 = np.concatenate((self.train_set_2,self.data[21094:31501,-2:]),axis=1)

        #self.train_set_2 = np.concatenate((self.data[21094:31501,:-3],self.data[21094:31501,5:23]),axis=1)
        #self.train_set_2 = np.concatenate((self.train_set_2,self.data[21094:31501,-2:]),axis=1)

        #data_depth = self.data_2.shape[0]
        #print('data_depth:  ',data_depth)
        #self.train_set_2 = np.concatenate((self.data_2[:data_depth-1046,:-3],self.data_2[:data_depth-1046,5:23]),axis=1)
        #self.train_set_2 = np.concatenate((self.train_set_2,self.data_2[:data_depth-1046,-2:]),axis=1)

        #self.train_set = self.train_set_1

        #self.train_set_2 = np.concatenate((self.data[:5000,:28],self.data[:5000,-26:-3]),axis=1)
        #self.train_set_2 = np.concatenate((self.train_set_2,self.data[:5000,-2:]),axis=1)

        self.train_set =  np.concatenate((self.train_set_1,self.train_set_2),axis=0)
        self.train_set =  np.concatenate((self.train_set,self.train_set_3),axis=0)

        #self.train_set_final = self.train_set_2
        #self.train_set = self.train_set_final[:data_n,:]
        #self.train_set = self.train_set_1

        print('data_train:', self.train_set.shape)

        #self.text_set_1 = np.concatenate((self.data[20094:21094,:5],self.data[20094:21094,-26:-3]),axis=1)
        #self.text_set_1 = np.concatenate((self.text_set_1,self.data[20094:21094,5:28]),axis=1)
        #self.text_set_1 = np.concatenate((self.text_set_1,self.data[20094:21094,-2:]),axis=1)

        #self.text_set_1 = np.concatenate((self.data[-2046:,:-3],self.data[-2046:,5:23]),axis=1)
        self.text_set_1 = np.concatenate((self.data[-2046:,:-3],self.data[-2046:,-2:]),axis=1)
        self.text_set_2 = np.concatenate((self.data_2[-2046:,:-3],self.data_2[-2046:,-2:]),axis=1)
        #self.text_set_3 = np.concatenate((self.data_3[:,:-3],self.data_3[:,-2:]),axis=1)

        #self.text_set = self.data[-2046:,:]

        #self.text_set_2 = np.concatenate((self.data[31501:,:5],self.data[31501:,-26:-3]),axis=1)
        #self.text_set_2 = np.concatenate((self.text_set_2,self.data[31501:,5:28]),axis=1)
        #self.text_set_2 = np.concatenate((self.text_set_2,self.data[31501:,-2:]),axis=1)

        #self.text_set_2 = np.concatenate((self.data_2[-1046:,:-3],self.data_2[-1046:,5:23]),axis=1)
        #self.text_set_2 = np.concatenate((self.text_set_2,self.data_2[-1046:,-2:]),axis=1)

        #self.text_set = np.concatenate((self.data[31501:,:5],self.data[31501:,-2:]),axis=1)
        #self.text_set = np.concatenate((self.text_set,self.data_2[50001:]),axis=0)

        self.text_set =  np.concatenate((self.text_set_1,self.text_set_2),axis=0)
        #self.text_set =  np.concatenate((self.text_set,self.text_set_3),axis=0)
        #self.text_set = self.text_set_1

        print('data_test:',self.text_set.shape)
        train_set_x = self.train_set[:,0:-2]
        train_set_y = self.train_set[:,-2:]
        text_set_x = self.text_set[:,0:-2]
        text_set_y = self.text_set[:,-2:]
        MEMORY_CAPACITY = train_set_x.shape[0]
        MEMORY_CAPACITY_TEST = text_set_x.shape[0]
        BATCH_SIZE = 2046
        self.MSE = 1
        for i in range(10000):
            index = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
            index_test = np.random.choice(MEMORY_CAPACITY_TEST, size=BATCH_SIZE)
            bs = train_set_x[index,:]
            bs2 = bs[:,5:]
            #bs2 = bs2.reshape([-1,198,1])
            ba = train_set_y[index,:]

            bt = text_set_x         #[index_test,:]
            bt2 = bt[:,5:]
            #bt2 = bt2.reshape([-1,198,1])
            #bt_y = text_set_y       #[index_test,:]

            self.sess.run(self.train,feed_dict={self.S1:bs[:,:5],self.S2:bs2,self.a_real:ba,self.keep_prob:0.5})
            if i % 100 == 0:
                cost = self.sess.run(self.loss,feed_dict={self.S1:text_set_x[:,:5],self.S2:bt2,self.a_real:text_set_y,self.keep_prob:1})
                #costlist.append(cost)
                print("after %i iteration, MSE: %f" %(i, cost))
                if cost < self.MSE:
                    self.MSE = cost

    def loader(self):
        loader= tf.train.Saver()
        #loader.restore(self.sess,tf.train.latest_checkpoint('Human_model_exp_21'+str(j)))
        loader.restore(self.sess,tf.train.latest_checkpoint('Human_model_New_05'))

    def saver(self,j):
        saver = tf.train.Saver()
        #saver.save(self.sess,"Human_model_exp_2"+str(j)+"/net")
        saver.save(self.sess,"Human_model_New_08"+"/net")
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
    for i in range(1):
        Human.sess.run(Human.initsess)
        for j in range(n_iteration):
            Human.train_net(j)
            Human.saver(j)
            MSE[i,j]=Human.MSE
            np.savetxt("MSE3.dat",MSE)





if __name__ == '__main__':
    main()
