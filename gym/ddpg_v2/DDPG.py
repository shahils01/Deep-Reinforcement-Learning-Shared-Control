#!/usr/bin/env python3
import tensorflow as tf
import numpy as np

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound, GAMMA = 0.99,TAU = 0.0005, MEMORY_CAPACITY = int(5e5),BATCH_SIZE = 128,LR_A = 0.0001,LR_C = 0.001,):
        self.MEMORY_CAPACITY = MEMORY_CAPACITY
        self.BATCH_SIZE = BATCH_SIZE
        self.memory = np.zeros((self.MEMORY_CAPACITY, s_dim * 2 + a_dim + 2), dtype=np.float32)
        self.pointer = 0
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        self.sess = tf.Session(config=config)
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S1 = tf.placeholder(tf.float32, [None, 8], 's1')
        self.S1_ = tf.placeholder(tf.float32, [None, 8], 's1_')
        self.S2 = tf.placeholder(tf.float32, [None, 198,1], 's2')
        self.S2_ = tf.placeholder(tf.float32, [None, 198,1], 's2_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.done = tf.placeholder(tf.float32, [None, 1], 'done')

        self.a,self.a_pre = self._build_a(self.S1,self.S2,)
        q = self._build_c(self.S1,self.S2, self.a, )
        self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)          # soft replacement
        # self.global_step = tf.Variable(0, trainable=False)
        # self.LR_A = tf.compat.v1.train.exponential_decay(0.0001,self.global_step,100000, 1, staircase=True)
        # self.LR_C = tf.compat.v1.train.exponential_decay(0.001,self.global_step,100000, 1, staircase=True)

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(self.a_params), ema.apply(self.c_params)]      # soft update operation
        a_,_ = self._build_a(self.S1_,self.S2_, reuse=True, custom_getter=ema_getter)   # replaced target parameters
        q_ = self._build_c(self.S1_,self.S2_, a_, reuse=True, custom_getter=ema_getter)

        self.a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(self.a_loss, var_list=self.a_params)

        with tf.control_dependencies(target_update):    # soft replacement happened at here
            q_target = self.R + (1 - self.done) * GAMMA * q_
            self.td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(self.td_error, var_list=self.c_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s, mode):
        bs= s[np.newaxis, :]
        bs1 = bs[:, :8]
        bs2 = np.hstack((bs[:, 8:],bs[:, 8:26])).reshape([-1,198,1])
        if mode==1:
            a = self.sess.run(self.a, {self.S1: bs1,self.S2: bs2})[0]
            print("original a:",a)
        elif mode==2:
            a = self.sess.run(self.a_pre, {self.S1: bs1,self.S2: bs2})[0]
        return a

    def learn(self):
        if self.pointer>self.MEMORY_CAPACITY:
            indices = np.random.choice(self.MEMORY_CAPACITY, size=self.BATCH_SIZE)
        else:
            indices = np.random.choice(self.pointer, size=self.BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        bs1 = bs[:, :8]
        bs2 = np.hstack((bs[:, 8:],bs[:, 8:26])).reshape([-1,198,1])
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 2: -self.s_dim -1]
        bs_ = bt[:, -self.s_dim - 1: -1]
        bs1_ = bs_[:, : 8]
        bs2_ = np.hstack((bs_[:, 8:],bs_[:, 8:26])).reshape([-1,198,1])
        bdone = bt[:, -1][:,np.newaxis]

        self.sess.run(self.atrain, {self.S1: bs1,self.S2: bs2})
        self.sess.run(self.ctrain, {self.S1: bs1,self.S2: bs2, self.a: ba, self.R: br, self.S1_: bs1_, self.S2_: bs2_, self.done: bdone})

    def store_transition(self, s, a, r, s_, done):
        transition = np.hstack((s, a, [r], s_, done))
        index = self.pointer % self.MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1
        print(self.pointer)

    def _build_a(self, s1, s2, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        orth_init = tf.initializers.orthogonal(gain=np.sqrt(2))
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            net1 = tf.compat.v1.layers.conv1d(s2, filters=32, kernel_size=19, strides=1, padding='valid', activation=tf.nn.relu, trainable=trainable,kernel_initializer=orth_init,name='net1')
            net2 = tf.compat.v1.layers.conv1d(net1, filters=32, kernel_size=8, strides=4, padding='valid', activation=tf.nn.relu, trainable=trainable,kernel_initializer=orth_init,name='net2')
            net3 = tf.compat.v1.layers.conv1d(net2, filters=64, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu, trainable=trainable,kernel_initializer=orth_init,name='net3')
            net4 = tf.compat.v1.layers.conv1d(net3, filters=64, kernel_size=3, strides=1, padding='valid', activation=tf.nn.relu, trainable=trainable,kernel_initializer=orth_init,name='net4')
            net4_flat=tf.reshape(net4,[-1,19*64])
            net5=tf.layers.dense(net4_flat, 512, activation=tf.nn.relu, trainable=trainable,kernel_initializer=orth_init,name='net5')
            net6 = tf.layers.dense(net5, 64, trainable=trainable, name='net6')
            net6 = tf.contrib.layers.layer_norm(net6, center=True, scale=True)
            net6 = tf.nn.relu(net6)
            net7_input = tf.concat([s1, net6], 1)
            net7 = tf.layers.dense(net7_input, 64, trainable=trainable,name='net7')
            net7 = tf.contrib.layers.layer_norm(net7, center=True, scale=True)
            net7 = tf.nn.relu(net7)
            net7 = tf.layers.dense(net7, 16, trainable=trainable,name='net7_')
            net7 = tf.contrib.layers.layer_norm(net7, center=True, scale=True)
            net7 = tf.nn.relu(net7)
            net8 = tf.layers.dense(net7, self.a_dim, trainable=trainable,kernel_initializer=tf.random_uniform_initializer(minval=-3e-3,maxval=3e-3))
            a =tf.nn.softmax(net8)
            return a,net8

    def _build_c(self, s1, s2, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        orth_init = tf.initializers.orthogonal(gain=np.sqrt(2))
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            net1 = tf.compat.v1.layers.conv1d(s2, filters=32, kernel_size=19, strides=1, padding='valid', activation=tf.nn.relu, trainable=trainable,kernel_initializer=orth_init,name='net1')
            net2 = tf.compat.v1.layers.conv1d(net1, filters=32, kernel_size=8, strides=4, padding='valid', activation=tf.nn.relu, trainable=trainable,kernel_initializer=orth_init,name='net2')
            net3 = tf.compat.v1.layers.conv1d(net2, filters=64, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu, trainable=trainable,kernel_initializer=orth_init,name='net3')
            net4 = tf.compat.v1.layers.conv1d(net3, filters=64, kernel_size=3, strides=1, padding='valid', activation=tf.nn.relu, trainable=trainable,kernel_initializer=orth_init,name='net4')
            net4_flat=tf.reshape(net4,[-1,19*64])
            net5=tf.layers.dense(net4_flat, 608, activation=tf.nn.relu, trainable=trainable,kernel_initializer=orth_init,name='net5')
            net6 = tf.layers.dense(net5, 128, trainable=trainable, name='net6')
            net6 = tf.contrib.layers.layer_norm(net6, center=True, scale=True)
            net6 = tf.nn.relu(net6)
            net7_input = tf.concat([s1, a, net6], 1)
            net7 = tf.layers.dense(net7_input, 64, trainable=trainable,name='net7')
            net7 = tf.contrib.layers.layer_norm(net7, center=True, scale=True)
            net7 = tf.nn.relu(net7)
            net7 = tf.layers.dense(net7, 16, trainable=trainable,name='net7_')
            net7 = tf.contrib.layers.layer_norm(net7, center=True, scale=True)
            net7 = tf.nn.relu(net7)
            return tf.layers.dense(net7, 1, trainable=trainable,kernel_initializer=tf.random_uniform_initializer(minval=-3e-3,maxval=3e-3),name='q_val')  # Q(s,a)

    def saver(self):
        saver = tf.train.Saver()
        saver.save(self.sess,"final_net/net")
        print("*****net_saved******")

    def backupsaver(self):
        saver = tf.train.Saver()
        saver.save(self.sess,"backup_net/net")
        print("*****net_saved******")

    def loader(self):
        loader= tf.train.Saver()
        loader.restore(self.sess,tf.train.latest_checkpoint('final_net'))

    def backuploader(self):
        loader= tf.train.Saver()
        loader.restore(self.sess,tf.train.latest_checkpoint('backup_net'))
