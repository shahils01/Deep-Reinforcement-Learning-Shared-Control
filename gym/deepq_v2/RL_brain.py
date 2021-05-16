import numpy as np
import tensorflow as tf
import baselines.common.tf_util as U

np.random.seed(1)
tf.set_random_seed(1)


class DQN:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.001,
            reward_decay=0.99,
            e_greedy=0.99,
            replace_target_iter=1000,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            sess=None,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0.7 if e_greedy_increment is not None else self.epsilon_max
        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features*2+3))
        self.memory_counter = 0
        self._build_net()
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "target_net")
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "eval_net")
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess
        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)


    def _build_net(self):
        def build_layers(s1,s2, trainable,scp):
            orth_init = tf.initializers.orthogonal(gain=np.sqrt(2))
            with tf.variable_scope(scp):
                net1 = tf.compat.v1.layers.conv1d(s2, filters=32, kernel_size=19, strides=1, padding='valid', activation=tf.nn.relu, trainable=trainable,kernel_initializer=orth_init,name='net1')
                net2 = tf.compat.v1.layers.conv1d(net1, filters=32, kernel_size=8, strides=4, padding='valid', activation=tf.nn.relu, trainable=trainable,kernel_initializer=orth_init,name='net2')
                net3 = tf.compat.v1.layers.conv1d(net2, filters=64, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu, trainable=trainable,kernel_initializer=orth_init,name='net3')
                net4 = tf.compat.v1.layers.conv1d(net3, filters=64, kernel_size=3, strides=1, padding='valid', activation=tf.nn.relu, trainable=trainable,kernel_initializer=orth_init,name='net4')
                net4_flat=tf.reshape(net4,[-1,19*64])
                net5=tf.layers.dense(net4_flat, 608, activation=tf.nn.relu, trainable=trainable,kernel_initializer=orth_init,name='net5')
                net6 = tf.layers.dense(net5, 128, trainable=trainable, name='net6')
                net6 = tf.contrib.layers.layer_norm(net6, center=True, scale=True)
                net6 = tf.nn.relu(net6)
                net7_input = tf.concat([s1, net6], 1)
                net7 = tf.layers.dense(net7_input, 64, trainable=trainable,name='net7')
                net7 = tf.contrib.layers.layer_norm(net7, center=True, scale=True)
                net7 = tf.nn.relu(net7)
                net7 = tf.layers.dense(net7, 16, trainable=trainable,name='net7_')
                net7 = tf.contrib.layers.layer_norm(net7, center=True, scale=True)
                net7 = tf.nn.relu(net7)
                net8 = tf.layers.dense(net7, self.n_actions, trainable=trainable,kernel_initializer=tf.random_uniform_initializer(minval=-3e-3,maxval=3e-3))
            return net8

        # ------------------ build evaluate_net ------------------
        self.s1 = tf.placeholder(tf.float32, [None, 8], name='s1')  # input
        self.s2 = tf.placeholder(tf.float32, [None, 198,1], name='s2')
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        self.q_eval = build_layers(self.s1,self.s2, True,'eval_net')
        self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss,var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "eval_net"))

        # ------------------ build target_net ------------------
        self.s1_ = tf.placeholder(tf.float32, [None, 8], name='s_1')    # input
        self.s2_ = tf.placeholder(tf.float32, [None, 198,1], name='s_2')
        self.q_next = build_layers(self.s1_,self.s2_, False,'target_net')

    def store_transition(self, s, a, r, s_,done):
        transition = np.hstack((s, [a, r], s_,[float(done)]))
        '''
        st=str(type(s))
        at=str(type(a))
        rt=str(type(r))
        print("s type:", st)
        print("a type:", at)
        print("r type:", rt)
        print("transition:",+transition)
        '''
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1
        print(self.memory_counter)

    def choose_action(self, s, mode):
        bs= s[np.newaxis, :]
        bs1 = bs[:, :8]
        bs2 = np.hstack((bs[:, 8:],bs[:, 8:26])).reshape([-1,198,1])
        if mode==1:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s1: bs1,self.s2: bs2})
            action = np.argmax(actions_value)
        else:
            if np.random.uniform() < self.epsilon:  # choosing action
                actions_value = self.sess.run(self.q_eval, feed_dict={self.s1: bs1,self.s2: bs2})
                action = np.argmax(actions_value)
            else:
                action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        if self.memory_counter>self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        bs_ = batch_memory[:, -self.n_features-1:-1]
        bs1_ = bs_[:, : 8]
        bs2_ = np.hstack((bs_[:, 8:],bs_[:, 8:26])).reshape([-1,198,1])
        q_next = self.sess.run(self.q_next, feed_dict={self.s1_: bs1_,self.s2_: bs2_}) # next observation
        bs = batch_memory[:, :self.n_features]
        bs1 = bs[:, :8]
        bs2 = np.hstack((bs[:, 8:],bs[:, 8:26])).reshape([-1,198,1])
        q_eval = self.sess.run(self.q_eval, {self.s1: bs1,self.s2: bs2})

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
        q_target[batch_index, eval_act_index] = reward + (1-batch_memory[:, - 1])*self.gamma * np.max(q_next, axis=1)
        self.sess.run(self._train_op, feed_dict={self.s1: bs1,self.s2: bs2, self.q_target: q_target})
        self.epsilon = self.epsilon * self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        print("epsilon:",self.epsilon)
        self.learn_step_counter += 1

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
