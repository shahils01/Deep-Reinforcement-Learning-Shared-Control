import rospy
import numpy
import random
from gym import spaces
from openai_ros.robot_envs import turtlebot2_env
from gym.envs.registration import register
from geometry_msgs.msg import Point
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from openai_ros.openai_ros_common import ROSLauncher
from scipy.spatial.transform import Rotation as R
import os
from gazebo_msgs.srv import SpawnModel, DeleteModel
from geometry_msgs.msg import Pose
import tensorflow as tf
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import GetModelState
import time
from math import atan2
'''
with Manually preprocessed U and lidar readings
'''
class TurtleBot2HumanModelEnv(turtlebot2_env.TurtleBot2Env):
    def __init__(self):
        """
        This Task Env is designed for having the TurtleBot2 in some kind of maze.
        It will learn how to move around the maze without crashing.
        """

        # This is the path where the simulation files, the Task and the Robot gits will be downloaded if not there
        ros_ws_abspath = rospy.get_param("/turtlebot2/ros_ws_abspath", None)
        assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath in your yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        assert os.path.exists(ros_ws_abspath), "The Simulation ROS Workspace path " + ros_ws_abspath + \
                                               " DOESNT exist, execute: mkdir -p " + ros_ws_abspath + \
                                               "/src;cd " + ros_ws_abspath + ";catkin_make"

        ROSLauncher(rospackage_name="turtlebot_gazebo",
                    launch_file_name="start_train_world.launch",
                    ros_ws_abspath=ros_ws_abspath)

        # Load Params from the desired Yaml file
        LoadYamlFileParamsTest(rospackage_name="openai_ros",
                               rel_path_from_package_to_file="src/openai_ros/task_envs/turtlebot2/config",
                               yaml_file_name="turtlebot2_goal_continuous_humanmodel.yaml")

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(TurtleBot2HumanModelEnv, self).__init__(ros_ws_abspath)

        # Only variable needed to be set here

        high = numpy.array([1,1,1,1])
        low = numpy.array([-1,-1,-1,-1])

        #high = numpy.array([0,1,1,1])
        #low = numpy.array([0,-1,-1,-1])

        self.action_space = spaces.Box(low, high)

        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-numpy.inf, numpy.inf)

        self.success = True
        #number_observations = rospy.get_param('/turtlebot2/n_observations')
        """
        We set the Observation space for the 6 observations
        cube_observations = [
            round(current_disk_roll_vel, 0),
            round(y_distance, 1),
            round(roll, 1),
            round(pitch, 1),
            round(y_linear_speed,1),
            round(yaw, 1),
        ]
        """

        # Actions and Observations

        self.init_linear_forward_speed = rospy.get_param('/turtlebot2/init_linear_forward_speed')
        self.init_linear_turn_speed = rospy.get_param('/turtlebot2/init_linear_turn_speed')

        self.new_ranges = 180
        self.min_range = rospy.get_param('/turtlebot2/min_range')
        self.max_laser_value = rospy.get_param('/turtlebot2/max_laser_value')
        self.min_laser_value = rospy.get_param('/turtlebot2/min_laser_value')

        # Get Desired Point to Get
        self.desired_point = Point()
        self.desired_point.x = rospy.get_param("/turtlebot2/desired_pose/x")
        self.desired_point.y = rospy.get_param("/turtlebot2/desired_pose/y")
        self.desired_point.z = rospy.get_param("/turtlebot2/desired_pose/z")

        self.state_msg = ModelState()
        self.state_msg.model_name = 'mobile_base'
        self.state_msg.pose.position.x = 0
        self.state_msg.pose.position.y = 0
        self.state_msg.pose.position.z = 0
        self.state_msg.pose.orientation.x = 0
        self.state_msg.pose.orientation.y = 0
        self.state_msg.pose.orientation.z = 0
        self.state_msg.pose.orientation.w = 0
        # We create two arrays based on the binary values that will be assigned
        # In the discretization method.
        laser_scan = self.get_laser_scan()
        #print("lidar data:", laser_scan)
        rospy.logdebug("laser_scan len===>" + str(len(laser_scan.ranges)))


        #high = numpy.array([0.5,1,1,1,1,1,6,3.14])#,numpy.array([12,6,3.14,1,3.14,0.5,1]),6*numpy.ones([self.new_ranges]),numpy.array([12,6,3.14,1,3.14,0.5,1]),6*numpy.ones([self.new_ranges])))


        high = numpy.hstack((numpy.array([0.5,1,0.5,1]),6*numpy.ones([self.new_ranges])))
        #high = numpy.hstack((numpy.array([0,1,0.5,1]),6*numpy.ones([self.new_ranges])))

        #high = numpy.hstack((numpy.array([1,1]),numpy.ones([self.new_ranges]),numpy.array([1,1]),numpy.ones([self.new_ranges]),numpy.array([1,1]),numpy.ones([self.new_ranges])))
        #low = numpy.array([-0.5,-1,-1,-1,-1,-1, 0,-3.14])#,numpy.array([-1,-1*6,-1*3.14,-1,-3.14,-0.5,-1]),numpy.zeros([self.new_ranges]),numpy.array([-1,-1*6,-1*3.14,-1,-3.14,-0.5,-1]),numpy.zeros([self.new_ranges])))

        low = numpy.hstack((numpy.array([-0.5,-1,-0.5,-1]),numpy.zeros([self.new_ranges])))

        #low = numpy.hstack((numpy.array([0,-1,-0.5,-1]),numpy.zeros([self.new_ranges])))

        #low = numpy.hstack((numpy.array([-1,-1]),numpy.zeros([self.new_ranges]),numpy.array([1,1]),numpy.ones([self.new_ranges]),numpy.array([1,1]),numpy.ones([self.new_ranges])))
        # We only use two integers
        self.observation_space = spaces.Box(low, high)

        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))

        # Rewards
        self.forwards_reward = rospy.get_param("/turtlebot2/forwards_reward")
        self.turn_reward = rospy.get_param("/turtlebot2/turn_reward")
        self.end_episode_points = rospy.get_param("/turtlebot2/end_episode_points")

        self.cumulated_steps = 0.0

        ############################## goal ##############################################
        self.goal_position = Pose()
        self.f = open('/home/i2rlab/shahil_files/shahil_RL_ws_new/src/turtlebot/turtlebot_gazebo/worlds/goal/model.sdf','r')
        self.sdff = self.f.read()


        ############################## Obstacle ##########################################
        self.angle=numpy.linspace(-179,179,180)/180*numpy.pi
        self.cos = numpy.cos(self.angle)
        self.sin = numpy.sin(self.angle)
        ############################## Human Model ######################################
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        self.sess = tf.Session(config=config)
        self.S1 = tf.placeholder(tf.float32, [None, 5], 'S1')
        self.S2 = tf.placeholder(tf.float32, [None, 180], 'S2')
        self.keep_prob = tf.placeholder(tf.float32)
        self.a_predict = self.build_c(self.S1,self.S2,self.keep_prob)
        self.loader()
        self.goal_space()
        #self.xy = numpy.array([-8.1,-7.0])

    def loader(self):
        loader= tf.train.Saver()
        loader.restore(self.sess,tf.train.latest_checkpoint('Human_model_3'))

    def build_c(self,S1,S2,keep_prob):
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

    def choose_action(self,s):
        bs1 = s[:, :5]
        #bs2 = numpy.hstack((s[:, 5:],s[:, -18:])).reshape([-1,198,1])
        bs2 = s[:,5:]
        a = self.sess.run(self.a_predict, {self.S1: bs1,self.S2:bs2,self.keep_prob:1})
        print("original a:",a)
        #return a
        return a[0]
        ################################################################################



    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self.move_base( self.init_linear_forward_speed,
                        self.init_linear_turn_speed,
                        epsilon=0.05,
                        update_rate=10)

        return True


    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # For Info Purposes
        self.cumulated_reward = 0.0
        # Set to false Done, because its calculated asyncronously
        self._episode_done = False
        self._outofrange = False
        #self.desired_point.x = random.uniform(1,10)
        #self.desired_point.y = random.uniform(-5,5)

        try:
            self.deleteModel()
        except:
            pass
        self.respawnModel()
        #self.get_statemsg()
        self.moveto()
        #time.sleep(2)
        #self.obstaclemoveto()
        odometry = self.get_odom()
        self.previous_distance_from_des_point = self.get_distance_from_desired_point(odometry.pose.pose.position)
        self.prev_position = [-1]
        self.prev_u_gtg = -1
        self.last_action = -1
        self.u_h=numpy.array([0,0])

    def _set_action(self, ratio):
        """
        """
        ratio=list(ratio)
        rospy.logdebug("Start Set Action ==>"+str(ratio))
        print("ratio:",ratio)

        linear_speed = ratio[0]*self.joy_linear+ratio[1]*self.a_ao[0,0]+ratio[2]*self.a_c[0,0]+ratio[3]*self.a_cc[0,0]
        angular_speed = ratio[0]*self.joy_angular+ratio[1]*self.a_ao[1,0]+ratio[2]*self.a_c[1,0]+ratio[3]*self.a_cc[1,0]

        #linear_speed = self.joy_linear
        #angular_speed = self.joy_angular

        #linear_speed = ratio[1]*self.a_ao[0,0]+ratio[2]*self.a_c[0,0]+ratio[3]*self.a_cc[0,0]
        #angular_speed = ratio[1]*self.a_ao[1,0]+ratio[2]*self.a_c[1,0]+ratio[3]*self.a_cc[1,0]

        '''lidar = numpy.array(self.get_laser_scan().ranges)
        n = numpy.argmin(lidar)
        ao_critical_angle = 20

        odometry = self.get_odom()
        x_position = odometry.pose.pose.position.x
        y_position = odometry.pose.pose.position.y
        base_orientation_quat = odometry.pose.pose.orientation
        base_roll, base_pitch, base_yaw = self.get_orientation_euler(base_orientation_quat)

        xdiff = self.desired_point.x - x_position
        ydiff = self.desired_point.y - y_position

        angle_to_goal = atan2(ydiff,xdiff)
        angle_to_obs = 2*n
        print('angle to obs:',angle_to_obs)

        obs_dist = lidar[n]'''
        '''if abs(angle_to_goal - base_yaw) > 0.5:
            linear_speed = 0.0
            angular_speed = 0.5
        else:
            if obs_dist > 1.1:
                linear_speed = 2
                angular_speed = 0.0
            elif obs_dist < 1.1 and obs_dist > 0.6:
                linear_speed = ratio[0]*(0)+ratio[1]*self.a_ao[0,0]+ratio[2]*self.a_c[0,0]+ratio[3]*self.a_cc[0,0]
                angular_speed = ratio[0]*(0)+ratio[1]*self.a_ao[1,0]+ratio[2]*self.a_c[1,0]+ratio[3]*self.a_cc[1,0]
            elif obs_dist < 0.6:
                linear_speed = self.a_ao[0,0]
                angular_speed = self.a_ao[1,0]'''

        '''if obs_dist > 1.1:
            if abs(angle_to_goal - base_yaw) > 0.5:
                linear_speed = 0.0
                angular_speed = 0.5
            else:
                linear_speed = 2
                angular_speed = 0.0
        elif obs_dist < 1.1 and obs_dist > 0.6: #and (angle_to_obs < 90 and angle_to_obs > 270):
            linear_speed = 0.0
            angular_speed = 0.0
            if angle_to_obs < 40 and angle_to_obs > 320:
                linear_speed = self.a_ao[0,0]
                angular_speed = self.a_ao[1,0]
            elif angle_to_obs > 20:
                linear_speed = self.a_c[0,0]
                angular_speed = self.a_c[1,0]
            elif angle_to_obs > 340:
                linear_speed = self.a_cc[0,0]
                angular_speed = self.a_cc[1,0]
        elif obs_dist <= 0.6:
            linear_speed = self.a_ao[0,0]
            angular_speed = self.a_ao[1,0]'''
        '''else:
            if abs(angle_to_goal - base_yaw) > 0.5:
                linear_speed = 0.0
                angular_speed = 0.5
            else:
                linear_speed = 0.8
                angular_speed = 0.0'''

        #angular_speed = 0.6*self.theta_dot+0.4*angular_speed
        # We tell TurtleBot2 the linear and angular speed to set to execute
        self.move_base(linear_speed, angular_speed, epsilon=0.05, update_rate=10)
        self.u_b = numpy.array([linear_speed,angular_speed])
        rospy.logdebug("END Set Action ==>"+str(ratio))

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        TurtleBot2Env API DOCS
        :return:
        """
        #self.obstaclemoveto()
        rospy.logdebug("Start Get Observation ==>")
        # We get the laser scan data
        laser_scan = self.get_laser_scan()
        #print("lidar data:", laser_scan)

        discretized_laser_scan = self.discretize_observation( laser_scan,                       #discretized lidar data
                                                                self.new_ranges
                                                                )
        # We get the odometry so that SumitXL knows where it is.
        odometry = self.get_odom()
        #joy = self.get_joy()
        x_position = odometry.pose.pose.position.x
        y_position = odometry.pose.pose.position.y
        base_orientation_quat = odometry.pose.pose.orientation
        base_roll, base_pitch, base_yaw = self.get_orientation_euler(base_orientation_quat)
        print('base_yaw:',base_yaw)
        v = odometry.twist.twist.linear.x
        self.theta_dot = odometry.twist.twist.angular.z
        xdiff = self.desired_point.x - x_position
        ydiff = self.desired_point.y - y_position
        #observations = [round(base_yaw, 2),round(v, 2),round(self.theta_dot, 2),round(xdiff, 2),round(ydiff, 2)]#+ discretized_laser_scan
        observations = [base_yaw,v,self.theta_dot,xdiff,ydiff]
        observations = numpy.append(observations,numpy.array(discretized_laser_scan))
        #observations = numpy.append(observations,numpy.array(discretized_laser_scan[:18]))
        #observations = numpy.append(observations,discretized_laser_scan[0])
        s = numpy.array([observations])
        a = self.choose_action(s)
        self.joy_linear = a[0]
        print(self.joy_linear)
        self.joy_angular = a[1]
        tran = numpy.array([[numpy.cos(0),-numpy.sin(0)],[numpy.sin(0),numpy.cos(0)]]).dot(numpy.array([[1,0],[0,0.115]]))
        self.u_gtg = tran.dot(numpy.array([[self.joy_linear],[self.joy_angular]]))
        self.obstacle_avoidance()
        # We round to only two decimals to avoid very big Observation space
        ################################################  Change Observations  ########################################################################################
        observations = [round(x_position, 2),round(y_position, 2),round(base_yaw, 2),round(v, 2),round(self.theta_dot, 2),round(self.joy_linear, 2),round(self.joy_angular, 2)]+discretized_laser_scan+[self.e_norm, round(self.theta_bt_uh_ob,2)]
        #print(observations)
        #print("Observations==>"+observations)
        rospy.logdebug("Observations==>"+str(observations))
        rospy.logdebug("END Get Observation ==>")
        return observations

    def obstacle_avoidance(self):
        lidar = numpy.array(self.get_laser_scan().ranges)
        #lidar = self.discretize_observation( lidar,self.new_ranges)
        # odometry = self.get_odom()
        # base_orientation_quat = odometry.pose.pose.orientation
        # base_roll, base_pitch, yaw = self.get_orientation_euler(base_orientation_quat)
        yaw = 0
        ################################## nearest ######################################################
        n = numpy.argmin(lidar)                     # gives angle of the obstracle
        if n<=89:
            delta_angle = ((n-89)*2-1)/180*numpy.pi
        else:
            delta_angle = ((n-90)*2+1)/180*numpy.pi
        orientation = yaw + delta_angle
        self.e_norm = lidar[n]
        e = numpy.array([[-self.e_norm*numpy.cos(orientation)],[-self.e_norm*numpy.sin(orientation)]])
        ################################## weighted #######################################################
        # n = numpy.argmin(lidar)
        # if lidar[n]>1.1:
        #     if n<=89:
        #         delta_angle = ((n-89)*2-1)/180*numpy.pi
        #     else:
        #         delta_angle = ((n-90)*2+1)/180*numpy.pi
        #     orientation = yaw + delta_angle
        #     self.e_norm = lidar[n]
        #     e = numpy.array([[-self.e_norm*numpy.cos(orientation)],[-self.e_norm*numpy.sin(orientation)]])
        # else:
        #     lidar_power = numpy.power(lidar, -2)
        #     lidar_power[numpy.argwhere(lidar>1.1)]=0
        #     lidar_power_sum = numpy.sum(lidar_power)
        #     weight=lidar_power/lidar_power_sum
        #     e_robot = numpy.array([numpy.sum(self.cos*lidar*weight),numpy.sum(self.sin*lidar*weight)])
        #     self.e_norm = numpy.linalg.norm(e_robot)
        #     orientation = numpy.arctan2(e_robot[1],e_robot[0])+yaw
        #     e = numpy.array([[-self.e_norm*numpy.cos(orientation)],[-self.e_norm*numpy.sin(orientation)]])
        # print(self.e_norm)
        ########################################### Controller velocities ########################################################

        #u_ao = 0.5/self.e_norm*(1/(self.e_norm**2+0.1))*e       # obstracle avoidance controller
        #u_c = 0.5*numpy.array([[0,1],[-1,0]]).dot(u_ao)         # Follow wall clockwise controller
        #u_cc =0.5*numpy.array([[0,-1],[1,0]]).dot(u_ao)         # Follow wall counter clockwise controller

        u_ao = 0.5/self.e_norm*(1/(self.e_norm**2+0.1))*e
        u_c = 0.5*numpy.array([[0,1],[-1,0]]).dot(u_ao)
        u_cc = 0.5*numpy.array([[0,-1],[1,0]]).dot(u_ao)

        self.a_ao = numpy.array([[1,0],[0,1/0.115]]).dot(numpy.array([[numpy.cos(-yaw),-numpy.sin(-yaw)],[numpy.sin(-yaw),numpy.cos(-yaw)]])).dot(u_ao)
        self.a_c = numpy.array([[1,0],[0,1/0.115]]).dot(numpy.array([[numpy.cos(-yaw),-numpy.sin(-yaw)],[numpy.sin(-yaw),numpy.cos(-yaw)]])).dot(u_c)
        self.a_cc = numpy.array([[1,0],[0,1/0.115]]).dot(numpy.array([[numpy.cos(-yaw),-numpy.sin(-yaw)],[numpy.sin(-yaw),numpy.cos(-yaw)]])).dot(u_cc)
        print("u_ao:",u_ao)
        self.limitu()
        self.theta_bt_uh_ob = abs(numpy.arctan2(self.u_gtg[1,0],self.u_gtg[0,0])-orientation)
        if self.theta_bt_uh_ob > numpy.pi:
            self.theta_bt_uh_ob = 2 * numpy.pi - self.theta_bt_uh_ob
        # self.e_norm = lidar[n]
        #####################################potantial field mathod##############################################################
        # n = numpy.argmin(lidar)
        # if lidar[n]>1.3:
        #     if n<=89:
        #         delta_angle = ((n-89)*2-1)/180*numpy.pi
        #     else:
        #         delta_angle = ((n-90)*2+1)/180*numpy.pi
        #     orientation = yaw + delta_angle
        #     self.e_norm = lidar[n]
        #     e = numpy.array([[-self.e_norm*numpy.cos(orientation)],[-self.e_norm*numpy.sin(orientation)]])
        #     print(self.e_norm)
        #     u_ao = 0.5/self.e_norm*(1/(self.e_norm**2+0.1))*e
        #     u_c = 0.5*numpy.array([[0,1],[-1,0]]).dot(u_ao)
        #     u_cc =0.5*numpy.array([[0,-1],[1,0]]).dot(u_ao)
        #     self.a_ao = numpy.array([[1,0],[0,1/0.115]]).dot(numpy.array([[numpy.cos(-yaw),-numpy.sin(-yaw)],[numpy.sin(-yaw),numpy.cos(-yaw)]])).dot(u_ao)
        #     self.a_c = numpy.array([[1,0],[0,1/0.115]]).dot(numpy.array([[numpy.cos(-yaw),-numpy.sin(-yaw)],[numpy.sin(-yaw),numpy.cos(-yaw)]])).dot(u_c)
        #     self.a_cc = numpy.array([[1,0],[0,1/0.115]]).dot(numpy.array([[numpy.cos(-yaw),-numpy.sin(-yaw)],[numpy.sin(-yaw),numpy.cos(-yaw)]])).dot(u_cc)
        #     self.limitu()
        #     self.theta_bt_uh_ob = abs(numpy.arctan2(self.u_gtg[1,0],self.u_gtg[0,0])-orientation)
        #     if self.theta_bt_uh_ob > numpy.pi:
        #         self.theta_bt_uh_ob = 2 * numpy.pi - self.theta_bt_uh_ob
        # else:
        #     orientation = yaw + self.angle
        #     e = numpy.array([-lidar*numpy.cos(orientation),-lidar*numpy.sin(orientation)])
        #     K = 0.5/lidar*(1/(numpy.power(lidar, 2)+0.1))
        #     K[numpy.argwhere(lidar>1.3)]=0
        #     u_ao = numpy.sum(K*e,axis=1).reshape(2,1)
        #     u_c = 0.5*numpy.array([[0,1],[-1,0]]).dot(u_ao)
        #     u_cc =0.5*numpy.array([[0,-1],[1,0]]).dot(u_ao)
        #     self.a_ao = numpy.array([[1,0],[0,1/0.115]]).dot(numpy.array([[numpy.cos(-yaw),-numpy.sin(-yaw)],[numpy.sin(-yaw),numpy.cos(-yaw)]])).dot(u_ao)
        #     self.a_c = numpy.array([[1,0],[0,1/0.115]]).dot(numpy.array([[numpy.cos(-yaw),-numpy.sin(-yaw)],[numpy.sin(-yaw),numpy.cos(-yaw)]])).dot(u_c)
        #     self.a_cc = numpy.array([[1,0],[0,1/0.115]]).dot(numpy.array([[numpy.cos(-yaw),-numpy.sin(-yaw)],[numpy.sin(-yaw),numpy.cos(-yaw)]])).dot(u_cc)
        #     self.limitu()
        #     self.theta_bt_uh_ob = abs(numpy.arctan2(self.u_gtg[1,0],self.u_gtg[0,0])-numpy.arctan2(-u_ao[1,0],-u_ao[0,0]))
        #     if self.theta_bt_uh_ob > numpy.pi:
        #         self.theta_bt_uh_ob = 2 * numpy.pi - self.theta_bt_uh_ob
        #     self.e_norm = lidar[n]

    def limitu(self,):
        if abs(self.a_ao[0,0])>0.5 or abs(self.a_ao[1,0])>0.3:
            self.a_ao =self.a_ao/max(abs(self.a_ao/[[0.5],[0.3]]))
        if abs(self.a_c[0,0]>0.5) or abs(self.a_c[1,0])>0.4:
            self.a_c =self.a_c/max(abs(self.a_c/[[0.5],[0.4]]))
        if abs(self.a_cc[0,0])>0.5 or abs(self.a_cc[1,0])>0.4:
            self.a_cc =self.a_cc/max(abs(self.a_cc/[[0.5],[0.4]]))
        #print(self.a_ao,self.a_c,self.a_cc)


    def _is_done(self, observations):
        lidar = numpy.array(self.get_laser_scan().ranges)
        n = numpy.argmin(lidar)
        print("Distance from obs:",lidar[n])
        #if self._episode_done:
        if lidar[n] < 0.3:
            rospy.logerr("TurtleBot2 is Too Close to wall==>")
        else:
            #rospy.logerr("TurtleBot2 didnt crash at least ==>")
            current_position = Point()
            current_position.x = observations[0]
            current_position.y = observations[1]
            #current_v = observations[3]
            #current_thetadot = observations[4]
            current_position.z = 0.0
            MAX_X = 10
            MIN_X = -10
            MAX_Y = 10
            MIN_Y = -10

            # We see if we are outside the Learning Space

            if current_position.x <= MAX_X and current_position.x > MIN_X:
                if current_position.y <= MAX_Y and current_position.y > MIN_Y:
                    rospy.logdebug("TurtleBot Position is OK ==>["+str(current_position.x)+","+str(current_position.y)+"]")

                    # We see if it got to the desired point
                    if self.is_in_desired_position(current_position):
                        #if current_v == 0 and current_thetadot == 0:
                            self._episode_done = True


                else:
                    rospy.logerr("TurtleBot to Far in Y Pos ==>"+str(current_position.x))
                    self._episode_done = True
                    self._outofrange = True
            else:
                rospy.logerr("TurtleBot to Far in X Pos ==>"+str(current_position.x))
                self._episode_done = True
                self._outofrange = True

        return self._episode_done

    def _compute_reward(self, observations, done):

        lidar = numpy.array(self.get_laser_scan().ranges)
        current_position = Point()
        current_position.x = observations[0]
        current_position.y = observations[1]
        current_position.z = 0.0
        min_distance = observations[-2]
        current_yaw = observations[2]

        ################    ##################################
        xdiff = self.desired_point.x - observations[0]
        ydiff = self.desired_point.y - observations[1]

        distance_from_des_point = self.get_distance_from_desired_point(current_position)
        #print("distance to des", + distance_from_des_point)
        distance_difference =  distance_from_des_point - self.previous_distance_from_des_point
        theta = numpy.arctan2(ydiff,xdiff)
        d_theta = abs(theta - current_yaw)
        if d_theta > numpy.pi:
            d_theta = 2 * numpy.pi - d_theta
        reward = 0
        ######################################################
        if not done:
            if distance_difference < 0.0:
                #rospy.logwarn("DECREASE IN DISTANCE GOOD")
                reward = -20 * abs(numpy.pi - d_theta) * distance_difference#-30*(1-numpy.tanh(4.5*(min_distance-0.5)))
                #reward = -5
            else:
                reward = -5
            # reward_sim = (numpy.linalg.norm(self.u_h-self.u_b))**2
            # if reward_sim > 2:
            #     reward -= reward_sim
            if min_distance < 0.8:
                reward = -5/min_distance
            self.success = True
        else:
            n = numpy.argmin(lidar)
            obs_dist = lidar[n]

            if self.is_in_desired_position(current_position):
                #reward = 0
                reward = 1000
                self.success = True
            elif self._outofrange:
                reward = -1000
                self.success = False
            else:
                reward = -500
                self.success = True
        self.u_h=numpy.array([self.joy_linear,self.joy_angular])
        print("reward:",+ reward)
        self.previous_distance_from_des_point = distance_from_des_point
        rospy.logdebug("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))
        return round(reward,2)


    # Internal TaskEnv Methods

    def discretize_observation(self,data,new_ranges):
        """
        Discards all the laser readings that are not multiple in index of new_ranges
        value.
        """
        self._episode_done = False

        discretized_ranges = []
        mod = len(data.ranges)/new_ranges

        rospy.logdebug("data=" + str(data))
        rospy.logwarn("new_ranges=" + str(new_ranges))
        rospy.logwarn("mod=" + str(mod))

        for i, item in enumerate(data.ranges):
            if (i%mod==0):
                if item == float ('Inf') or numpy.isinf(item):
                    discretized_ranges.append(self.max_laser_value)
                elif numpy.isnan(item):
                    discretized_ranges.append(self.min_laser_value)
                else:
                    discretized_ranges.append(round(item,3))

                if (self.min_range > item > 0):
                    #rospy.logerr("done Validation >>> item=" + str(item)+"< "+str(self.min_range))
                    self._episode_done = True
                #else:
                    #rospy.logwarn("NOT done Validation >>> item=" + str(item)+"> "+str(self.min_range))


        return discretized_ranges


    def is_in_desired_position(self,current_position, epsilon=0.5):
        """
        It return True if the current position is similar to the desired poistion
        """

        is_in_desired_pos = False
        is_in_desired_pos = self.get_distance_from_desired_point(current_position) <= epsilon
        '''
        x_pos_plus = self.desired_point.x + epsilon
        x_pos_minus = self.desired_point.x - epsilon
        y_pos_plus = self.desired_point.y + epsilon
        y_pos_minus = self.desired_point.y - epsilon

        x_current = current_position.x
        y_current = current_position.y

        x_pos_are_close = (x_current <= x_pos_plus) and (x_current > x_pos_minus)
        y_pos_are_close = (y_current <= y_pos_plus) and (y_current > y_pos_minus)

        is_in_desired_pos = x_pos_are_close and y_pos_are_close
        '''
        return is_in_desired_pos


    def get_distance_from_desired_point(self, current_position):
        """
        Calculates the distance from the current position to the desired point
        :param start_point:
        :return:
        """
        distance = self.get_distance_from_point(current_position,
                                                self.desired_point)

        return distance

    def get_distance_from_point(self, pstart, p_end):
        """
        Given a Vector3 Object, get distance from current position
        :param p_end:
        :return:
        """
        a = numpy.array((pstart.x, pstart.y, pstart.z))
        b = numpy.array((p_end.x, p_end.y, p_end.z))

        distance = numpy.linalg.norm(a - b)

        return distance

    def get_orientation_euler(self, quaternion_vector):
        # We convert from quaternions to euler
        orientation_list = [quaternion_vector.x,
                            quaternion_vector.y,
                            quaternion_vector.z,
                            quaternion_vector.w]

        r = R.from_quat(orientation_list)
        roll, pitch, yaw = r.as_rotvec()
        return roll, pitch, yaw

    def respawnModel(self):
        if self.success:
            #self.obstacle2_msg.pose.position.x= random.uniform(4.5,7.5)
            #lw = 1.5-abs(self.obstacle2_msg.pose.position.x-6)
            #self.obstacle2_msg.pose.position.y= random.uniform(-lw,lw)
            n = random.randint(0,self.xy.shape[0]-1)
            self.desired_point.x = self.xy[n,0]
            self.desired_point.y = self.xy[n,1]
            '''
            ob_position=numpy.array([[6.94,-3.8],[5.55,0.41],[6.13,2.66],[3.21,3.98],[2.33,0.95],[1.57,0.11],[3.3,-2.1],[0.63,-4.26],[-0.21,-3.41]])
            while (abs(ob_position-numpy.array([self.desired_point.x,self.desired_point.y]))<2).any():
                self.desired_point.x = random.uniform(0,8)
                self.desired_point.y = random.uniform(-4.5,4.5)
                print("try new goal position")
            '''
        rospy.wait_for_service('gazebo/spawn_sdf_model')
        self.goal_position.position.x = self.desired_point.x
        self.goal_position.position.y = self.desired_point.y
        spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
        spawn_model_prox("goal", self.sdff, "", self.goal_position, "world")
        #spawn_model_prox("obstacle1", self.obstacle, "", self.obstacle_position_1, "world")
        #spawn_model_prox("obstacle2", self.obstacle2, "", self.obstacle_position_2, "world")

    def deleteModel(self):
        rospy.wait_for_service('gazebo/delete_model')
        del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
        del_model_prox("goal")
        #del_model_prox("obstacle1")
        #del_model_prox("obstacle2")

    def moveto(self):
        rospy.wait_for_service('/gazebo/set_model_state')
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_statemsg()
        set_state(self.state_msg )

    def get_statemsg(self):
        if self.success:
            n = random.randint(0,self.xy.shape[0]-1)
            self.state_msg.pose.position.x = self.xy[n,0]
            self.state_msg.pose.position.y = self.xy[n,1]
            goal_position=numpy.array([self.desired_point.x,self.desired_point.y])
            while numpy.linalg.norm(goal_position-numpy.array([self.state_msg.pose.position.x,self.state_msg.pose.position.y]))<5:
                n = random.randint(0,self.xy.shape[0]-1)
                self.state_msg.pose.position.x = self.xy[n,0]
                self.state_msg.pose.position.y = self.xy[n,1]
                print("try new robot position")
            D = random.uniform(0,360)
            r = R.from_euler('z', D, degrees=True)
            self.state_msg.pose.orientation.x, self.state_msg.pose.orientation.y, self.state_msg.pose.orientation.z, self.state_msg.pose.orientation.w = r.as_quat()

    def obstaclemoveto(self):
        ################### obstacle 1##############################################################################
        #self.obstacle1_msg.pose.orientation.x = 1
        rospy.wait_for_service('/gazebo/get_model_state')
        gms = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        resp1 = gms('obstacle1',"world")
        resp2 = gms('mobile_base',"world")
        position = numpy.array([resp1.pose.position.x,resp1.pose.position.y])
        position_m = numpy.array([resp2.pose.position.x,resp2.pose.position.y])
        self.obstacle1_msg.pose.position = resp1.pose.position
        if resp1.pose.position.x<=2 and resp1.twist.linear.x<=0:
            if position_m[1]>=0:
                g=3
            else:
                g=2
            if position_m[0]>=4:
                speed_value = 1.3
            else:
                speed_value = numpy.random.uniform(0.4,0.7)
            speed = speed_value*(self.A[g]-position)/numpy.linalg.norm((self.A[g]-position))
            self.obstacle1_msg.twist.linear.x = speed[0]
            self.obstacle1_msg.twist.linear.y = speed[1]
        elif resp1.pose.position.x>=10 and resp1.twist.linear.x>=0:
            if position_m[1]>=0:
                g=3
            else:
                g=2
            if position_m[0]<=4:
                speed_value = 1.3
            else:
                speed_value = numpy.random.uniform(0.4,0.7)
            speed = speed_value*(self.A[g]-position)/numpy.linalg.norm((self.A[g]-position))
            self.obstacle1_msg.twist.linear.x = speed[0]
            self.obstacle1_msg.twist.linear.y = speed[1]
        elif resp1.pose.position.y<=-4 and resp1.twist.linear.y<=0:
            if position_m[0]<=4:
                g=0
            else:
                g=1
            if position_m[1]>=0:
                speed_value = 1.3
            else:
                speed_value = numpy.random.uniform(0.4,0.7)
            speed = speed_value*(self.A[g]-position)/numpy.linalg.norm((self.A[g]-position))
            self.obstacle1_msg.twist.linear.x = speed[0]
            self.obstacle1_msg.twist.linear.y = speed[1]
        elif resp1.pose.position.y>=4 and resp1.twist.linear.y>=0:
            if position_m[0]<=4:
                g=0
            else:
                g=1
            if position_m[1]<=0:
                speed_value = 1.3
            else:
                speed_value = numpy.random.uniform(0.4,0.7)
            speed = speed_value*(self.A[g]-position)/numpy.linalg.norm((self.A[g]-position))
            self.obstacle1_msg.twist.linear.x = speed[0]
            self.obstacle1_msg.twist.linear.y = speed[1]
        #self.obstacle2_msg.pose.orientation.x = 1
        rospy.wait_for_service('/gazebo/set_model_state')
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        set_state(self.obstacle1_msg)

    def goal_space(self):

        '''self.xy = numpy.array([[0.6,-1.3],[-0.2,4.7],[5.2,3.3],[8.7,5.1],[8.4,-1.4],[7.8,-5.2],[2.6,-5],[2.8,4.0],[6.6,-5.2],[-0.2,-4.8]])'''

        #self.xy = numpy.array([[8.1,-1],[8.2,-5],[9,4.5],[2,1],[0.5,-1],[6.5,5],[8.2,-5],[0,1],[7.3,-2.5],[0.5,-1.5]])
        X = []
        Y = []

        X,Y = numpy.mgrid[8.1:8.9:0.1, -5.8:-0.4:0.1]                 # mesh for bottom left room
        self.xy = numpy.vstack((X.flatten(), Y.flatten())).T
        X = []
        Y = []

        X,Y = numpy.mgrid[4.6:7.3:0.1, -5.7:-4.8:0.1]                 # mesh for bottom left room
        self.xy = numpy.append(self.xy,numpy.vstack((X.flatten(), Y.flatten())).T, axis=0)
        X = []
        Y = []

        X,Y = numpy.mgrid[4.6:7.1:0.1, -2.3:0.2:0.1]                 # mesh for bottom left room
        self.xy = numpy.append(self.xy,numpy.vstack((X.flatten(), Y.flatten())).T, axis=0)
        X = []
        Y = []

        X,Y = numpy.mgrid[4.3:5.6:0.1, -4.2:-2.9:0.1]                 # mesh for bottom left room
        self.xy = numpy.append(self.xy,numpy.vstack((X.flatten(), Y.flatten())).T, axis=0)
        X = []
        Y = []

        X,Y = numpy.mgrid[3.8:6.1:0.1, 1.0:6.1:0.1]                 # mesh for middle room
        self.xy = numpy.append(self.xy,numpy.vstack((X.flatten(), Y.flatten())).T, axis=0)
        X = []
        Y = []

        X,Y = numpy.mgrid[8.4:8.7:0.1, 2:5.8:0.1]                 # mesh for bottom right room
        self.xy = numpy.append(self.xy,numpy.vstack((X.flatten(), Y.flatten())).T, axis=0)
        X = []
        Y = []

        X,Y = numpy.mgrid[-0.8:2.5:0.1, 4.3:6.1:0.1]                 # mesh for top right room
        self.xy = numpy.append(self.xy,numpy.vstack((X.flatten(), Y.flatten())).T, axis=0)
        X = []
        Y = []

        X,Y = numpy.mgrid[-0.7:2.5:0.1, -2.7:-0.2:0.1]                 # mesh for top left room
        self.xy = numpy.append(self.xy,numpy.vstack((X.flatten(), Y.flatten())).T, axis=0)
        X = []
        Y = []

        X,Y = numpy.mgrid[-0.1:1.4:0.1, 0.6:2.4:0.1]                 # mesh for top left room
        self.xy = numpy.append(self.xy,numpy.vstack((X.flatten(), Y.flatten())).T, axis=0)
        X = []
        Y = []

        X,Y = numpy.mgrid[0:2.6:0.1, -5.7:-4.7:0.1]                 # mesh for top left room
        self.xy = numpy.append(self.xy,numpy.vstack((X.flatten(), Y.flatten())).T, axis=0)
