import rospy
import numpy
import random
from gym import spaces
from openai_ros.robot_envs import real_turtlebot
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
'''
with Manually preprocessed U and lidar readings
'''
class RealTurtleBot2Env(real_turtlebot.TurtleBot2Env):
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
        super(RealTurtleBot2Env, self).__init__(ros_ws_abspath)

        # Only variable needed to be set here
        high = numpy.array([1,1,1,1])
        low = numpy.array([-1,-1,-1,-1])
        self.action_space = spaces.Box(low, high)

        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-numpy.inf, numpy.inf)

        self.success = True
        #number_observations = rospy.get_param('/turtlebot2/n_observations')

        # Actions and Observations

        self.init_linear_forward_speed = rospy.get_param('/turtlebot2/init_linear_forward_speed')
        self.init_linear_turn_speed = rospy.get_param('/turtlebot2/init_linear_turn_speed')

        self.new_ranges = 180
        self.min_range = rospy.get_param('/turtlebot2/min_range')
        self.max_laser_value = rospy.get_param('/turtlebot2/max_laser_value')
        self.min_laser_value = rospy.get_param('/turtlebot2/min_laser_value')


        # We create two arrays based on the binary values that will be assigned
        # In the discretization method.
        laser_scan = self.get_laser_scan()
        rospy.logdebug("laser_scan len===>" + str(len(laser_scan.ranges)))


        high = numpy.hstack((numpy.array([0.5,1,0.5,1]),6*numpy.ones([self.new_ranges])))
        low = numpy.hstack((numpy.array([-0.5,-1,-0.5,-1]),numpy.zeros([self.new_ranges])))
        self.observation_space = spaces.Box(low, high)

        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))

        self.cumulated_steps = 0.0


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
        self._episode_done = False
        self._outofrange = False
        odometry = self.get_odom()

    def _set_action(self, ratio):
        """
        """
        ratio=list(ratio)
        rospy.logdebug("Start Set Action ==>"+str(ratio))
        print("ratio:",ratio)
        if self.joy_linear==0 and self.joy_angular==0:
            linear_speed=0
            angular_speed=0
        elif abs(self.theta_bt_uh_ob)> 1.3 or self.e_norm>1.5:
            linear_speed=self.joy_linear
            angular_speed=self.joy_angular
        else:
            linear_speed = ratio[0]*self.joy_linear+ratio[1]*self.a_ao[0,0]+ratio[2]*self.a_c[0,0]+ratio[3]*self.a_cc[0,0]
            angular_speed = ratio[0]*self.joy_angular+ratio[1]*self.a_ao[1,0]+ratio[2]*self.a_c[1,0]+ratio[3]*self.a_cc[1,0]
        #
        # linear_speed = ratio[0]*self.joy_linear+ratio[1]*self.a_ao[0,0]+ratio[2]*self.a_c[0,0]+ratio[3]*self.a_cc[0,0]
        # angular_speed = ratio[0]*self.joy_angular+ratio[1]*self.a_ao[1,0]+ratio[2]*self.a_c[1,0]+ratio[3]*self.a_cc[1,0]
        self.move_base(linear_speed, angular_speed, epsilon=0.05, update_rate=10)
        rospy.logdebug("END Set Action ==>"+str(ratio))

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        TurtleBot2Env API DOCS
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        laser_scan = self.get_laser_scan()

        self.discretized_laser_scan = self.discretize_observation( laser_scan,
                                                                self.new_ranges
                                                                )
        # We get the odometry so that SumitXL knows where it is.
        odometry = self.get_odom()
        x_position = odometry.pose.pose.position.x
        y_position = odometry.pose.pose.position.y
        base_orientation_quat = odometry.pose.pose.orientation
        base_roll, base_pitch, base_yaw = self.get_orientation_euler(base_orientation_quat)
        v = odometry.twist.twist.linear.x
        self.theta_dot = odometry.twist.twist.angular.z

        ###################human input ##############################
        joy = self.get_joy()
        self.joy_linear = joy.linear.x
        self.joy_angular = joy.angular.z

        tran = numpy.array([[numpy.cos(0),-numpy.sin(0)],[numpy.sin(0),numpy.cos(0)]]).dot(numpy.array([[1,0],[0,0.115]]))
        self.u_gtg = tran.dot(numpy.array([[self.joy_linear],[self.joy_angular]]))
        self.obstacle_avoidance()
        # We round to only two decimals to avoid very big Observation space
        ################################################  Change Observations  ########################################################################################
        observations = [round(x_position, 2),round(y_position, 2),round(base_yaw, 2),round(v, 2),round(self.theta_dot, 2),round(self.joy_linear, 2),round(self.joy_angular, 2)]+self.discretized_laser_scan+[self.e_norm, round(self.theta_bt_uh_ob,2)]
        #print("Observations==>"+observations)
        rospy.logdebug("Observations==>"+str(observations))
        rospy.logdebug("END Get Observation ==>")
        return observations

    def obstacle_avoidance(self):
        lidar = self.discretized_laser_scan
        print(lidar[45])
        yaw = 0
        ################################## nearest ######################################################
        n = numpy.argmin(lidar)
        if n<=89:
            delta_angle = ((n-89)*2-1)/180*numpy.pi
        else:
            delta_angle = ((n-90)*2+1)/180*numpy.pi
        orientation = yaw + delta_angle
        self.e_norm = lidar[n]
        if self.e_norm<1.5:
            share = 1
        else:
            share = 0
        self.share_talker.publish(share)
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
        ###################################################################################################
        u_ao = 0.5/self.e_norm*(1/(self.e_norm**2+0.1))*e
        u_c = 0.5*numpy.array([[0,1],[-1,0]]).dot(u_ao)
        u_cc =0.5*numpy.array([[0,-1],[1,0]]).dot(u_ao)
        self.a_ao = numpy.array([[1,0],[0,1/0.115]]).dot(numpy.array([[numpy.cos(-yaw),-numpy.sin(-yaw)],[numpy.sin(-yaw),numpy.cos(-yaw)]])).dot(u_ao)
        self.a_c = numpy.array([[1,0],[0,1/0.115]]).dot(numpy.array([[numpy.cos(-yaw),-numpy.sin(-yaw)],[numpy.sin(-yaw),numpy.cos(-yaw)]])).dot(u_c)
        self.a_cc = numpy.array([[1,0],[0,1/0.115]]).dot(numpy.array([[numpy.cos(-yaw),-numpy.sin(-yaw)],[numpy.sin(-yaw),numpy.cos(-yaw)]])).dot(u_cc)
        print("u_ao:",u_ao)
        self.limitu()
        self.theta_bt_uh_ob = abs(numpy.arctan2(self.u_gtg[1,0],self.u_gtg[0,0])-orientation)
        if self.theta_bt_uh_ob > numpy.pi:
            self.theta_bt_uh_ob = 2 * numpy.pi - self.theta_bt_uh_ob

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
        if abs(self.a_ao[0,0])>0.22 or abs(self.a_ao[1,0])>0.3:
            self.a_ao =self.a_ao/max(abs(self.a_ao/[[0.22],[0.3]]))
        if abs(self.a_c[0,0]>0.22) or abs(self.a_c[1,0])>0.4:
            self.a_c =self.a_c/max(abs(self.a_c/[[0.22],[0.4]]))
        if abs(self.a_cc[0,0])>0.22 or abs(self.a_cc[1,0])>0.4:
            self.a_cc =self.a_cc/max(abs(self.a_cc/[[0.22],[0.4]]))
        #print(self.a_ao,self.a_c,self.a_cc)


    def _is_done(self, observations):
        return self._episode_done

    def _compute_reward(self, observations, done):
        return 0


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
                if item == float ('Inf') or numpy.isinf(item) or item==0:
                    discretized_ranges.append(self.max_laser_value)
                elif numpy.isnan(item):
                    discretized_ranges.append(0.4)
                else:
                    discretized_ranges.append(round(item,3)*3)

                if (self.min_range > item*3 > 0):
                    #rospy.logerr("done Validation >>> item=" + str(item)+"< "+str(self.min_range))
                    self._episode_done = True
                #else:
                    #rospy.logwarn("NOT done Validation >>> item=" + str(item)+"> "+str(self.min_range))
        discretized_ranges = discretized_ranges[int(len(discretized_ranges)/2):]+discretized_ranges[:int(len(discretized_ranges)/2)]
        return discretized_ranges


    def is_in_desired_position(self,current_position, epsilon=0.5):
        is_in_desired_pos = False
        return is_in_desired_pos


    def get_orientation_euler(self, quaternion_vector):
        # We convert from quaternions to euler
        orientation_list = [quaternion_vector.x,
                            quaternion_vector.y,
                            quaternion_vector.z,
                            quaternion_vector.w]

        r = R.from_quat(orientation_list)
        roll, pitch, yaw = r.as_rotvec()
        return roll, pitch, yaw
