#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import time
from std_msgs.msg import Int64
import rospy

class signal(object):
    def __init__(self):
        rospy.init_node('sign', anonymous=True, log_level=rospy.WARN)
        rospy.Subscriber("/sign", Int64, self._sign_callback)
        rospy.Subscriber("/share", Int64, self._share_callback)
        style.use('ggplot')
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111,projection = 'polar')


    def _sign_callback(self, data):
        self.sign = data

    def _share_callback(self, data):
        self.share = data

    def animate(self,i):
        self.ax.clear()
        try:
            if self.sign.data < -200:
                self.ax.scatter(0, 0, marker='s', c='red', s=40000, cmap='hsv', alpha =0.75)
            elif self.sign.data > 200:
                self.ax.scatter(0, 0, marker='s', c='green', s=40000, cmap='hsv', alpha =0.75)
            elif self.share.data ==1:
                self.ax.scatter(0, 0, marker='s', c='yellow', s=40000, cmap='hsv', alpha =0.75)
            else:
                self.ax.scatter(0, 0, marker='s', c='blue', s=40000, cmap='hsv', alpha =0.75)
            self.ax.axis('off')
            self.ax.grid(False)
        except:
            pass


Get = signal()
ani = animation.FuncAnimation(Get.fig,Get.animate,interval=200)
plt.show()
