#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import math

def main():
    data = np.loadtxt('scaled_house_data_new_range.dat')
    for i in range (data.shape[1]-1):
        r_main = data[i,5:-3]
        #print('r main',r_main)
        plt.axes(projection = 'polar')
        rads = np.arange(0, (2 * np.pi), np.pi/90)

        plt.polar(rads, r_main, 'g.')

        # display the polar plot
        #plt.show()
        plt.pause(0.05)
        plt.clf()

if __name__ == '__main__':
    main()
