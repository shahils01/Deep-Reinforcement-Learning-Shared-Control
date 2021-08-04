#!/usr/bin/env python3
import os
import time
import numpy as np

#class data_correction(object):

def main():
    data = np.loadtxt('scaled_house_data_1.dat')

    for i in range (data.shape[0]):
        data_new = []
        data_new = np.zeros(15)
        data_new[:5] = data[i,:5]
        data_new[-2:] = data[i,-2:]

        for j in range (8):
            if min(data[i,5:28]) <= 3:
                data_new[5] = min(data[i,5:28])
            else:
                data_new[5] = 10

            if min(data[i,28:50]) <= 3:
                data_new[6] = min(data[i,28:50])
            else:
                data_new[6] = 10

            if min(data[i,50:73]) <= 3:
                data_new[7] = min(data[i,50:73])
            else:
                data_new[7] = 10

            if min(data[i,73:95]) <= 3:
                data_new[8] = min(data[i,73:95])
            else:
                data_new[8] = 10

            if min(data[i,95:118]) <= 3:
                data_new[9] = min(data[i,95:118])
            else:
                data_new[9] = 10

            if min(data[i,118:140]) <= 3:
                data_new[10] = min(data[i,118:140])
            else:
                data_new[10] = 10

            if min(data[i,140:163]) <= 3:
                data_new[11] = min(data[i,140:163])
            else:
                data_new[11] = 10

            if min(data[i,163:185]) <= 3:
                data_new[12] = min(data[i,163:185])
            else:
                data_new[12] = 10

        with open("scaled_house_data_annotated_1.dat", "a", newline='') as f:
            #f.write(data_needed+b"\n")           # write the data to the file
            f.write(str(data_new).replace('\n','').replace('[','').replace(']','')+'\n')

    #data_new = data_new[1:,:]
    #print('data shape: ',data.shape)
    #print('corrected data shape: ',data_new.shape)

if __name__ == '__main__':
    main()
