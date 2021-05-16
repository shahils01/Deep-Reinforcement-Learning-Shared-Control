import os
import time
import numpy as np

for i in range(0,1):
    folder="ep4"+str(i)
    os.system("mkdir "+folder)
    pointer = 0
    os.system('python3 shared_TD3_safe_cnn.py')
    time.sleep(10)
    while pointer<5e5:
        os.system('python3 shared_TD3_safe_cnn_load.py')
        pointer = int(np.load('parameters1.npy')[1])
        time.sleep(10)
    os.system('cp -r reward1.npy parameters1.npy crash.npy eval_reward.npy memory1.npy plot.py final_net backup_net '+folder+'/')
