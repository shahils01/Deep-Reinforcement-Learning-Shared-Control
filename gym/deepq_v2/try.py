import os
import time
import numpy as np

folder="eplinearspeed1"+str(3)
pointer = 0
while pointer<5e5:
    os.system('python3 DQL_shared_load.py')
    pointer = int(np.load('parameters1.npy')[0])
    time.sleep(10)
os.system('cp -r reward1.npy parameters1.npy crash.npy eval_reward.npy memory1.npy plot.py final_net backup_net '+folder+'/')
for i in range (4,10):
    folder="eplinearspeed1"+str(i)
    os.system("mkdir "+folder)
    pointer = 0
    os.system('python3 DQL_shared.py')
    time.sleep(10)
    while pointer<5e5:
        os.system('python3 DQL_shared_load.py')
        pointer = int(np.load('parameters1.npy')[0])
        time.sleep(10)
    os.system('cp -r reward1.npy parameters1.npy crash.npy eval_reward.npy memory1.npy plot.py final_net backup_net '+folder+'/')
