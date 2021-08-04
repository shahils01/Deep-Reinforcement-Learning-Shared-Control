import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import time

style.use('ggplot')
fig = plt.figure()
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,2,3)
ax3 = fig.add_subplot(3,2,4)
ax4 = fig.add_subplot(3,1,3)


def remove_zero_rows(X):
    # X is a scipy sparse matrix. We want to remove all zero rows from it
    nonzero_row_indice, _ = X.nonzero()
    unique_nonzero_indice = np.unique(nonzero_row_indice)
    return X[unique_nonzero_indice]



def animate(i):
    reward=np.load('reward1.npy')
    ax1.clear()
    #ax1.set_ylim(-5000,1500)
    ax1.plot(range(len(reward)),reward)
    ax1.set_ylabel('reward')
    ax2.clear()
    try:
        ax2.plot(range(100),reward[-100:])
    except:
        pass
    ax2.set_ylabel('reward')

    width = 0.35  # the width of the bars
    ax3.clear()
    ax3.set_xlim(0,2)
    ax3.set_ylim(0,1500)
    try:
        rects1 = ax3.bar(1, np.mean(reward[-100:]), width, label='average')
    except:
        pass
    ax3.set_ylabel('reward')

    try:
        eval_reward = np.load('eval_reward.npy')
        ax4.clear()
        ax4.plot(eval_reward[:,0],eval_reward[:,1])
        ax4.set_ylabel('reward')
        # ax4.set_ylim(-2000,1500)
    except:
        pass

'''
    pointer = [int(np.loadtxt('parameters1.dat')[1])/50000*100]
    width = 0.35  # the width of the bars
    ax2.clear()
    ax2.set_xlim(0,2)
    ax2.set_ylim(0,100)
    rects1 = ax2.bar(1, pointer, width, label='ddpg.pointer')
    ax2.set_ylabel('pointer')

    var = [float(np.loadtxt('parameters1.dat')[0])]
    width = 0.35  # the width of the bars
    ax3.clear()
    ax3.set_xlim(0,2)
    ax3.set_ylim(0,2)
    rects2 = ax3.bar(1, var, width, label='var')
    ax3.set_ylabel('var')
'''

ani = animation.FuncAnimation(fig,animate,interval=3000)
'''
reward=np.loadtxt('reward1.dat')
ax1.clear()
ax1.plot(range(len(reward)),reward)
ax1.set_ylabel('reward')
ax2.clear()
ax2.plot(range(200),reward[-200:])
ax2.set_ylabel('reward')

width = 0.35  # the width of the bars
ax3.clear()
ax3.set_xlim(0,2)
ax3.set_ylim(0,2000)
rects1 = ax3.bar(1, np.mean(reward[-200:]), width, label='average')
ax3.set_ylabel('reward')
'''
plt.show()
