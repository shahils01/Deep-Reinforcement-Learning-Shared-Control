import numpy as np
import matplotlib.pyplot as plt
collision1 = []
collision2 = []
SA1 = []
SA2 = []
Workload1 = []
Workload2 = []

for i in range(20):
	file = "/home/i2rlab/rl_gym/example/TD3/shared_v2/human_exp/Human_exp_"+str(i)+"/data.dat"
	data = np.loadtxt(file)
	collision1=np.hstack((collision1,data[0,2]))
	SA1=np.hstack((SA1,data[0,-2]))
	Workload1=np.hstack((Workload1,data[0,-1]))
	collision2=np.hstack((collision2,data[1,2]))
	SA2=np.hstack((SA2,data[1,-2]))
	Workload2=np.hstack((Workload2,data[1,-1]))

print("collision1:",collision1)
print("collision2:",collision2)
print("SA1:",SA1)
print("SA2:",SA2)
print("Workload1:",Workload1)
print("Workload2:",Workload2)

print("avg_collision1:",np.mean(collision1))
print("std_collision1:",np.std(collision1))
print("avg_collision2:",np.mean(collision2))
print("std_collision2:",np.std(collision2))
print("avg_SA1:",np.mean(SA1))
print("std_SA1:",np.std(SA1))
print("avg_SA2:",np.mean(SA2))
print("std_SA2:",np.std(SA2))
print("avg_Workload1:",np.mean(Workload1))
print("std_Workload1:",np.std(Workload1))
print("avg_Workload2:",np.mean(Workload2))
print("std_Workload2:",np.std(Workload2))