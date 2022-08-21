import numpy as np
max = []
last_t = []
last_s=[]
for i in range(0, 32):
    dir = '/home/wyd/informer_jsd_deap/checkpoint/result3/'+str(i)+'/result.npz'
    A = np.load(dir)
    max.append(A['best_accu_t'])
    last_t.append(A['accu_t'])
    last_s.append(A['accu_s'])
avg = np.average(max)
avg_std = np.std(max)
avg_t = np.average(last_t)
std_t = np.std(last_t)
avg_s = np.average(last_s)
std_s = np.std(last_s)
print("max", np.max(max))
print("min", np.min(max))
print("avg", avg)
print("avg_t", avg_t)
print("avg_s", avg_s)
print("std_s", std_s)
print("std_t", std_t)
print("avg_std", avg_std)
