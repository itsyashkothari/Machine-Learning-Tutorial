import numpy as np
import matplotlib.pyplot as plt
import time

start2_time = time.time()

x = np.random.rand(10)
y = np.random.rand(10)

sum = 0
i=0
for a in x:
	sum = sum + (x[i]*y[i])
	i=i+1
print(sum)

print("--- %s seconds ---" % (time.time() - start2_time))
