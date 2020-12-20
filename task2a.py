import numpy as np
import matplotlib.pyplot as plt
import time

start1_time = time.time()

x = np.random.rand(10)
y = np.random.rand(10)

#print(x)
#print(y)

print(x.dot(y))
print("--- %s seconds ---" % (time.time() - start1_time))

