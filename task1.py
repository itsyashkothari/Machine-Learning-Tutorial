import numpy as np
import matplotlib.pyplot as plt

a = np.random.random((1,500))

x = np.arange(0.1,100,.1)
y = (1/0.001)* np.log(x)

#print(y)
#print(a)
plt.plot(x,y)
plt.show()
