import numpy as np
def calculate(x):
	return((2**-1)*np.log10(x))
x=np.random.randint(1, 1000, size=(1, 500))
y= calculate(x)
a=np.row_stack((x, y))
import matplotlib.pyplot as plt
plt.scatter(x,y)