import numpy as np
import time
start1 = time.clock()
x=np.random.randint(1, 100, size=(10, 1))
y=np.random.randint(1, 100, size=(10, 1))
a=np.dot (np.transpose(x),y)
end1=time.clock()
print(end1-start1)


start2 = time.clock()
sum=0
for i in range(len(x)):
	sum+=x[i]*y[i]
end2=time.clock()
print(end2-start2)

if (end1-start1)<(end2-start2):
	print("First method	 is faster")
elif (end1-start1)>(end2-start2):
	print("Second method is faster")
else:
	print("Clock isn't precise")