#引入一个科学图形库matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
points = np.array([
    [0,0],
    [0,5],
    [3,5],
    [3,4],
    [1,4],
    [1,3],
    [2,3],
    [2,2],
    [1,2],
    [1,0],
    [0,0],
])

plt.plot(points[:,0],points[:,1])
#图像旋转
matrix = np.array([
    [math.cos(math.pi/2), -math.sin(math.pi/2)],
    [math.sin(math.pi/2), math.cos(math.pi/2)]
])
newPoints = np.dot(points,matrix.T)
plt.plot(newPoints[:, 0], newPoints[:, 1])

plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.show()
