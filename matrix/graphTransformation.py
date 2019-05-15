#引入一个科学图形库matplotlib
import matplotlib.pyplot as plt
import numpy as np
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
#图像平移
matrix = np.array([-5, 0])
newPoints = points + matrix
plt.plot(newPoints[:, 0], newPoints[:, 1])

plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.show()

