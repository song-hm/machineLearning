import numpy as np
A = np.array([
    [3,1,2],
    [-5,4,1],
    [0,3,-8],
    [3,3,2],
]) # 4,3
B = np.array([
    [0,5,-1],
    [3,2,-1],
    [10,0.5,4],
]) # 3,3

print(np.dot(A, B))