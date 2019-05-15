import numpy as np
import collections as c

# data = np.array([
#         [154,1],
#         [126,2],
#         [70,2],
#         [196,2],
#         [161,2],
#         [371,4]
# ])
# feature = (data[:,0])
# print(feature)
# print(list(map(lambda x:abs(200-x),feature)))


# sortedlabel = np.array([2,2,2,2,3,3,3,3,3,3,2,1])
# print(c.Counter(sortedlabel).most_common(1)[0][0])

data = np.loadtxt("data0.csv",delimiter=",")
print(data)
