import numpy as np

array1 = np.array([1,2,3])
array2 = np.array([55,62,73])
array3 = np.array([98,82,23])

tuple1 = (array1, array2, array3)

if isinstance(tuple1, tuple):
    args = tuple((tuple1[n] for n in range(len(tuple1))))

print(type(tuple1),type(args),args)