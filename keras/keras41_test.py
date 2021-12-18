import keras41_split as split
import numpy as np

a = np.array(range(1,201))
size = 7
print(split.split_x(a,size))