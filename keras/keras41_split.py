import numpy as np

a = range(1,10)
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
# a = scaler.fit_transform(a)
# print(a)
size = 7

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1 ):
        subset = dataset[i:(i+size)]
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(a,size)
x = dataset[:,:-1]
y = dataset[:,-1]
x = scaler.fit_transform(x)
print(x)

        
        