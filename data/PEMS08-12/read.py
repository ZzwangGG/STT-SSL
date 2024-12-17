import numpy as np
data = np.load('train.npz')
print(data.files)
print(data['x'].shape)
print(data['y'].shape)

data1 = np.load('val.npz')
print(data1['x'].shape)
print(data1['y'].shape)

data2 = np.load('test.npz')
print(data2['x'].shape)
print(data2['y'].shape)

data3 = np.load('adj_mx.npz')
print(data3.files)
print(data3['adj_mx'].shape)



