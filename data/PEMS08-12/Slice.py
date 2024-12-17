# # data = np.load('PEMS08-12.npz')
# # print(data["data"].shape)
import csv

import numpy
import numpy as np

# Load traffic data from file
traffic_data = np.load("../PEMS08.npz")

# Determine indices to split the data
n_train = int(len(traffic_data['data']) * 0.7)
n_val = int(len(traffic_data['data']) * 0.1)
n_test = len(traffic_data['data']) - n_train - n_val

# Split the data into train, validation, and test sets
train_data = traffic_data['data'][:n_train][:, :, 2][:, :, np.newaxis]
val_data = traffic_data['data'][n_train:n_train+n_val][:, :, 2][:, :, np.newaxis]
test_data = traffic_data['data'][n_train+n_val:][:, :, 2][:, :, np.newaxis]

# Convert the data into input/output pairs
def generate_examples(data):
    X, Y = [], []
    for i in range(12, len(data)):
        if i + 11 < len(data):
            X.append(data[i-12:i])
            Y.append(data[i+11])
    X = np.array(X)
    Y = np.array(Y).reshape(-1, 1, 170, 1)
    return X, Y
#
#
train_x, train_y = generate_examples(train_data)
# print(train_x.shape)
# print(train_y.shape)
np.savez('train.npz', x=train_x, y=train_y)
val_x, val_y = generate_examples(val_data)
# print(val_x.shape)
# print(val_y.shape)
np.savez('val.npz', x=val_x, y=val_y)
test_x, test_y = generate_examples(test_data)
np.savez('test.npz', x=test_x, y=test_y)
# print(test_x.shape)
# print(test_y.shape)
def get_adj_matrix(file, nodenumber, graph_type='distance'):
    """

    :param file: 文件地址 如 'PEMS04/PEMS04.csv'
    :param nodenumber: 节点数目 如 307
    :param graph_type: 邻接矩阵类型 如 是否考虑距离
    :return: 如 307*307 邻接矩阵
    """
    adj = np.zeros([int(nodenumber), int(nodenumber)])
    with open(file, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for line in reader:
            if len(line) != 3:
                continue
            i, j, distance = int(line[0]), int(line[1]), float(line[2])

            if graph_type == 'connect':
                adj[i, j], adj[j, i] = 1., 1.
            elif graph_type == 'distance':
                adj[i, j] = 1. / distance
                adj[j, i] = 1. / distance
            else:
                raise ValueError('graph_type must be (connect or distance)')
    return adj  # (170, 170)
adj_matrix = get_adj_matrix('../PEMS08.csv', nodenumber=170)
np.savez('adj_mx', adj_mx=adj_matrix)
