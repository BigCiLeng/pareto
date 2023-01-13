import scipy.io as scio
import numpy as np
import evaluate as ev
import pandas as pd


def read_from_file(name='Initial_Solution'):
    path = '/share/code/dataset/ship_design/' + name + '.mat'
    data = scio.loadmat(path)
    pri_dim = data[name][0][0]  # 主尺度参数：船长，船宽，型深，吃水，方形系数 5
    # print(len(pri_dim))
    func_pla = data[name][0][1]  # 功能布置参数 30
    # print(len(func_pla))
    contour = data[name][0][2]  # 型线
    temp = np.append(pri_dim, func_pla, axis=0)
    data = np.append(temp, contour, axis=0).T
    # print(data.shape)
    return data


# data = read_from_file('Initial_Solution')
# main_value = np.append(data[:, 0:5], data[:, 35:data.shape[1]], axis=1)
# pd.DataFrame(np.array(main_value)).to_csv('main_value.csv',
#                                           index=False,
#                                           header=0)
samples = np.loadtxt("samples.csv", delimiter=",")
pd.DataFrame(np.array(samples)).to_csv('samples1.csv', header=0)