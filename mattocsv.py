import scipy.io as scio
import numpy as np
import evaluate as ev
import pandas as pd
import effect_fun
def read_from_file(name='data_source/data1.mat'):
    dataset_dir='/share/code/dataset/ship_design/'
    path = dataset_dir+name+'.mat'
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
def data_solve(datas):
    datas_len = len(datas)  # 数组长度
    R_tatals = []  # 总阻力
    Bales_levels = []  # 耐波贝尔斯品级
    Diameters = []  # 回转直径
    Displacements = []  # 排水量
    GMs = []  # 初稳性高
    for data in datas:  # 循环数组中的数据
        data_len = len(data)  # 每个数据的维度
        # [船长 船宽 吃水 进流段长度 平行中体长度 去流段长度 浮心纵向位置 棱形系数
        # 半进流角 水线面系数 艉部纵向斜度 艏部倾斜角 球艏长度 球艏宽度 球艏基线以上高度
        # 球艏基线以下高度 艉部倾斜角 艉切点位置 #1舷侧倾斜角 #4舷侧倾斜角 #7舷侧倾斜角
        # #10舷侧倾斜角 #13舷侧倾斜角 #16舷侧倾斜角 #19舷侧倾斜角 #22舷侧倾斜角]
        temp = [data[0], data[1], data[3]] + list(data[35:40]) + [data[42], data[40], data[41]] + list(
            data[43:len(data)])
        temp1 = []
        for i in temp:
            temp1.append(round(i, 2))
        R_total, Bales_level, Diameter, Displacement, GM = ev.evaluate(temp1)
        R_tatals.append(R_total)
        Bales_levels.append(Bales_level)
        Diameters.append(Diameter)
        Displacements.append(Displacement)
        GMs.append(GM)
    # 归一化
    # print(R_tatals)
    result = {}
    # R_tatals_norm, R_tatals_max, R_tatals_min = normalize(R_tatals)
    # Bales_levels_norm, Bales_levels_max, Bales_levels_min = normalize(Bales_levels)
    # Diameters_norm, Diameters_max, Diameters_min = normalize(Diameters)
    # Displacements_norm, Displacements_max, Displacements_min = normalize(Displacements)
    # GMs_norm, GMs_max, GMs_min = normalize(GMs)
    result['R_tatals'] = R_tatals
    result['Bales_levels'] = Bales_levels
    result['Diameters'] = Diameters
    result['Displacements'] = Displacements
    result['GMs'] = GMs
    samples=[]
    for i in range(len(result['R_tatals'])):
        sample=[result['R_tatals'][i],result['Bales_levels'][i],result['Diameters'][i],result['Displacements'][i],result['GMs'][i]]
        samples.append(sample)
    # pd.DataFrame(np.array(samples)).to_csv('samples.csv',index=False, header=0)
    return result
def add_effun():
    f=effect_fun.EffectFun()
    main_fives = np.loadtxt("../dataset/ship_design/main_five_index.csv", delimiter=",")
    new_main_fives=[]
    for k in main_fives:
        k[1]=f.r_tatals_effectFun(k[1])
        k[2]=f.bales_levels_effectFun(k[2])
        k[3]=f.diameters_effectFun(k[3])
        k[4]=f.displacements_effectFun(k[4])
        k[5]=f.gms_effectFun(k[5])
        new_main_fives.append(k)
    pd.DataFrame(np.array(new_main_fives)).to_csv('main_five_fun.csv', index=False,header=0)
def main():
    datas=read_from_file('Initial_Solution')
    data_solve(datas)
if __name__=='__main__':
    main()
# data=read_from_file('Initial_Solution')
# result=data_solve(data)
# python pareto.py ../dataset/ship_design/main_five_fun_index.csv --output output.csv --maximize-all --delimiter=',' --header=1