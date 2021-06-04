import pandas as pd
import numpy as np
from fancyimpute import KNN, SoftImpute, IterativeImputer, BiScaler
from collections import Counter
import matplotlib.pyplot as plt


# 用最高频率值来填补缺失值
def na_max(path):
    wine_data = pd.read_csv(path, header=0, index_col=0, engine='python', encoding='utf-8')
    wine_data = wine_data.values
    max_time = []
    for cl in range(wine_data.shape[1]):
        counter = Counter(wine_data[:, cl])
        counter = counter.most_common()
        if counter[0][0] == counter[0][0]:
            max_time.append(counter[0][0])

        else:  # 如果最大频数为空值
            max_time.append(counter[1][0])
    # 对每个属性的空值进行替换
    wine_max = pd.DataFrame(wine_data)
    for cl in range(wine_data.shape[1]):
        wine_max[cl] = wine_max[cl].fillna(max_time[cl])
        # print(max_time[cl])
    wine_max.to_csv('C:/学习/办公/研究生课程/研一第二学期课程/数据挖掘/archive/winemag-data-130k-v2.csv')


    return wine_max


path = 'C:/学习/办公/研究生课程/研一第二学期课程/数据挖掘/archive/winemag-data-130k-v2.csv'
wine_max = na_max(path)  # 用最高频率值来填补缺失值
