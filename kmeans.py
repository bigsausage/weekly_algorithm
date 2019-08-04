# 自己实现一下Kmeans 算法:
# samples n*m n个样本 每个样本m个特征
# 算法步骤 1. 随机生成k个质心 2.将样本分到k个质心，若k个质心的样本无变化，退出。（只要遍历样本观察即可）
#         3.若有变化，再根据各个类别生成k个质心，重复2，直到收敛或者到max_iter

import numpy as np

# 样本 n * m
# 随机生成k个质心 返回长度为 k*m 的 np.ndarray 把
def randCenter(data , k):
    if not isinstance(data, np.ndarray):
        try:
            data = np.asarray(data)
        except:
            raise TypeError("numpy.ndarray required for data")

    n, m = np.shape(data)[0] ,np.shape(data)[1]
    centers = np.mat(np.zeros((k , m)))
    # 根据每一列（每个特征）随机生成质心
    for j in range(m):
        minValue = min(data[:,j])
        maxValue = max(data[:,j])
        centers[:,j] = np.random.uniform(minValue,maxValue,(k,1))
    return centers

# 根据data(n*m)和category n
# 分类生成新的 k * m :
def averCenter(data, category, k ):
    # 根据每个样本的category 分成k 类别, 计算新的质心
    # 弄个字典 data[k] = [] 计算每个样本
    dataDict = dict()
    for kind in range(k):
        dataDict[kind] = []
    for i in range(len(category)):
        # data[i] 第i个样本, category[i] 第i个样本所属的类别
        dataDict[category[i]].append(data[i])

    newCenters = np.mat(np.zeros((k,np.shape(data)[1])))
    for key,values in dataDict.items():
        if len(values) > 0:
            averValues = np.average(values,axis=0)
            newCenters[key,:] = averValues

    return newCenters


def calDist(vecA,vecB):
    return np.sqrt(np.sum(np.power(vecA-vecB,2)))


# 根据长度为K的质心，对样本进行分类，返回长度为n的List类别，
# 将 n * m 的data 返回 长度为n 的 list
def judgeCategory(data , k_centers):
    category = []
    n ,m = np.shape(data)[0], np.shape(data)[1]
    k = np.shape(k_centers)[0]
    # 对于每个样本data[i]
    for i in range(n):
        # 计算 data[i] 和 k_centers[j] 哪个近
        minDist = np.inf
        minK = 0
        for j in range(k):
            distance = calDist(data[i],k_centers[j])
            if distance < minDist:
                minDist = distance
                minK = j
        category.append(minK)

    return category

# 返回List 即各个样本的类别
def Kmeans(data, k, max_iter = 500):
    # 原来所属类别
    pre_category = [-1] * np.shape(data)[0]
    # 生成center点 (k * m)
    k_centers = randCenter(data , k)

    # 根据k个质心重新判断类别
    after_category = judgeCategory(data, k_centers)

    # 迭代
    while pre_category != after_category and max_iter > 0:
        max_iter -= 1
        pre_category = after_category
        k_centers = averCenter(data, pre_category, k)
        after_category = judgeCategory(data, k_centers)

        # print(after_category,k_centers)

    return after_category, k_centers


data =  np.asarray([
    [1, 2],
    [2, 1],
    [3, 1],
    [5, 4],
    [5, 5],
    [6, 5],
    [10, 8],
    [7, 9],
    [11, 5],
    [14, 9],
    [14, 14],
    ])

finalCat , k_centers = Kmeans(data, 3)
print(finalCat, k_centers)
