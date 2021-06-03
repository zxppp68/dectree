from sklearn import datasets
import math
import numpy as np
import sys


#获得属性的每个值的熵
def getEntropy(counter):
    res = 0
    denominator = np.sum(counter)
    if denominator == 0:
        return 0
    for value in counter:
        if value == 0:
            continue
        res += value / denominator * math.log2(value / denominator if value > 0 and 
                                               denominator > 0 else 1)
    return -res

#随机抽取60%的训练集和40%的测试集
def divideData():
    completeData = np.c_[iris.data, iris.target.T]
    np.random.shuffle(completeData)
    trainData = completeData[range(int(length * 0.6)), :]
    testData = completeData[range(int(length * 0.6), length), :]
    return [trainData, testData]


#采用二分法对连续属性离散化
#取连续集每两个数的中位点作为候选划分点,计算每个候选划分点的信息增益(这里只计算熵),获得最佳的划分点
def discretization(index):
    temp = np.array(iris.data[:, index])
    temp= temp[temp[:].argsort()]
    feature=[]
    #计算小于划分点和大于划分点的结果数量集
    for i in range(len(temp)-1):
        counter1=[0,0,0]
        counter2=[0,0,0]
        counter1[0]=np.sum(iris.target[:i+1]==0)
        counter1[1] = np.sum(iris.target[:i + 1] == 1)
        counter1[2] = np.sum(iris.target[:i + 1] == 2)
        counter2[0] = np.sum(iris.target[i + 1:] == 0)
        counter2[1] = np.sum(iris.target[i + 1:] == 1)
        counter2[2] = np.sum(iris.target[i + 1:] == 2)
        feature.append([(temp[i]+temp[i+1])/2,counter1,counter2])
    minEntropy=sys.maxsize
    razor=feature[0][0]
    #对每个划分点计算熵
    for i in range(len(feature)):
        leng=i+1
        remain = length - leng
        d1=getEntropy(feature[i][1])
        d2=getEntropy(feature[i][2])
        remain=length-leng
        entropy=(leng/length)*d1+(remain/length)*d2
        #选择熵最小(即信息增益最大)的特征作为划分点
        if entropy<minEntropy:
            minEntropy=entropy
            razor=feature[i][0]
    return razor

#连续值分割
def getRazor():
    a = []
    for i in range(len(iris.feature_names)):
        a.append(discretization(i))
    #print("切割点: ",a)
    return np.array(a)

#寻找最大索引
def findMaxIndex(dataSet):
    maxIndex = 0
    maxValue = -1
    for index, value in enumerate(dataSet):
        if value > maxValue:
            maxIndex = index
            maxValue = value
    return maxIndex

#递归生成树
#featureSet: 特征集, dataSet: 数据集, counterSet: 三种花的个数
def tree(featureSet, dataSet, counterSet):
    if (counterSet[0] == 0 and counterSet[1] == 0 and counterSet[2] != 0):
        return iris.target_names[2]
    if (counterSet[0] != 0 and counterSet[1] == 0 and counterSet[2] == 0):
        return iris.target_names[0]
    if (counterSet[0] == 0 and counterSet[1] != 0 and counterSet[2] == 0):
        return iris.target_names[1]
    if len(featureSet) == 0:
        return iris.target_names[findMaxIndex(counterSet)]
    if len(dataSet) == 0:
        return []
    res = sys.maxsize
    final = 0
    for feature in featureSet:
        i = razors[feature]
        set1 = []
        set2 = []

        counter1 = [0, 0, 0]
        counter2 = [0, 0, 0]

        for data in dataSet:
            index = int(data[-1])

            if data[feature] < i:
                set1.append(data)
                counter1[index] = counter1[index] + 1
            elif data[feature] >= i:
                set2.append(data)
                counter2[index] = counter2[index] + 1

        #计算属性的熵
        a = (len(set1) * getEntropy(counter1) + len(set2) * getEntropy(counter2)) / len(dataSet)
        #获得熵最小的属性作为树节点(即信息增益最大的属性)
        if a < res:
            res = a
            final = feature

    featureSet.remove(final)
    child = [0, 0, 0]
    child[0] = final

    #递归生成其他树节点
    child[1] = tree(featureSet, set1, counter1)
    child[2] = tree(featureSet, set2, counter2)
    return child

#模型评估
def judge(test_data, tree):
    root = "unknow"
    while (len(tree) > 0):
        if isinstance(tree, str) and tree in iris.target_names:
            return tree
        root = tree[0]
        if (isinstance(root, str)):
            return root
        if isinstance(root, int):
            if test_data[root] < razors[root] and tree[1] != []:
                tree = tree[1]
            elif tree[2] != [] and (tree[1] == [] or (test_data[root] >= razors[root])):
                tree = tree[2]
    return root


if __name__ == '__main__':

    iris = datasets.load_iris()

    #随机获得60%的训练集和40%的测试集
    length = len(iris.target)
    [trainData, testData] = divideData()
    num = [0, 0, 0]
    for row in iris.data:
        num[int(row[-1])] = num[int(row[-1])] + 1
    #连续值分割
    razors = getRazor()

    #递归生成树
    tree = tree(list(range(len(iris.feature_names))), trainData,
                     [np.sum(trainData[:, -1] == 0), np.sum(trainData[:, -1] == 1)
                      , np.sum(trainData[:, -1] == 2)])
    print("本次选取的训练集构建出的树： ", tree)

    #使用测试集进行模型评估
    index = 0
    right = 0
    for data in testData:
        predict= judge(testData[index], tree)
        truth = iris.target_names[int(testData[index][-1])]
        index = index + 1
        if predict == truth:
            right = right + 1
    print("决策树正确率: ", format(right / index*100,".2f"),"%")
