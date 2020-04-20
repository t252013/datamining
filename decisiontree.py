import numpy as np
from math import log
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz
import pydotplus

data= np.loadtxt('iris.txt', delimiter=',', usecols=(0, 1, 2, 3))  #取前四列特征为data
target= np.loadtxt('iris.txt',str,delimiter=',',unpack=True, usecols=(4))   #取最后一列类别为target
args = np.mean(data,axis = 0)   #均值
def Entropy(target1):       #计算熵的函数
    n = target1.size
    count = {}  #统计次数
    for currentLabel in target1:    #对每组特征向量进行统计
        if currentLabel not in count.keys():    #如果标签没有放入统计次数的字典，添加进去
            count[currentLabel] = 0 #标签计数
        count[currentLabel] += 1
    entropy = 0.0   #熵
    for key in count:
        prob = float(count[key]) / n    #选择该标签的概率
        entropy -= prob * log(prob, 2)  #根据公式计算
    return entropy  #返回
def splitdata(data1, idxfea):       #划分数据集
    arg = args[idxfea]
    less = []
    greater = []
    n = len(data1)
    for idx in range(n):    #遍历数据集
        d = data1[idx]
        if d[idxfea] < arg:
            less.append(idx)
        else:
            greater.append(idx)
    return less, greater
def chooseBest(data1, target1):     #获取最大的信息增益的feature
    nfea = len(data1[0])
    n = len(target1)
    baseEntropy = Entropy(target1)  #计算熵
    bestGain = -1
    for i in range(nfea):   #遍历所有特征
        curEntropy = 0
        less, greater = splitdata(data1, i)
        prob_less = float(len(less)) / n
        prob_greater = float(len(greater)) / n
        curEntropy += prob_less * Entropy(target1[less])
        curEntropy += prob_greater * Entropy(target1[greater])

        infoGain = baseEntropy - curEntropy
        print("特征",i+1,"的信息增益为：",infoGain)   #输出每个feature的信息增益
        if (infoGain > bestGain):   #计算信息增益
            bestGain = infoGain #更新信息增益，找到最大的信息增益
            bestIdx = i #记录信息增益最大的特征的索引值
    return bestIdx,bestGain #返回信息增益最大特征的索引值
print("计算出的信息熵为：",Entropy(target))
BestInx,BestGain=chooseBest(data,target)
print("计算出最佳信息增益feature为：",BestInx+1)
print("计算出最佳信息增益为：",BestGain)

data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.3, random_state=1)#定义训练样本和测试样本
clf=tree.DecisionTreeClassifier()
clf.fit(data_train,target_train)
target_predict=clf.predict(data_test)
print("测试目标为：")
print(target_test)      #测试目标
print("决策树预测目标为：")
print(target_predict)   #预测目标
print("预测准确率、召回率为：")
print(metrics.classification_report(target_predict, target_test))   #预测准确率、召回率

with open("iris.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)
dot_data = tree.export_graphviz(clf, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)

graph.write_pdf("iris.pdf")     #输出PDF
