import numpy as np
import itertools
import matplotlib.pyplot as plt

a=np.loadtxt('magic04.txt',encoding='utf-8', usecols=(0,1,2,3,4,5,6,7,8,9),delimiter=',')   #读Magic04数据集前十个属性为矩阵
amean=np.mean(a,axis=0) #压缩行，求每一列的均值向量
print('均值向量为',amean)
arr1=np.ones((19020,1))    #生成全1矩阵
z=a-arr1*amean  #生成中心数据矩阵
cov1=np.dot(z.T,z)/19020    #计算样本协方差矩阵作为中心数据矩阵列之间的内积
cov2=np.cov(a.T)    #计算样本协方差矩阵作为中心数据点之间的外积
v1=z[:,0]   #属性一的中心属性向量
v2=z[:,1]   #属性二中的心属性向量
num = np.dot(v1.T ,v2) #计算两个向量的点乘
denom = np.linalg.norm(v1) * np.linalg.norm(v2) #计算两个向量的范数乘积
cos = num / denom #相除得余弦值
print('余弦值为：',cos)  #输出cos值
x1 = v1
y1 = v2
fig = plt.figure()  # 创建画图窗口
ax = fig.add_subplot(1, 1, 1)  # 将画图窗口分成1行1列，选择第一块区域作子图
ax.set_title('scatter')    #设置标题
ax.set_xlabel('v1')    #h横轴为v1
ax.set_ylabel('v2')    #纵轴为v2
ax.scatter(x1, y1,c='k', marker='.')  #画散点图
plt.show()  #显示图
def normfun(x,mu, sigma):   #正态分布函数
    pdf = np.exp(-((x - mu)**2) / (2* sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return pdf
x=np.arange(0,210,1)    #起点为0，终点为210，步长为1
y=normfun(x,amean[0],v1.std())
plt.plot(x,y, color='red',linewidth = 1.5)
plt.title('pdf')    #标题：概率密度函数
plt.xlabel('v1')    #x轴
plt.ylabel('f(v1)') #y轴
plt.show()

avar=np.var(a,axis=0) #对行压缩，求各列属性的方差
varmin=np.min(avar) #定义最小方差
varmax=np.max(avar) #定义最大方差

print('第',np.argwhere( avar == varmin)+1,'个属性方差最小，为：',varmin)
print('第',np.argwhere( avar == varmax)+1,'个属性方差最大，为：',varmax)   #输出

covmin=np.min(np.cov(a.T))  #最小协方差
covmax=np.max(np.cov(a.T))  #最大

print('协方差最小的一对属性的标号是：',np.argwhere( np.cov(a.T) == covmin)+1,'协方差是：',covmin)
print('协方差最大的一对属性的标号是：',np.argwhere( np.cov(a.T) == covmax)+1,'协方差是：',covmax)
