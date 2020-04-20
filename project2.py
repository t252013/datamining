import numpy as np
D=np.loadtxt('iris.txt',encoding='utf-8', usecols=(0,1,2,3),delimiter=',')  #读取数据
n=D.shape[0]    #样本个数
feature=[]
def kernel(x1,x2):  #定义核函数
    return np.dot(x1,x2)*np.dot(x1,x2)  #齐次二次，是两个特征向量内积的平方
k=np.ones([n,n])
for i in range(0,n):
    for j in range(0,n):
        k[i][j]=kernel(D[i,:],D[j,:])
print('齐次二次核：',k)
kmean=np.mean(k,axis=0)    #均值
Ck=k-np.ones((k.shape[0],1),dtype=float)*kmean #中心化
kstd=np.std(k,axis=0)    #求出标准差
Nk=Ck/kstd  #标准化
print("核矩阵中心化：", Ck)
print("核矩阵标准化：",Nk)

def trs():  #转置函数
    vl=np.vsplit(D,n) #按行分割矩阵
    for i in range (0,len(vl)):
        vl[i]=vl[i][0]  #去括号
        nf=[]
        for j in range (0,len(vl[i])):
            nf.append(vl[i][j]*vl[i][j]) #取平方
        for k in range (0,len(vl[i])):
            for x in range(k+1,len(vl[i])):
                if x-k>=1:
                    nf.append((2**0.5)*vl[i][k]*vl[i][x])   #乘根号2
        feature.append(nf)

trs()
D1=np.array(feature)    #转为矩阵
print('特征空间矩阵',D1)
D2=np.dot(D1,D1.T)  #将特征空间矩阵求内积
Dmean = np.mean(D1,axis=0)  #均值
CD1=D1-np.ones((D1.shape[0],1),dtype=float)*Dmean   #中心化
std=np.std(D1,axis=0)  #标准差
ND1=CD1/std #标准化
print('验证点积：',D2,'与齐次二次核K相同')
print('中心化',CD1)
print('标准化',ND1)


