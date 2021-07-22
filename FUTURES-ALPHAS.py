import os
os.chdir('D:/data-vietquant/futures-alpha-rolling')


import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import random
from random import randint
from glob import  glob

from cvxopt import matrix
from cvxopt import solvers
from tabulate import tabulate
from datetime import datetime
import itertools

class infoTest:
    def __init__(self):
        pass
    def calculateSharpe(self,npArray):
        sr = npArray.mean()/npArray.std() * np.sqrt(252)
        return sr
    def max_drawdown(self,booksize,returnSeries):
        mdd = 0
        X = returnSeries +booksize
        peak = X[0]
        for x in X:
            if x > peak:
                peak = x
            dd = (peak - x) / booksize
            if dd > mdd:
                mdd = dd
        pd.Series(X).plot()
        plt.show()
        return mdd, print(X)
        # rets is array of returns
    def randomAllocateWeigh(self,rets):
        remaining = 1
        weigh = []
        for i in range(len(rets)):
            tempWeigh = round(random(),2)
            weigh.append(tempWeigh)
            remaining = remaining-tempWeigh
        weigh = np.asarray(weigh) / np.sum(weigh)
        # print(np.sum(weigh))
        portfolio = []
        for i in range(len(rets)):
            if len(portfolio) ==0:
                portfolio = rets[i] * weigh[i]
            else:
                portfolio += rets[i] * weigh[i]
        # portfolio = np.asarray(portfolio)/np.sum(weigh)
        return weigh,portfolio
    def randomAllocateListReturns(self,df):
        remaining = 1
        weigh = []
        counter =0
        ret = []
        portfolio = []
        for (columnName, columnData) in df.iteritems():
            tempWeigh = round(random(), 2)
            weigh.append(tempWeigh)
        weigh = np.asarray(weigh) / np.sum(weigh)
        # print(np.sum(weigh))


        portfolio = []
        counter = 0
        for (columnName, columnData) in df.iteritems():
            if len(portfolio) == 0:
                portfolio = columnData * weigh[counter]
            else:
                portfolio += columnData * weigh[counter]
            counter+=1
        # portfolio = np.asarray(portfolio)/np.sum(weigh)
        return weigh, portfolio


    def allocateForMaxSharpe(self,df,itertimes):
        maxSharpe = 0
        maxWeigh = []
        finalPnl = []
        for i in range(itertimes):
            weigh, mergePnl = self.randomAllocateListReturns(df)
            tempSharpe = self.calculateSharpe(mergePnl)
            if tempSharpe >= maxSharpe:
                maxSharpe = tempSharpe
                maxWeigh = weigh
                finalPnl = mergePnl
        # maxWeigh = np.asarray(maxWeigh) / np.sum(maxWeigh)
        return maxSharpe, maxWeigh,finalPnl


    def allocateForMinDD(self,df,itertimes,booksize):
        minDD = 1;
        minDDWeigh = []
        finalPnl = []
        for i in range(itertimes):
            weigh, mergePnl = self.randomAllocateListReturns(df)
            tempDD = self.max_drawdown(booksize,mergePnl)
            if tempDD < minDD:
                minDD = tempDD
                minDDWeigh = weigh
                finalPnl = mergePnl
        return minDD, minDDWeigh,finalPnl

def trad_sharpe(dataframe,rf,upperbound,test):
    it=infoTest()
    cov = (dataframe.cov()).to_numpy()
    meanvec = (dataframe.mean()).to_numpy()
    meanvec = [i - rf for i in meanvec]
    P = matrix(cov, tc='d')
    q = matrix(np.zeros(len(meanvec)), (len(meanvec), 1), tc='d')
    G=[]
    for i in range(len(meanvec)):
        k=[0 for x in range(len(meanvec)-1)]
        k.insert(i, -1)
        G.append(k)
    for i in range(len(meanvec)):
        k=[-upperbound for x in range(len(meanvec)-1)]
        k.insert(i,1-upperbound)
        G.append(k)
    G=matrix(np.array(G))
    H = np.zeros(2*len(meanvec))
    h = matrix(H, tc='d')
    A = (matrix(meanvec)).trans()
    b = matrix([1], (1, 1), tc='d')
    sol = ((solvers.qp(P, q, G, h, A, b))['x'])
    solution = [x for x in sol]
    sum = 0
    for i in range(len(solution)):
        sum += solution[i]
    optimizedWeigh = [x / sum for x in solution]
    print(cum_ret1(test,optimizedWeigh))
    return optimizedWeigh

def copula_sharpe(dataframe,upperbound,cov,rf,test):
    it=infoTest()
    cov = cov.to_numpy()
    meanvec = (dataframe.std()).to_numpy()
    #meanvec = std
    meanvec = [i - rf for i in meanvec]
    P = matrix(cov, tc='d')
    q = matrix(np.zeros(len(meanvec)), (len(meanvec), 1), tc='d')
    G=[]
    for i in range(len(meanvec)):
        k=[0 for x in range(len(meanvec)-1)]
        k.insert(i, -1)
        G.append(k)
    for i in range(len(meanvec)):
        k=[-upperbound for x in range(len(meanvec)-1)]
        k.insert(i,1-upperbound)
        G.append(k)
    G=matrix(np.array(G))
    H = np.zeros(2*len(meanvec))
    h = matrix(H, tc='d')
    A = (matrix(meanvec)).trans()
    b = matrix([1], (1, 1), tc='d')
    sol = ((solvers.qp(P, q, G, h, A, b))['x'])
    solution = [x for x in sol]
    sum1 = 0
    for i in range(len(solution)):
        sum1 += solution[i]
    optimizedWeigh = [x / sum1 for x in solution]

    print(cum_ret1(test,optimizedWeigh))
    return optimizedWeigh

def copula_futures(list_group,list_normal,dataframe,booksize,upperbound, bounded_list,dataframe1,cova):
    # print(tabulate(dataframe.corr(method='pearson'), tablefmt="pipe", headers="keys"))
    # upperbound = 0.3
    it = infoTest()
    cov = (cova).to_numpy()
    meanvec = (dataframe.mean()).to_numpy()
    P = matrix(cov, tc='d')
    # print(P)
    q = matrix(np.zeros(len(meanvec)), (len(meanvec), 1), tc='d')
    G = []
    for i in range(len(meanvec)):
        k = [0 for x in range(len(meanvec) - 1)]
        k.insert(i, -1)
        G.append(k)
    for i in range(len(meanvec)):
        k = [-upperbound for x in range(len(meanvec) - 1)]
        k.insert(i, 1 - upperbound)
        G.append(k)
    k = [-bounded_list for i in range((list_normal))]
    for i in range((list_group)):
        k.insert(i,1-bounded_list)
    G.append(k)
    k = [bounded_list for i in range((list_normal))]
    for i in range((list_group)):
        k.insert(i,  bounded_list-1)
    G.append(k)
    G = matrix(np.array(G))
    H = np.zeros(2 * len(meanvec)+2)
    h = matrix(H, tc='d')
    A = (matrix(meanvec)).trans()
    b = matrix([1], (1, 1), tc='d')
    # print('G',G)
    # print('h',h)
    # print('A',A)
    # print('b',b)
    sol = (solvers.qp(P, q, G, h, A, b))['x']
    solution = [x for x in sol]
    sum = 0
    for i in range(len(solution)):
        sum += solution[i]
    optimizedWeigh = [x / sum for x in solution]
    print(optimizedWeigh)
    merge = []
    counter = 0
    for (columnName, columnData) in dataframe1.iteritems():
        # print(real[counter])
        if len(merge) == 0:
            merge = dataframe1[columnName] * optimizedWeigh[counter]
        else:
            merge = merge + dataframe1[columnName] * optimizedWeigh[counter]
        counter += 1
    # print(merge)
    # print('dd,', it.max_drawdown(booksize=booksize, returnSeries=merge))
    print('sharpe,', it.calculateSharpe(merge))
    merge = merge*10
    # print('value',merge)
    # merge.to_csv(r'/home/hoainam/PycharmProjects/multi_strategy/v_multi/f1m.csv')
    # print(np.cumsum(merge))
    plt.plot(np.cumsum(merge))
    # plt.grid(True)
    # plt.legend(('old', 'maxsharpe', 'minDD'))
    # plt.show()
    return optimizedWeigh

def trad_futures(list_group,list_normal,dataframe,booksize,upperbound, bounded_list,dataframe1):
    # print(tabulate(dataframe.corr(method='pearson'), tablefmt="pipe", headers="keys"))
    # upperbound = 0.3
    it = infoTest()
    cov = (dataframe.cov()).to_numpy()
    meanvec = (dataframe.mean()).to_numpy()
    P = matrix(cov, tc='d')
    # print(P)
    q = matrix(np.zeros(len(meanvec)), (len(meanvec), 1), tc='d')
    G = []
    for i in range(len(meanvec)):
        k = [0 for x in range(len(meanvec) - 1)]
        k.insert(i, -1)
        G.append(k)
    for i in range(len(meanvec)):
        k = [-upperbound for x in range(len(meanvec) - 1)]
        k.insert(i, 1 - upperbound)
        G.append(k)
    k = [-bounded_list for i in range((list_normal))]
    for i in range((list_group)):
        k.insert(i,1-bounded_list)
    G.append(k)
    k = [bounded_list for i in range((list_normal))]
    for i in range((list_group)):
        k.insert(i,  bounded_list-1)
    G.append(k)
    G = matrix(np.array(G))
    H = np.zeros(2 * len(meanvec)+2)
    h = matrix(H, tc='d')
    A = (matrix(meanvec)).trans()
    b = matrix([1], (1, 1), tc='d')
    # print('G',G)
    # print('h',h)
    # print('A',A)
    # print('b',b)
    sol = (solvers.qp(P, q, G, h, A, b))['x']
    solution = [x for x in sol]
    sum = 0
    for i in range(len(solution)):
        sum += solution[i]
    optimizedWeigh = [x / sum for x in solution]
    print(optimizedWeigh)
    merge = []
    counter = 0
    for (columnName, columnData) in dataframe1.iteritems():
        # print(real[counter])
        if len(merge) == 0:
            merge = dataframe1[columnName] * optimizedWeigh[counter]
        else:
            merge = merge + dataframe1[columnName] * optimizedWeigh[counter]
        counter += 1
    # print(merge)
    # print('dd,', it.max_drawdown(booksize=booksize, returnSeries=merge))
    print('sharpe,', it.calculateSharpe(merge))
    merge = merge*10
    # print('value',merge)
    # merge.to_csv(r'/home/hoainam/PycharmProjects/multi_strategy/v_multi/f1m.csv')
    # print(np.cumsum(merge))
    plt.plot(np.cumsum(merge))
    # plt.grid(True)
    # plt.legend(('old', 'maxsharpe', 'minDD'))
    # plt.show()
    return optimizedWeigh

it = infoTest()

def cum_ret1(df,weight):
    ret = np.dot(df,weight)
    sharpe = it.calculateSharpe(ret)
    return sharpe

#whole period cumulative return
def cum_ret(df1, weight1, df2, weight2, df3, weight3,df4, weight4, df5, weight5, df6, weight6,df7, weight7):
    return_1 = np.dot(df1, weight1).reshape(-1,1)
    return_2 = np.dot(df2, weight2).reshape(-1,1)
    return_3 = np.dot(df3, weight3).reshape(-1,1)
    return_4 = np.dot(df4, weight4).reshape(-1,1)
    return_5 = np.dot(df5, weight5).reshape(-1,1)
    return_6 = np.dot(df6, weight6).reshape(-1,1)
    return_7 = np.dot(df7, weight7).reshape(-1,1)
    dis_ret = np.vstack((return_1, return_2,return_3,return_4,return_5,return_6,return_7))
    cum_ret = np.cumsum(dis_ret)
    sharpe = it.calculateSharpe(dis_ret)
    pd.Series(cum_ret).plot()
    plt.legend([1,2,3])

    return print(sharpe)


#import data:

#train:
train1 = pd.read_csv('train1.csv',parse_dates=['datetime'])
train2 = pd.read_csv('train2.csv',parse_dates=['datetime'])
train3 = pd.read_csv('train3.csv',parse_dates=['datetime'])
train4 = pd.read_csv('train4.csv',parse_dates=['datetime'])
train5 = pd.read_csv('train5.csv',parse_dates=['datetime'])
train6 = pd.read_csv('train6.csv',parse_dates=['datetime'])
train7 = pd.read_csv('train7.csv',parse_dates=['datetime'])

#test:

test1 = pd.read_csv('test1.csv',parse_dates=['datetime'],index_col=['datetime'])
test2 = pd.read_csv('test2.csv',parse_dates=['datetime'],index_col=['datetime'])
test3 = pd.read_csv('test3.csv',parse_dates=['datetime'],index_col=['datetime'])
test4 = pd.read_csv('test4.csv',parse_dates=['datetime'],index_col=['datetime'])
test5 = pd.read_csv('test5.csv',parse_dates=['datetime'],index_col=['datetime'])
test6 = pd.read_csv('test6.csv',parse_dates=['datetime'],index_col=['datetime'])
test7 = pd.read_csv('test7.csv',parse_dates=['datetime'],index_col=['datetime'])


#Clayton covariance matrix:
cov1_C = pd.read_csv('train1_covC.csv', header= None,skiprows = 1 )
cov1_C = cov1_C.iloc[:,1:]

cov2_C = pd.read_csv('train2_covC.csv', header= None,skiprows = 1 )
cov2_C = cov2_C.iloc[:,1:]

cov3_C = pd.read_csv('train3_covC.csv', header= None,skiprows = 1 )
cov3_C = cov3_C.iloc[:,1:]

cov4_C = pd.read_csv('train4_covC.csv', header= None,skiprows = 1 )
cov4_C = cov4_C.iloc[:,1:]

cov5_C = pd.read_csv('train5_covC.csv', header= None,skiprows = 1 )
cov5_C = cov5_C.iloc[:,1:]

cov6_C = pd.read_csv('train6_covC.csv', header= None,skiprows = 1 )
cov6_C = cov6_C.iloc[:,1:]

cov7_C = pd.read_csv('train7_covC.csv', header= None,skiprows = 1 )
cov7_C = cov7_C.iloc[:,1:]


#Gaussian covariance matrix
cov1_G = pd.read_csv('train1_covG.csv', header= None,skiprows = 1 )
cov1_G = cov1_G.iloc[:,1:]

cov2_G = pd.read_csv('train2_covG.csv', header= None,skiprows = 1 )
cov2_G = cov2_G.iloc[:,1:]

cov3_G = pd.read_csv('train3_covG.csv', header= None,skiprows = 1 )
cov3_G = cov3_G.iloc[:,1:]

cov4_G = pd.read_csv('train4_covG.csv', header= None,skiprows = 1 )
cov4_G = cov4_G.iloc[:,1:]

cov5_G = pd.read_csv('train5_covG.csv', header= None,skiprows = 1 )
cov5_G = cov5_G.iloc[:,1:]

cov6_G = pd.read_csv('train6_covG.csv', header= None,skiprows = 1 )
cov6_G = cov6_G.iloc[:,1:]

cov7_G = pd.read_csv('train7_covG.csv', header= None,skiprows = 1 )
cov7_G = cov7_G.iloc[:,1:]



trad_w1 = trad_futures(4,5,train1, 10**9,0.16,0.36,test1)
Clayton_w1 = copula_futures(4,5, train1, 10**9, 0.16, 0.36, test1, cov1_C)
Gaussian_w1 = copula_futures(4,5,train1, 10**9, 0.16, 0.36, test1, cov1_G)
plt.show()


trad_w2 = trad_futures(4,5,train2, 10**9,0.16,0.36,test2)
Clayton_w2 = copula_futures(4,5, train2, 10**9, 0.16, 0.36, test2, cov2_C)
Gaussian_w2 = copula_futures(4,5,train2, 10**9, 0.16, 0.36, test2, cov2_G)
plt.show()


trad_w3 = trad_futures(4,5,train3, 10**9,0.16,0.36,test3)
Clayton_w3 = copula_futures(4,5, train3, 10**9, 0.16, 0.36, test3, cov3_C)
Gaussian_w3 = copula_futures(4,5,train3, 10**9, 0.16, 0.36, test3, cov3_G)
plt.show()


trad_w4 = trad_futures(4,5,train4, 10**9,0.16,0.36,test4)
Clayton_w4 = copula_futures(4,5, train4, 10**9, 0.16, 0.36, test4, cov4_C)
Gaussian_w4 = copula_futures(4,5,train4, 10**9, 0.16, 0.36, test4, cov4_G)
plt.show()


trad_w5 = trad_futures(4,5,train5, 10**9,0.16,0.36,test5)
Clayton_w5 = copula_futures(4,5, train5, 10**9, 0.16, 0.36, test5, cov5_C)
Gaussian_w5 = copula_futures(4,5,train5, 10**9, 0.16, 0.36, test5, cov5_G)
plt.show()


trad_w6 = trad_futures(4,5,train6, 10**9,0.16,0.36,test6)
Clayton_w6 = copula_futures(4,5, train6, 10**9, 0.16, 0.36, test6, cov6_C)
Gaussian_w6 = copula_futures(4,5,train6, 10**9, 0.16, 0.36, test6, cov6_G)
plt.show()


trad_w7 = trad_futures(4,5,train7, 10**9,0.16,0.36,test7)
Clayton_w7 = copula_futures(4,5, train7, 10**9, 0.16, 0.36, test7, cov7_C)
Gaussian_w7 = copula_futures(4,5,train7, 10**9, 0.16, 0.36, test7, cov7_G)
plt.show()

#
# trad_w1 = trad_sharpe(train1, 0,0.1,test1)
# trad_w2 = trad_sharpe(train2,0, 0.1,test2 )
# # trad_w3 = trad_sharpe(train3, 0,0.1,test3 )
# trad_w4 = trad_sharpe(train4, 0,0.1,test4 )
# trad_w5 = trad_sharpe(train5, 0,0.1,test5 )
# trad_w6 = trad_sharpe(train6, 0,0.1,test6 )
# trad_w7 = trad_sharpe(train7, 0,0.1,test7 )
#
#
#
# Clayton_w1 = copula_sharpe(train1,0.1,cov1_C,0 ,test1)
# Clayton_w2 = copula_sharpe(train2,0.1,cov2_C,0 ,test2)
# Clayton_w3 =copula_sharpe(train3,0.1,cov3_C,0 ,test3)
# Clayton_w4 = copula_sharpe(train4,0.1,cov4_C,0 ,test4)
# Clayton_w5 = copula_sharpe(train5,0.1,cov5_C,0 ,test5)
# Clayton_w6 =copula_sharpe(train6,0.1,cov6_C,0 ,test6)
# Clayton_w7 =copula_sharpe(train7,0.1,cov7_C,0 ,test7)
#
#
#
# Gaussian_w1 = copula_sharpe(train1,0.1,cov1_G,0 ,test1)
# Gaussian_w2 = copula_sharpe(train2,0.1,cov2_G,0 ,test2)
# Gaussian_w3 = copula_sharpe(train3,0.1,cov3_G,0 ,test3)
# Gaussian_w4 = copula_sharpe(train4,0.1,cov4_G,0 ,test4)
# Gaussian_w5 = copula_sharpe(train5,0.1,cov5_G,0 ,test5)
# Gaussian_w6 = copula_sharpe(train6,0.1,cov6_G,0 ,test6)
# Gaussian_w7 = copula_sharpe(train7,0.1,cov7_G,0 ,test7)


# #PERIOD 1
# port1 = {'trad': trad_w1,
#         'Clay':Clayton_w1,
#         'Gaus': Gaussian_w1}
# port1 = pd.DataFrame(port1)
# port1.index = train1.columns
# port1
#
#
# #PERIOD 2
# port2 = {'trad': trad_w2,
#         'Clay':Clayton_w2,
#         'Gaus': Gaussian_w2}
# port2 = pd.DataFrame(port2)
# port2.index = train1.columns
# port2
#
#
# #PERIOD 3
# port3 = {'trad': trad_w3,
#         'Clay':Clayton_w3,
#         'Gaus': Gaussian_w3}
# port3 = pd.DataFrame(port3)
# port3.index = train1.columns
# port3
#
#
#
#

cum_ret(test1,trad_w1, test2, trad_w2, test3, trad_w3,test4,
        trad_w4, test5, trad_w5, test6, trad_w6, test7,trad_w7)
cum_ret(test1,Clayton_w1, test2, Clayton_w2, test3, Clayton_w3,
        test4,Clayton_w4, test5, Clayton_w5, test6, Clayton_w6,test7, Clayton_w7)
cum_ret(test1,Gaussian_w1, test2, Gaussian_w2, test3, Gaussian_w3,
        test4,Gaussian_w4, test5, Gaussian_w5, test6, Gaussian_w6,test7, Gaussian_w7)
plt.show()
#

Trad = [cum_ret1(test1,trad_w1),cum_ret1(test2,trad_w2),
        cum_ret1(test3,trad_w3),cum_ret1(test4,trad_w4),
        cum_ret1(test5,trad_w5),cum_ret1(test6,trad_w6),cum_ret1(test7,trad_w7)]

Clay = [cum_ret1(test1,Clayton_w1),cum_ret1(test2,Clayton_w2),cum_ret1(test3,Clayton_w3),
        cum_ret1(test4,Clayton_w4),cum_ret1(test5,Clayton_w5),cum_ret1(test6,Clayton_w6),
        cum_ret1(test7,Clayton_w7)]


Gauss = [cum_ret1(test1,Gaussian_w1),cum_ret1(test2,Gaussian_w2),cum_ret1(test3,Gaussian_w3)
                ,cum_ret1(test4,Gaussian_w4),cum_ret1(test5,Gaussian_w5),cum_ret1(test6,Gaussian_w6),
                                                                          cum_ret1(test7,Gaussian_w7)]



Sharpe = {'trad' : Trad, 'Clay' : Clay, 'Gauss' : Gauss}

Sharpe = pd.DataFrame(Sharpe)

print(Sharpe)


