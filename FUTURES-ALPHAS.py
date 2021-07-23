import os
os.chdir('D:/data-vietquant/futures-alpha-rolling')


import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import random
from random import randint
from glob import  glob
import matplotlib.gridspec as gridspec
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
    # plt.plot(np.cumsum(merge))
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
    # plt.plot(np.cumsum(merge))
    # plt.grid(True)
    # plt.legend(('old', 'maxsharpe', 'minDD'))
    # plt.show()
    return optimizedWeigh

def MDD_constrained_futures(dataframe, bound_group, bound_alpha, alpha, v3):
    dataframe = dataframe.iloc[:,1:]/10**3
    # alpha = 0.95
    v3 = v3
    #input to be cumulative return: y,u,z,epsilon
    cumulative = np.cumsum(dataframe, axis = 0)
    cumulative= pd.DataFrame(cumulative)
    print(cumulative)

    #inequality constraint
    #1st constraint: wi >=0 <=> -wi <= 0
    J = dataframe.shape[0] #number of row
    N = dataframe.shape[1] #number of col
    g11 = pd.DataFrame(np.diag(np.ones(N)))     #xk
    g12 = pd.DataFrame(np.zeros([N,J*2+1]))     #uk, zk, epsilon
    # print(g12)
    G1 = pd.concat([g11, g12], axis =1, ignore_index= True)
    G1 = -G1
    # print(matrix(np.array(G1)))
    h1  = pd.DataFrame(np.zeros(N).reshape(-1,1))
    # print(h1)


    #2nd constraint: -yk*x +uk - zk - epsilon <= 0
    g21 =  -cumulative                      #xk
    g22 = pd.DataFrame(np.diag(np.ones(J))) #uk
    g23 = -pd.DataFrame(np.diag(np.ones(J))) #zk
    g24 = - pd.DataFrame(np.ones([J,1]))    #epsilon
    G2 = pd.concat([g21,g22,g23,g24], axis = 1, ignore_index = True)
    # print(G2)
    h2 = pd.DataFrame(np.zeros(J).reshape(-1,1))

    #3rd constraint: zk >= 0 <=> -zk <= 0
    g31 = pd.DataFrame(np.zeros([J,N])) #xk
    g32 = pd.DataFrame(np.zeros([J,J])) #uk
    g33 = pd.DataFrame(np.diag(np.ones(J)))    #zk
    g34 = pd.DataFrame(np.zeros([J,1]))   #epsilon
    G3 = pd.concat([g31, g32, -g33, g34], axis = 1, ignore_index = True)
    # print(G3)

    h3 = pd.DataFrame(np.zeros(J).reshape(-1,1))


    #4th constraint: yk*x - uk <= 0
    g41 = cumulative                        #xk
    g42 = -pd.DataFrame(np.diag(np.ones(J))) #uk
    g43 = pd.DataFrame(np.zeros([J,J])) #zk
    g44 = pd.DataFrame(np.zeros([J,1])) #epsilion
    G4 = pd.concat([g41, g42, g43, g44], axis = 1, ignore_index= True)
    # print(G4)
    h4 = pd.DataFrame(np.zeros(J).reshape(-1,1))

    #5th constraint: uk-1  -- uk <= 0

    g51 = pd.DataFrame(np.zeros([J,N])) #xk
    g52 =[]
    a = [0 for x in range(J-1)]
    a.insert(0,-1)
    g52.append(a)
    for i in range(J-1):
        k = [0 for x in range(J - 2)]
        k.insert(i,-1)
        k.insert(i,1)
        g52.append(k)
    g52 = pd.DataFrame(g52) #uk
    # print(g52)
    g53 = pd.DataFrame(np.diag(np.zeros(J))) #zk
    g54 = pd.DataFrame(np.zeros([J,1])) #epsilon
    G5 = pd.concat([g51, g52,g53, g54], axis = 1, ignore_index=True)
    # print(G5)
    h5 = pd.DataFrame(np.zeros(J).reshape(-1,1))


    #6th constraint: wi <= 0.16
    g61 = pd.DataFrame(np.diag(np.ones(N)))     #xk
    g62 = pd.DataFrame(np.zeros([N,J*2+1]))     #uk, zk, epsilon
    # print(g12)
    G6 = pd.concat([g61, g62], axis =1, ignore_index= True)
    h6  = [bound_alpha for x in range(N)]
    h6 = pd.DataFrame(np.array(h6).reshape(-1,1))
    # print(G6)
    # print(h6)

    #7th constraint: proportion v3

    g71 = [0 for x in range(N+J)]
    g72=  [1/((1-alpha)*J) for y in range(J)]
    G7 = np.append(g71,g72)
    G7 = np.append(G7,1)
    G7 = G7.reshape(1,-1)
    G7 = pd.DataFrame(G7)
    # print(G7)
    h7 = pd.Series(v3)



    #total inequality matrix
    G = pd.concat([G1, G2, G3, G4, G5, G6,G7], axis = 0 , ignore_index=True)
    G = matrix(np.array(G))
    h = pd.concat([h1, h2,h3, h4, h5, h6,h7], axis = 0,ignore_index=True)
    h = matrix(np.array(h))
    # print(G)
    # print(h)
    # breakpoint()


    #equality constraint:
    # total = 1
    a11 =[1 for x in range(N)]
    a12 = [0 for x in range(2*J+1)]
    A1 = np.append(a11,a12).reshape(1, -1)
    A1 = matrix(A1, tc= 'd')
    # print(A)
    b1 = matrix([1],(1,1), tc='d')
    # print(b)


    #group 1 = 0.36
    a21 = [1 for x in range(N-5)]
    a22 = [0 for x in range(2*J + 1+ 5)]
    A2 = np.append(a21,a22).reshape(1, -1)
    A2 = matrix(A2, tc= 'd')
    b2 = matrix([bound_group], (1,1), tc='d')
    # print(A2)
    # print(b2)
    #
    A = pd.concat([pd.DataFrame(A1), pd.DataFrame(A2)], axis  =1).T
    A = matrix(np.array(A))
    # print(A)

    b = matrix(np.array(pd.concat([pd.DataFrame(b1), pd.DataFrame(b2)], axis = 1).T))
    # print(b)



    #objective function: max return
    c1 = (cumulative.iloc[-1,:]).to_numpy()
    c2 = [0 for x in range(2*J+1)]
    c = np.append(c1, c2).reshape(1,-1)
    c = -matrix(c.T)
    # print(c)

    # #
    # c1 = [0 for x in range(N+J)]
    # c2=  [1/((1-alpha)*J) for y in range(J)]
    # c = np.append(c1,c2)
    # c = np.append(c,1)
    # c = c.reshape(1,-1)
    # c = pd.DataFrame(c)
    # c = matrix(np.array(c).T)
    # print(c)



    sol = solvers.lp(c, G, h, A, b,solver ='glpk')['x']
    # print(sol['x'])
    # print(solution[:8])
    solution = [x for x in sol]
    solution = solution[:9]
    return solution


it = infoTest()

def cum_ret1(df,weight):
    ret = np.dot(df,weight)
    sharpe = it.calculateSharpe(ret)
    return sharpe

#whole period cumulative return
def cum_ret(df1, weight1, df2, weight2, df3, weight3,df4, weight4, df5, weight5, df6, weight6,df7, weight7, booksize):
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

    #mdd plot
    mdd = 0
    X = cum_ret + booksize
    peak = X[0]
    ddsr = []
    count =0
    # calculate dd
    for x in X:
        if x > peak:
            peak = x
        dd = (peak - x) / booksize
        ddsr.append(dd)
        # setting new mdd:
        if dd > mdd:
            mdd = dd
            count +=1
    # pd.Series(cum_ret).plot()
    # plt.legend([1,2,3])
    ax1 = plt.subplot(gs[0, :])
    ax2 = plt.subplot(gs[1:, :])
    ax1.plot(pd.Series(ddsr))
    ax2.plot(pd.Series(cum_ret))
    print('count: ', count)
    print('25th quantile: %s, meadian: %s, 75th quantile: %s, 90th quantile: %s, max drawdown: %s, average drawdown: %s' %(np.quantile(ddsr, 0.25), np.quantile(ddsr, 0.5),np.quantile(ddsr, 0.75),
                                                                                                        np.quantile(ddsr, 0.90),np.max(ddsr), np.mean(ddsr)))
    return print('sharpe =', sharpe)

def md_calculator(booksize, weight, dataframe):
    dis_ret = np.dot(dataframe, weight)
    cum_ret = np.cumsum(dis_ret)
    mdd = 0
    X = cum_ret + booksize
    peak = X[0]
    ddsr = []
    #calculate dd
    for x in X:
        if x > peak:
            peak = x
        dd = (peak - x) / booksize
        ddsr.append(dd)
    #setting new mdd:
        if dd > mdd:
            mdd = dd
    ax1 = plt.subplot(gs[0, :])
    ax2 = plt.subplot(gs[1:, :])
    ax1.plot(pd.Series(ddsr))
    ax2.plot(pd.Series(cum_ret))
    # plt.tight_layout()
    # pd.Series(ddsr).plot()
    # plt.show()
    return print('25th quantile: %s, meadian: %s, 75th quantile: %s, 90th quantile: %s, max drawdown: %s, average drawdown: %s' %(np.quantile(ddsr, 0.25), np.quantile(ddsr, 0.5),
                                                                                                                                  np.quantile(ddsr, 0.75),np.quantile(ddsr, 0.90),
                                                                                                                                  np.max(ddsr), np.mean(ddsr)))



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



# #max SHARPE:
# MDDw1 = MDD_constrained_futures(train1,0.36, 0.16, 0.95, 0.06241379310344828)
# MDDw2 = MDD_constrained_futures(train2,0.36, 0.16, 0.95, 0.06448275862068965)
# MDDw3 = MDD_constrained_futures(train3,0.36, 0.16, 0.9277777777777777, 0.059310344827586215)
# MDDw4 = MDD_constrained_futures(train4,0.36, 0.16, 0.8944444444444444, 0.056206896551724145)
# MDDw5 = MDD_constrained_futures(train5,0.36, 0.16, 0.8833333333333333, 0.05517241379310345)
# MDDw6 = MDD_constrained_futures(train6,0.36, 0.16, 0.8722222222222222, 0.056206896551724145)
# MDDw7 = MDD_constrained_futures(train7,0.36, 0.16, 0.861111111111111, 0.05827586206896552)


# #max SHARPE/CDAR:
# MDDw1 = MDD_constrained_futures(train1,0.36, 0.16, 0.8722222222222222,0.05)
# MDDw2 = MDD_constrained_futures(train2,0.36, 0.16, 0.8722222222222222,0.05)
# MDDw3 = MDD_constrained_futures(train3,0.36, 0.16, 0.8722222222222222,0.05)
# MDDw4 = MDD_constrained_futures(train4,0.36, 0.16, 0.8722222222222222,0.05)
# MDDw5 = MDD_constrained_futures(train5,0.36, 0.16, 0.861111111111111, 0.05)
# MDDw6 = MDD_constrained_futures(train6,0.36, 0.16, 0.85, 0.05)
# MDDw7 = MDD_constrained_futures(train7,0.36, 0.16, 0.8722222222222222, 0.05)

#max E(r)/CDAR:
# MDDw1 = MDD_constrained_futures(train1,0.36, 0.16, 0.85,0.05)
# MDDw2 = MDD_constrained_futures(train2,0.36, 0.16, 0.85, 0.05)
# MDDw3 = MDD_constrained_futures(train3,0.36, 0.16, 0.85, 0.05)
# MDDw4 = MDD_constrained_futures(train4,0.36, 0.16, 0.85, 0.05)
# MDDw5 = MDD_constrained_futures(train5,0.36, 0.16, 0.85, 0.05)
# MDDw6 = MDD_constrained_futures(train6,0.36, 0.16, 0.85, 0.05)
# MDDw7 = MDD_constrained_futures(train7,0.36, 0.16, 0.85,  0.05)

#max sharpe/meandd: (good sharpe - good drawdown - not good return)
MDDw1 = MDD_constrained_futures(train1,0.36, 0.16, 0.9388888888888889,0.05)
MDDw2 = MDD_constrained_futures(train2,0.36, 0.16, 0.9388888888888889, 0.05)
MDDw3 = MDD_constrained_futures(train3,0.36, 0.16, 0.9388888888888889, 0.05)
MDDw4 = MDD_constrained_futures(train4,0.36, 0.16, 0.9388888888888889, 0.05)
MDDw5 = MDD_constrained_futures(train5,0.36, 0.16, 0.9388888888888889, 0.05)
MDDw6 = MDD_constrained_futures(train6,0.36, 0.16, 0.9055555555555556, 0.053103448275862074)
MDDw7 = MDD_constrained_futures(train7,0.36, 0.16, 0.8944444444444444,  0.05517241379310345)

#max return/meamdd:
# MDDw1 = MDD_constrained_futures(train1,0.36, 0.16, 0.95, 0.053103448275862074)
# MDDw2 = MDD_constrained_futures(train2,0.36, 0.16, 0.9166666666666666, 0.05827586206896552)
# MDDw3 = MDD_constrained_futures(train3,0.36, 0.16, 0.95, 0.06344827586206897)
# MDDw4 = MDD_constrained_futures(train4,0.36, 0.16, 0.9388888888888889, 0.05517241379310345)
# MDDw5 = MDD_constrained_futures(train5,0.36, 0.16, 0.9166666666666666, 0.05)
# MDDw6 = MDD_constrained_futures(train6,0.36, 0.16, 0.85, 0.05)
# MDDw7 = MDD_constrained_futures(train7,0.36, 0.16, 0.861111111111111,  0.05)

#max return/maxdd


trad_w1 = trad_futures(4,5,train1, 10**9,0.16,0.36,test1)
Clayton_w1 = copula_futures(4,5, train1, 10**9, 0.16, 0.36, test1, cov1_C)
Gaussian_w1 = copula_futures(4,5,train1, 10**9, 0.16, 0.36, test1, cov1_G)
trad_w2 = trad_futures(4,5,train2, 10**9,0.16,0.36,test2)

Clayton_w2 = copula_futures(4,5, train2, 10**9, 0.16, 0.36, test2, cov2_C)
Gaussian_w2 = copula_futures(4,5,train2, 10**9, 0.16, 0.36, test2, cov2_G)

trad_w3 = trad_futures(4,5,train3, 10**9,0.16,0.36,test3)
Clayton_w3 = copula_futures(4,5, train3, 10**9, 0.16, 0.36, test3, cov3_C)
Gaussian_w3 = copula_futures(4,5,train3, 10**9, 0.16, 0.36, test3, cov3_G)

trad_w4 = trad_futures(4,5,train4, 10**9,0.16,0.36,test4)
Clayton_w4 = copula_futures(4,5, train4, 10**9, 0.16, 0.36, test4, cov4_C)
Gaussian_w4 = copula_futures(4,5,train4, 10**9, 0.16, 0.36, test4, cov4_G)

trad_w5 = trad_futures(4,5,train5, 10**9,0.16,0.36,test5)
Clayton_w5 = copula_futures(4,5, train5, 10**9, 0.16, 0.36, test5, cov5_C)
Gaussian_w5 = copula_futures(4,5,train5, 10**9, 0.16, 0.36, test5, cov5_G)

trad_w6 = trad_futures(4,5,train6, 10**9,0.16,0.36,test6)
Clayton_w6 = copula_futures(4,5, train6, 10**9, 0.16, 0.36, test6, cov6_C)
Gaussian_w6 = copula_futures(4,5,train6, 10**9, 0.16, 0.36, test6, cov6_G)

trad_w7 = trad_futures(4,5,train7, 10**9,0.16,0.36,test7)
Clayton_w7 = copula_futures(4,5, train7, 10**9, 0.16, 0.36, test7, cov7_C)
Gaussian_w7 = copula_futures(4,5,train7, 10**9, 0.16, 0.36, test7, cov7_G)
#MDD:
print('MDD1')
gs = gridspec.GridSpec(3, 3)
md_calculator(10**3, trad_w1, test1)
md_calculator(10**3, Clayton_w1, test1)
md_calculator(10**3,Gaussian_w1, test1)
md_calculator(10**3,MDDw1, test1)
plt.show()



print('MDD2')
gs = gridspec.GridSpec(3, 3)
md_calculator(10**3, trad_w2, test2)
md_calculator(10**3, Clayton_w2, test2)
md_calculator(10**3,Gaussian_w2, test2)
md_calculator(10**3,MDDw2, test2)
plt.show()



print('MDD3')
gs = gridspec.GridSpec(3, 3)
md_calculator(10**3, trad_w3, test3)
md_calculator(10**3, Clayton_w3, test3)
md_calculator(10**3,Gaussian_w3, test3)
md_calculator(10**3,MDDw3, test3)
plt.show()

print('MDD4')
gs = gridspec.GridSpec(3, 3)
md_calculator(10**3, trad_w4, test4)
md_calculator(10**3, Clayton_w4, test4)
md_calculator(10**3,Gaussian_w4, test4)
md_calculator(10**3,MDDw4, test4)
plt.show()

print('MDD5')
gs = gridspec.GridSpec(3, 3)
md_calculator(10**3, trad_w5, test5)
md_calculator(10**3, Clayton_w5, test5)
md_calculator(10**3,Gaussian_w5, test5)
md_calculator(10**3,MDDw5, test5)
plt.show()

print('MDD6')
gs = gridspec.GridSpec(3, 3)
md_calculator(10**3, trad_w6, test6)
md_calculator(10**3, Clayton_w6, test6)
md_calculator(10**3,Gaussian_w6, test6)
md_calculator(10**3,MDDw6, test6)
plt.show()

print('MDD7')
gs = gridspec.GridSpec(3, 3)
md_calculator(10**3, trad_w7, test7)
md_calculator(10**3, Clayton_w7, test7)
md_calculator(10**3,Gaussian_w7, test7)
md_calculator(10**3,MDDw7, test7)
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
print("WHOLE PERIOD")
gs = gridspec.GridSpec(3, 3)
cum_ret(test1,trad_w1, test2, trad_w2, test3, trad_w3,test4,
        trad_w4, test5, trad_w5, test6, trad_w6, test7,trad_w7, 10**3)
cum_ret(test1,Clayton_w1, test2, Clayton_w2, test3, Clayton_w3,
        test4,Clayton_w4, test5, Clayton_w5, test6, Clayton_w6,test7, Clayton_w7,10**3)
cum_ret(test1,Gaussian_w1, test2, Gaussian_w2, test3, Gaussian_w3,
        test4,Gaussian_w4, test5, Gaussian_w5, test6, Gaussian_w6,test7, Gaussian_w7,10**3)
cum_ret(test1,MDDw1, test2, MDDw2, test3, MDDw3, test4, MDDw4, test5, MDDw5, test6, MDDw6, test7, MDDw7,10**3)
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


CDaR = [cum_ret1(test1,MDDw1),cum_ret1(test2,MDDw2),cum_ret1(test3,MDDw3)
                ,cum_ret1(test4,MDDw4),cum_ret1(test5,MDDw5),cum_ret1(test6,MDDw6),
                    cum_ret1(test7, MDDw7)]

Sharpe = {'trad' : Trad, 'Clay' : Clay, 'Gauss' : Gauss, 'CDaR': CDaR}

Sharpe = pd.DataFrame(Sharpe)

print(Sharpe)


