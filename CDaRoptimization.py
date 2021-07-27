import os
os.chdir('D:/data-vietquant/futures-alpha-rolling')
# os.chdir('/Users/tanvu10/Downloads/data-vietquant/futures-alpha-rolling')

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


train1 = pd.read_csv('train1.csv',parse_dates=['datetime'])
train2 = pd.read_csv('train2.csv',parse_dates=['datetime'])
train3 = pd.read_csv('train3.csv',parse_dates=['datetime'])
train4 = pd.read_csv('train4.csv',parse_dates=['datetime'])
train5 = pd.read_csv('train5.csv',parse_dates=['datetime'])
train6 = pd.read_csv('train6.csv',parse_dates=['datetime'])
train7 = pd.read_csv('train7.csv',parse_dates=['datetime'])

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
it = infoTest()

#sharpe calculator
def cum_ret1(df,weight):
    ret = np.dot(df,weight)
    sharpe = it.calculateSharpe(ret)
    return sharpe

#drawdown calculator
def md_calculator1(booksize, weight, dataframe):
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
    # pd.Series(ddsr).plot()
    # plt.show()
    return np.mean(ddsr), np.max(ddsr)

#tradtional
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

#traditional copula
def trad_copula_sharpe(dataframe,upperbound,cov,rf,test):
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

#copula futures
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

#traditional futures
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

#max return with CDAR constraint:
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

#obj: MIN conditional drawdown at risk
def MinCVaR_futures(dataframe, bound_group, bound_alpha, alpha):
    dataframe = dataframe.iloc[:,1:]/10**3
    # alpha = 0.95
    # v3 = 60
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
    #
    # g71 = [0 for x in range(N+J)]
    # g72=  [1/((1-alpha)*J) for y in range(J)]
    # G7 = np.append(g71,g72)
    # G7 = np.append(G7,1)
    # G7 = G7.reshape(1,-1)
    # G7 = pd.DataFrame(G7)
    # # print(G7)
    # h7 = pd.Series(v3)



    #total inequality matrix
    G = pd.concat([G1, G2, G3, G4, G5, G6], axis = 0 , ignore_index=True)
    G = matrix(np.array(G))
    h = pd.concat([h1, h2,h3, h4, h5, h6], axis = 0,ignore_index=True)
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



    #objective function:
    # c1 = (dataframe.mean()).to_numpy()
    # c2 = [0 for x in range(2*J+1)]
    # c = np.append(c1, c2).reshape(1,-1)
    # c = -matrix(c.T)
    # print(c)

    #objective function: min CDAR
    c1 = [0 for x in range(N+J)]
    c2=  [1/((1-alpha)*J) for y in range(J)]
    c = np.append(c1,c2)
    c = np.append(c,1)
    c = c.reshape(1,-1)
    c = pd.DataFrame(c)
    c = matrix(np.array(c).T)
    # print(c)



    sol = solvers.lp(c, G, h, A, b,solver ='glpk')['x']
    # print(sol['x'])
    # print(solution[:8])
    solution = [x for x in sol]
    solution = solution[:9]
    return print(solution)

#max_sharpe with delta parameter
def MDD_delta_futures(dataframe, bound_group, bound_alpha, alpha, theta):
    # dataframe = train1
    # bound_group = 0.36
    # bound_alpha = 0.16
    # alpha = 0.95
    # theta = 0.1
    dataframe = dataframe.iloc[:, 1:] / 10**3
    # alpha = 0.95
    # v3 = v3
    # input to be cumulative return: y,u,z,epsilon
    cumulative = np.cumsum(dataframe, axis=0)
    cumulative = pd.DataFrame(cumulative)
    # print(cumulative)

    # inequality constraint
    # 1st constraint: wi >=0 <=> -wi <= 0
    J = dataframe.shape[0]  # number of row
    N = dataframe.shape[1]  # number of col
    g11 = pd.DataFrame(np.diag(np.ones(N)))  # xk
    g12 = pd.DataFrame(np.zeros([N, J * 2 + 1]))  # uk, zk, epsilon
    # print(g12)
    G1 = pd.concat([g11, g12], axis=1, ignore_index=True)
    G1 = -G1
    # print(matrix(np.array(G1)))
    h1 = pd.DataFrame(np.zeros(N).reshape(-1, 1))
    # print(h1)

    # 2nd constraint: -yk*x +uk - zk - epsilon <= 0
    g21 = -cumulative  # xk
    g22 = pd.DataFrame(np.diag(np.ones(J)))  # uk
    g23 = -pd.DataFrame(np.diag(np.ones(J)))  # zk
    g24 = - pd.DataFrame(np.ones([J, 1]))  # epsilon
    G2 = pd.concat([g21, g22, g23, g24], axis=1, ignore_index=True)
    # print(G2)
    h2 = pd.DataFrame(np.zeros(J).reshape(-1, 1))

    # 3rd constraint: zk >= 0 <=> -zk <= 0
    g31 = pd.DataFrame(np.zeros([J, N]))  # xk
    g32 = pd.DataFrame(np.zeros([J, J]))  # uk
    g33 = pd.DataFrame(np.diag(np.ones(J)))  # zk
    g34 = pd.DataFrame(np.zeros([J, 1]))  # epsilon
    G3 = pd.concat([g31, g32, -g33, g34], axis=1, ignore_index=True)
    # print(G3)

    h3 = pd.DataFrame(np.zeros(J).reshape(-1, 1))

    # 4th constraint: yk*x - uk <= 0
    g41 = cumulative  # xk
    g42 = -pd.DataFrame(np.diag(np.ones(J)))  # uk
    g43 = pd.DataFrame(np.zeros([J, J]))  # zk
    g44 = pd.DataFrame(np.zeros([J, 1]))  # epsilion
    G4 = pd.concat([g41, g42, g43, g44], axis=1, ignore_index=True)
    # print(G4)
    h4 = pd.DataFrame(np.zeros(J).reshape(-1, 1))

    # 5th constraint: uk-1  -- uk <= 0

    g51 = pd.DataFrame(np.zeros([J, N]))  # xk
    g52 = []
    a = [0 for x in range(J - 1)]
    a.insert(0, -1)
    g52.append(a)
    for i in range(J - 1):
        k = [0 for x in range(J - 2)]
        k.insert(i, -1)
        k.insert(i, 1)
        g52.append(k)
    g52 = pd.DataFrame(g52)  # uk
    # print(g52)
    g53 = pd.DataFrame(np.diag(np.zeros(J)))  # zk
    g54 = pd.DataFrame(np.zeros([J, 1]))  # epsilon
    G5 = pd.concat([g51, g52, g53, g54], axis=1, ignore_index=True)
    # print(G5)
    h5 = pd.DataFrame(np.zeros(J).reshape(-1, 1))

    # 6th constraint: wi <= 0.16
    g61 = pd.DataFrame(np.diag(np.ones(N)))  # xk
    g62 = pd.DataFrame(np.zeros([N, J * 2 + 1]))  # uk, zk, epsilon
    # print(g12)
    G6 = pd.concat([g61, g62], axis=1, ignore_index=True)
    h6 = [bound_alpha for x in range(N)]
    h6 = pd.DataFrame(np.array(h6).reshape(-1, 1))
    # print(G6)
    # print(h6)

    # 7th constraint: proportion v3
    #
    # g71 = [0 for x in range(N + J)]
    # g72 = [1 / ((1 - alpha) * J) for y in range(J)]
    # G7 = np.append(g71, g72)
    # G7 = np.append(G7, 1)
    # G7 = G7.reshape(1, -1)
    # G7 = pd.DataFrame(G7)
    # # print(G7)
    # h7 = pd.Series(v3)

    # total inequality matrix
    G = pd.concat([G1, G2, G3, G4, G5, G6], axis=0, ignore_index=True)
    G = matrix(np.array(G))
    h = pd.concat([h1, h2, h3, h4, h5, h6], axis=0, ignore_index=True)
    h = matrix(np.array(h))
    # print(G)
    # print(h)
    # breakpoint()

    # equality constraint:
    # total = 1
    a11 = [1 for x in range(N)]
    a12 = [0 for x in range(2 * J + 1)]
    A1 = np.append(a11, a12).reshape(1, -1)
    A1 = matrix(A1, tc='d')
    # print(A)
    b1 = matrix([1], (1, 1), tc='d')
    # print(b)

    # group 1 = 0.36
    a21 = [1 for x in range(N - 5)]
    a22 = [0 for x in range(2 * J + 1 + 5)]
    A2 = np.append(a21, a22).reshape(1, -1)
    A2 = matrix(A2, tc='d')
    b2 = matrix([bound_group], (1, 1), tc='d')
    # print(A2)
    # print(b2)
    #
    A = pd.concat([pd.DataFrame(A1), pd.DataFrame(A2)], axis=1).T
    A = matrix(np.array(A))
    # print(A)

    b = matrix(np.array(pd.concat([pd.DataFrame(b1), pd.DataFrame(b2)], axis=1).T))
    # print(b)

    # objective function: max return
    c11 = (cumulative.iloc[-1, :]).to_numpy()
    c12 = [0 for x in range(2 * J + 1)]
    c1 = np.append(c11, c12).reshape(1, -1)
    # c1 = -matrix(c1.T)
    # print(c1)


    c21 = [0 for x in range(N + J)]
    c22 = [1 / ((1 - alpha) * J) for y in range(J)]
    c2 = np.append(c21, c22)
    c2 = np.append(c2, 1)
    c2 = c2.reshape(1, -1)
    # print(c2)

    c = c1 - theta*c2
    # print(c)
    c = -pd.DataFrame(c).T
    # print(c)
    c = matrix(np.array(c))
    # print(c)

    # c1 = [0 for x in range(N+J)]
    # c2=  [1/((1-alpha)*J) for y in range(J)]
    # c = np.append(c1,c2)
    # c = np.append(c,1)
    # c = c.reshape(1,-1)
    # c = pd.DataFrame(c)
    # c = matrix(np.array(c).T)
    # print(c)
    #
    sol = solvers.lp(c, G, h, A, b, solver='glpk')['x']
    # print(sol['x'])
    # print(solution[:8])
    solution = [x for x in sol]


    #SHARPE RATIO solution
    mu11 = (dataframe.mean()).to_numpy()
    mu12 = [0 for x in range(2 * J + 1)]
    mu = np.append(mu11, mu12).reshape(1, -1)


    MuX = np.dot(mu,solution)
    Mx = np.dot(c2,solution)
    print(Mx)
    Rx = np.dot(c1,solution)
    print(Rx)
    sharpe = Rx/Mx
    sharpe1 = MuX/Mx
    print(sharpe)
    print(sharpe1)
    #objective = np.dot(np.array(c).T,np.array(solution))
    #print(objective)
    solution = solution[:9]
    print(solution)
    return sharpe, sharpe1

#modified max_sharpe
def max_sharpe(dataframe):
    #max sharpe:
    v3 = 0.1
    # dataframe = train1
    bound_group = 0.36
    bound_alpha= 0.16
    alpha = 0.95
    dataframe = dataframe.iloc[:,1:]/10**3
    # print(dataframe)
    # alpha = 0.95
    # v3 = v3*1000
    #input to be cumulative return: y,u,z,epsilon
    cumulative = np.cumsum(dataframe, axis = 0)
    cumulative= pd.DataFrame(cumulative)
    # print(cumulative)

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
    G6 = pd.concat([g61, g62], axis =1, ignore_index= True) - bound_alpha
    h6  = [0 for x in range(N)]
    h6 = pd.DataFrame(np.array(h6).reshape(-1,1))
    # print(G6)
    # print(h6)

    # 7th constraint: proportion v3
    g71 = [0 for x in range(N+J)]
    g72=  [1/((1-alpha)*J) for y in range(J)]
    G7 = np.append(g71,g72)
    G7 = np.append(G7,1)
    G7 = G7.reshape(1,-1)
    G7 = pd.DataFrame(G7) - v3
    print(G7)
    h7 = pd.Series(0)
    # print(h7)


    #8th constraint: group 1 = 0.36
    # (sum <= 0.36)
    g81 = [ 1 for x in range(N-5)]
    g82 = [0 for x in range(2*J + 1+ 5)]
    G81 = np.append(g81,g82).reshape(1, -1) - bound_group
    G81 = pd.DataFrame(G81)
    # print(G81)
    h81= pd.Series(0)

    #
    # # (sum >= 0.36)
    # g83 = [-1 for x in range(N-5)]
    # g84 = [0 for x in range(2*J + 1+ 5)]
    # G82 = np.append(g83,g84).reshape(1, -1) + bound_group
    # G82 = pd.DataFrame(G82)
    # # print(G82)
    # h82 = pd.Series(0)







    #total inequality matrix
    G = pd.concat([G1, G2, G3, G4, G5, G6,G7, G81], axis = 0 , ignore_index=True)
    G = matrix(np.array(G))
    h = pd.concat([h1, h2,h3, h4, h5, h6,h7, h81], axis = 0,ignore_index=True)
    h = matrix(np.array(h))
    # print(G)
    # print(h)
    # breakpoint()


    #equality constraint:
    # total = 1
    a11 = (dataframe.mean()).to_numpy()
    a12 = [0 for x in range(2*J+1)]
    A1 = np.append(a11,a12).reshape(1, -1)
    # A1 = matrix(A1, tc= 'd')
    # print(A)
    b1 = pd.Series(1)
    # b1 = matrix([1],(1,1), tc='d')
    # print(b)
    #
    # a21 = a11[:4]
    # a22 = [0 for x in range(5+2*J+1)]
    # A2 = np.append(a21, a22).reshape(1,-1)
    # b2 = pd.Series(0.36)



    #
    # A = pd.concat([pd.DataFrame(A1), pd.DataFrame(A2)], axis  =0)
    A = matrix(np.array(A1))
    print(A)

    # b = matrix(np.array(pd.concat([pd.DataFrame(b1), pd.DataFrame(b2)], axis = 1)).T)
    b = matrix([1], (1,1), tc='d')
    print(b)



    #objective function:
    # c1 = (dataframe.mean()).to_numpy()
    # c2 = [0 for x in range(2*J+1)]
    # c = np.append(c1, c2).reshape(1,-1)
    # c = -matrix(c.T)
    # print(c)


    #objective function: max_sharpe:
    p1 = pd.DataFrame((dataframe.cov()).to_numpy())       #xk
    p2 = pd.DataFrame(np.zeros([N,J]))        #uk
    p3 = pd.DataFrame(np.zeros([N,J])) #zk
    p4 = pd.DataFrame(np.zeros([N,1])) #epsilion
    p5 = pd.DataFrame(np.zeros([2*J+1,N+2*J+1]))
    P = pd.concat([p1, p2, p3, p4], axis = 1, ignore_index= True)
    P = pd.concat([P, p5], axis = 0, ignore_index=True)
    # print(P)
    P = matrix(np.array(P), tc ='d')
    # print(P)

    q = matrix(np.zeros(N+2*J+1), (N+2*J+1, 1), tc='d')
    # print(q)
    #objective function: min CDAR
    # c1 = [0 for x in range(N+J)]
    # c2=  [1/((1-alpha)*J) for y in range(J)]
    # c = np.append(c1,c2)
    # c = np.append(c,1)
    # c = c.reshape(1,-1)
    # c = pd.DataFrame(c)
    # c = matrix(np.array(c).T)
    # print(c)



    sol = solvers.qp(P, q, G, h, A, b)['x']
    # # print(sol['x'])
    # # print(solution[:8])
    solution = [x for x in sol]
    solution = solution[:9]
    print(solution)
    solution = [x/ (np.sum(solution)) for x in solution]
    print(solution)
    print(np.sum(solution[:4]))
    return solution

#parameter searching function:
def MDD_constrained_futures_run(dataframe, bound_group, bound_alpha, alpha, v3):
    dataframe = dataframe.iloc[:,1:]/10**3
    # alpha = 0.95
    v3 = v3
    #input to be cumulative return: y,u,z,epsilon
    cumulative = np.cumsum(dataframe, axis = 0)
    cumulative= pd.DataFrame(cumulative)
    # print(cumulative)

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
    solution = [x for x in sol]
    # conditional drawdown at risk:
    cdar11 = [0 for x in range(N + J)]
    cdar12 = [1 / ((1 - alpha) * J) for y in range(J)]
    cdar = np.append(cdar11, cdar12)
    cdar = np.append(cdar, 1)
    cdar = cdar.reshape(1, -1)
    CDARX = np.dot(cdar, solution)*(10**3)

    #weight list
    weight = solution[:9]
    mean_drawdown, max_drawdown = md_calculator(10**3, weight, dataframe)
    #calculate sharpe
    dis_ret = np.dot(dataframe, weight)
    cum_ret = np.cumsum(dis_ret)
    sr = dis_ret.mean() / dis_ret.std() * np.sqrt(252) #sharpe ratio
    #sharpe/cdar:
    new_ratio = sr/CDARX
    new_ratio1 = dis_ret.mean()/CDARX
    new_ratio2 = sr/mean_drawdown
    new_ratio3 = sr/max_drawdown
    new_ratio4 = dis_ret.mean()/mean_drawdown
    new_ratio5 = dis_ret.mean()/max_drawdown
    return sr

# MDD_constrained_futures(train1,0.36, 0.16, 0.95, 0.1)
# MDD_constrained_futures(train2,0.36, 0.16, 0.95, 0.1)
# MDD_constrained_futures(train3,0.36, 0.16, 0.95, 0.1)
# MDD_constrained_futures(train4,0.36, 0.16, 0.95, 0.1)
# MDD_constrained_futures(train5,0.36, 0.16, 0.95, 0.1)
# MDD_constrained_futures(train6,0.36, 0.16, 0.95, 0.1)
# MDD_constrained_futures(train7,0.36, 0.16, 0.95, 0.1)
# max_sharpe(train1)


#filter for delta:
# ranging = np.linspace(0.1, 200, 1000)
# list_i = []
# list_i1 = []
# for i in (ranging):
#     sharpe_i, sharpe_i1 = MDD_delta_futures(train7, 0.36, 0.16, 0.95,i)
#     list_i.append(sharpe_i)
#     list_i1.append(sharpe_i1)
#     print(i)
# print(np.max(list_i))
# print(np.max(list_i1))
# index = (np.argmax(list_i))
# index1 = np.argmax(list_i1)
# print(ranging[index])
# print(ranging[index1])


# MDD_delta_futures(train1, 0.36, 0.16, 0.95,48.14324324324324)
# MDD_delta_futures(train2, 0.36, 0.16, 0.95,62.155855855855854)
# MDD_delta_futures(train3, 0.36, 0.16, 0.95,95.18558558558559)
# MDD_delta_futures(train4, 0.36, 0.16, 0.95,59.153153153153156)
# MDD_delta_futures(train5, 0.36, 0.16, 0.95,54.343434343434346)
# MDD_delta_futures(train6, 0.36, 0.16, 0.95,57.52872872872873)
# MDD_delta_futures(train7, 0.36, 0.16, 0.95,49.324624624624626)

#filter for v3:

if __name__ == '__main__':
    ranging = np.linspace(0.06, 0.08, 20)
    alpha_range = np.linspace(0.85, 0.95, 10)
    train_set = [train1, train2, train3, train4, train5, train6, train7]
    best_list = []
    best_ilist =[]
    for j in train_set:
        v_list =[]
        i_list =[]
        for k in alpha_range:
            for i in (ranging):
                try:
                    new_ratio = MDD_constrained_futures_run(j,0.36, 0.16, k, i)
                    v_list.append(new_ratio)
                    i_list.append([k, i])
                except:
                    pass
        max_index = np.argmax(v_list)
        best_ilist.append(i_list[max_index])
        best_list.append(v_list[max_index])
    print(best_list)
    print(best_ilist)