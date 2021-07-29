import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import random
from random import randint
from glob import glob
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
it = infoTest()

#sharpe calculator
def cum_ret1(df,weight):
    ret = np.dot(df,weight)
    sharpe = it.calculateSharpe(ret)
    return sharpe

#drawdown calculator for parameter searching function
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

#tradition method
#with 0.1 constraint:
def trad_sharpe_with_bounded(dataframe,list_num,bounded_list,upperbound,test):
    it=infoTest()
    cov = (dataframe.cov()).to_numpy()
    meanvec = (dataframe.mean()).to_numpy()
    meanvec = [i for i in meanvec]
    P = matrix(cov, tc='d')
    q = matrix(np.zeros(len(meanvec)), (len(meanvec), 1), tc='d')
    G=[]
    #wi >= 0

    for i in range(len(meanvec)):
        k=[0 for x in range(len(meanvec)-1)]
        k.insert(i, -1)
        G.append(k)

    #group 1,2 wi <= 0.06
    # for i in range(len(meanvec)-list_num):
    for i in range(list_num):
        k=[-bounded_list for x in range(len(meanvec)-1)]
        k.insert(i,1-bounded_list)
        G.append(k)
    #group 1,2 wi <= 0.1
    for i in range(len(meanvec)-list_num):
        k =[-upperbound for x in range(len(meanvec)-1)]
        k.insert(i+list_num, 1- upperbound)
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
#without 0.1 constraint:
def trad_sharpe_with_unbounded(dataframe,list_num,bounded_list,test):
    it=infoTest()
    cov = (dataframe.cov()).to_numpy()
    meanvec = (dataframe.mean()).to_numpy()
    meanvec = [i for i in meanvec]
    P = matrix(cov, tc='d')
    q = matrix(np.zeros(len(meanvec)), (len(meanvec), 1), tc='d')
    G=[]
    #wi >= 0
    for i in range(len(meanvec)):
        k=[0 for x in range(len(meanvec)-1)]
        k.insert(i, -1)
        G.append(k)
    #group 1,2 wi <= 0.06
    # for i in range(len(meanvec)-list_num):
    for i in range(list_num):
        k=[-bounded_list for x in range(len(meanvec)-1)]
        k.insert(i,1-bounded_list)
        G.append(k)
    #group 1,2 wi <= 0.1
    # for i in range(len(meanvec)-list_num):
    #     k =[-upperbound for x in range(len(meanvec)-1)]
    #     k.insert(i, 1- upperbound)
    #     G.append(k)
    G=matrix(np.array(G))
    # H = np.zeros(2*len(meanvec))
    H = np.zeros(len(meanvec)+ list_num)
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

#copula method:
#with 0.1 constraint:
def copula_sharpe_with_bounded(dataframe,list_num,bounded_list,upperbound,test, cova):
    it=infoTest()
    cov = (cova).to_numpy()
    meanvec = (dataframe.std()).to_numpy()
    meanvec = [i for i in meanvec]
    P = matrix(cov, tc='d')
    q = matrix(np.zeros(len(meanvec)), (len(meanvec), 1), tc='d')
    G=[]
    #wi >= 0
    for i in range(len(meanvec)):
        k=[0 for x in range(len(meanvec)-1)]
        k.insert(i, -1)
        G.append(k)

    #group 1,2 wi <= 0.06
    # for i in range(len(meanvec)-list_num):
    for i in range(list_num):
        k=[-bounded_list for x in range(len(meanvec)-1)]
        k.insert(i,1-bounded_list)
        G.append(k)
    #group 1,2 wi <= 0.1
    for i in range(len(meanvec)-list_num):
        k =[-upperbound for x in range(len(meanvec)-1)]
        k.insert(i+list_num, 1- upperbound)
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
#without 0.1 constraint:
def copula_sharpe_with_unbounded(dataframe,list_num,bounded_list,test, cova):
    it=infoTest()
    cov = (cova).to_numpy()
    meanvec = (dataframe.std()).to_numpy()
    meanvec = [i for i in meanvec]
    P = matrix(cov, tc='d')
    q = matrix(np.zeros(len(meanvec)), (len(meanvec), 1), tc='d')
    G=[]
    #wi >= 0
    for i in range(len(meanvec)):
        k=[0 for x in range(len(meanvec)-1)]
        k.insert(i, -1)
        G.append(k)
    #group 1,2 wi <= 0.06
    # for i in range(len(meanvec)-list_num):
    for i in range(list_num):
        k=[-bounded_list for x in range(len(meanvec)-1)]
        k.insert(i,1-bounded_list)
        G.append(k)
    #group 1,2 wi <= 0.1
    # for i in range(len(meanvec)-list_num):
    #     k =[-upperbound for x in range(len(meanvec)-1)]
    #     k.insert(i, 1- upperbound)
    #     G.append(k)
    G=matrix(np.array(G))
    # H = np.zeros(2*len(meanvec))
    H = np.zeros(len(meanvec)+ list_num)
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

#MDD method:
#with 0.1 constraint:
def MDD_constrained_fudamental_with_bounded(dataframe, list_num, bound_group, bound_alpha, alpha, v3):
    dataframe = dataframe.iloc[:,1:]
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


    #6th constraint: wi <= 0.06 and 0.1
    g61 = pd.DataFrame(np.diag(np.ones(N)))     #xk
    g62 = pd.DataFrame(np.zeros([N,J*2+1]))     #uk, zk, epsilon
    # print(g12)
    G6 = pd.concat([g61, g62], axis =1, ignore_index= True)
    h61  = [bound_group for x in range(list_num)]
    h62  = [bound_alpha for x in range(N-list_num)]
    h6= np.append(h61, h62)
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


    # #group 1 = 0.36
    # a21 = [1 for x in range(N-6)]
    # a22 = [0 for x in range(2*J + 1+ 6)]
    # A2 = np.append(a21,a22).reshape(1, -1)
    # A2 = matrix(A2, tc= 'd')
    # b2 = matrix([bound_group], (1,1), tc='d')
    # print(A2)
    # print(b2)
    #
    # A = pd.concat([pd.DataFrame(A1), pd.DataFrame(A2)], axis  =1).
    A = pd.DataFrame(A1).T
    A = matrix(np.array(A))
    # print(A)

    # b = matrix(np.array(pd.concat([pd.DataFrame(b1), pd.DataFrame(b2)], axis = 1).T))
    b = pd.DataFrame(b1).T
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
    solution = solution[:len(dataframe.columns)]
    return solution
#withous 0.1 constraint:
def MDD_constrained_fudamental_with_unbounded(dataframe, list_num, bound_group, alpha, v3):
    dataframe = dataframe.iloc[:,1:]
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


    #6th constraint: wi <= 0.06 in 2 groups
    g61 = pd.DataFrame(np.diag(np.ones(list_num)))     #xk
    g62 = pd.DataFrame(np.zeros([list_num,J*2+1]))     #uk, zk, epsilon
    # print(g12)
    G6 = pd.concat([g61, g62], axis =1, ignore_index= True)
    h6  = [bound_group for x in range(list_num)]
    # h62  = [bound_alpha for x in range(N-list_num)]
    # h6= np.append(h61, h62)
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


    # #group 1 = 0.36
    # a21 = [1 for x in range(N-6)]
    # a22 = [0 for x in range(2*J + 1+ 6)]
    # A2 = np.append(a21,a22).reshape(1, -1)
    # A2 = matrix(A2, tc= 'd')
    # b2 = matrix([bound_group], (1,1), tc='d')
    # print(A2)
    # print(b2)
    #
    # A = pd.concat([pd.DataFrame(A1), pd.DataFrame(A2)], axis  =1).
    A = pd.DataFrame(A1).T
    A = matrix(np.array(A))
    # print(A)

    # b = matrix(np.array(pd.concat([pd.DataFrame(b1), pd.DataFrame(b2)], axis = 1).T))
    b = pd.DataFrame(b1).T
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
    solution = solution[:len(dataframe.columns)]
    return solution

#parameter_checking
def MDD_constrained_fudamental_with_bounded_pc(dataframe, list_num, bound_group, bound_alpha, alpha, v3):
    dataframe = dataframe.iloc[:,:]
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
    print(G1.size)
    h1  = pd.DataFrame(np.zeros(N).reshape(-1,1))
    print(h1.size)


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


    #6th constraint: wi <= 0.06 and 0.1
    g61 = pd.DataFrame(np.diag(np.ones(N)))     #xk
    g62 = pd.DataFrame(np.zeros([N,J*2+1]))     #uk, zk, epsilon
    # print(g12)
    G6 = pd.concat([g61, g62], axis =1, ignore_index= True)
    h61  = [bound_group for x in range(list_num)]
    h62  = [bound_alpha for x in range(N-list_num)]
    h6= np.append(h61, h62)
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
    print(G.size)
    print(h.size)
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


    # #group 1 = 0.36
    # a21 = [1 for x in range(N-6)]
    # a22 = [0 for x in range(2*J + 1+ 6)]
    # A2 = np.append(a21,a22).reshape(1, -1)
    # A2 = matrix(A2, tc= 'd')
    # b2 = matrix([bound_group], (1,1), tc='d')
    # print(A2)
    # print(b2)
    #
    # A = pd.concat([pd.DataFrame(A1), pd.DataFrame(A2)], axis  =1).
    A = pd.DataFrame(A1).T
    A = matrix(np.array(A))
    # print(A)

    # b = matrix(np.array(pd.concat([pd.DataFrame(b1), pd.DataFrame(b2)], axis = 1).T))
    b = pd.DataFrame(b1).T
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

    # sol = solvers.lp(c, G, h, A, b,solver ='glpk')['x']
    # # print(sol['x'])
    # # print(solution[:8])
    # solution = [x for x in sol]
    # solution = solution[:len(dataframe.columns)

    sol = solvers.lp(c, G, h, A, b,solver ='glpk')['x']
    solution = [x for x in sol]

    # conditional drawdown at risk:
    cdar11 = [0 for x in range(N + J)]
    cdar12 = [1 / ((1 - alpha) * J) for y in range(J)]
    cdar = np.append(cdar11, cdar12)
    cdar = np.append(cdar, 1)
    cdar = cdar.reshape(1, -1)
    CDARX = np.dot(cdar, solution)

    #weight list
    weight = solution[:len(dataframe.columns)]
    # print(weight)
    mean_drawdown, max_drawdown = md_calculator1(10**10, weight, dataframe)
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

def MDD_constrained_fudamental_with_unbounded_pc(dataframe, list_num, bound_group, alpha, v3):
    dataframe = dataframe.iloc[:,1:]
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


    #6th constraint: wi <= 0.06 in 2 groups
    g61 = pd.DataFrame(np.diag(np.ones(list_num)))     #xk
    g62 = pd.DataFrame(np.zeros([list_num,J*2+1]))     #uk, zk, epsilon
    # print(g12)
    G6 = pd.concat([g61, g62], axis =1, ignore_index= True)
    h6  = [bound_group for x in range(list_num)]
    # h62  = [bound_alpha for x in range(N-list_num)]
    # h6= np.append(h61, h62)
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


    # #group 1 = 0.36
    # a21 = [1 for x in range(N-6)]
    # a22 = [0 for x in range(2*J + 1+ 6)]
    # A2 = np.append(a21,a22).reshape(1, -1)
    # A2 = matrix(A2, tc= 'd')
    # b2 = matrix([bound_group], (1,1), tc='d')
    # print(A2)
    # print(b2)
    #
    # A = pd.concat([pd.DataFrame(A1), pd.DataFrame(A2)], axis  =1).
    A = pd.DataFrame(A1).T
    A = matrix(np.array(A))
    # print(A)

    # b = matrix(np.array(pd.concat([pd.DataFrame(b1), pd.DataFrame(b2)], axis = 1).T))
    b = pd.DataFrame(b1).T
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

    sol = solvers.lp(c, G, h, A, b, solver='glpk')['x']
    solution = [x for x in sol]

    # conditional drawdown at risk:
    cdar11 = [0 for x in range(N + J)]
    cdar12 = [1 / ((1 - alpha) * J) for y in range(J)]
    cdar = np.append(cdar11, cdar12)
    cdar = np.append(cdar, 1)
    cdar = cdar.reshape(1, -1)
    CDARX = np.dot(cdar, solution)

    # weight list
    weight = solution[:len(dataframe.columns)]
    # print(weight)
    mean_drawdown, max_drawdown = md_calculator1(10 ** 10, weight, dataframe)
    # calculate sharpe
    dis_ret = np.dot(dataframe, weight)
    cum_ret = np.cumsum(dis_ret)
    sr = dis_ret.mean() / dis_ret.std() * np.sqrt(252)  # sharpe ratio
    # sharpe/cdar:
    new_ratio = sr / CDARX
    new_ratio1 = dis_ret.mean() / CDARX
    new_ratio2 = sr / mean_drawdown
    new_ratio3 = sr / max_drawdown
    new_ratio4 = dis_ret.mean() / mean_drawdown
    new_ratio5 = dis_ret.mean() / max_drawdown
    return


if __name__ == "__main__":

    #import source data
    # import os
    # os.chdir('D:/data-vietquant/futures-alpha-rolling')
    # os.chdir('/Users/tanvu10/Downloads/data-vietquant/futures-alpha-rolling')
    # os.chdir('D:/data-vietquant/10-futures-alpha-rolling')

    sample = 'OS'
    list_1 = glob('D:/data-vietquant/fundamental/group1/*.csv'.format(sample))
    # list_group.sort()
    list_2 = glob('D:/data-vietquant/fundamental/group2/*.csv'.format(sample))
    # list_nomal.sort()
    list_3 = glob('D:/data-vietquant/fundamental/group3/*.csv'.format(sample))
    print(list_3)
    fileList = list_1 + list_2 + list_3
    print(fileList)
    print(len(fileList))

    sample = "OS"
    m = pd.DataFrame()
    for file in fileList:
        # try:
        tempDf = pd.read_csv(file, parse_dates=[3], index_col=3)
        # print(tempDf.columns)
        # except:
        # print("ERROR : ", file)
        # print(file)
        # print(tempDf)
        # tempDf = pd.read_csv(file,parse_dates=[3],index_col=3)
        # tempDf = tempDf[tempDf.index >= datetime.utcnow().replace(year = 2020, month = 10, day =1, hour =0, minute=0, second=0)]

        tempPnl = tempDf[['value']]
        # print(tempPnl)
        tempPnl = tempPnl[tempPnl.index.dayofweek < 5]
        tempPnl['ret'] = (tempPnl - tempPnl.shift(1))
        tempPnl = tempPnl[['ret']].resample("1D").apply(lambda x: x.sum() if len(x) else np.nan).dropna(how="all")
        # print("strat " + file[9:], calculate_sharp(merge=tempPnl))
        if len(m) == 0:
            m = tempPnl
        else:
            m = pd.merge(m, tempPnl, how='inner', left_index=True, right_index=True)
        # count +=1
        # print(count)
    colList = []
    for i in fileList:
        # print(i)
        colList.append((i.split('/')[-1][:-4]).split("\\")[-1][:])
    m.columns = colList
    # print(m.columns)
    m=m/(10**10)
    # print(m)


    train_1 = m['2017-01-01':'2019-10-31']
    train_2 = m['2017-01-01':'2020-01-31']
    train_3 = m['2017-01-01':'2020-04-30']
    train_4 = m['2017-01-01':'2020-07-31']
    train_5 = m['2017-01-01':'2020-10-31']
    train_6 = m['2017-01-01':'2021-01-31']
    train_7 = m['2017-01-01':'2021-04-30']

    test_1 = m['2019-11-01':'2020-01-31']
    test_2 = m['2020-02-01':'2020-04-30']
    test_3 = m['2020-05-01':'2020-07-31']
    test_4 = m['2020-08-01':'2020-10-31']
    test_5 = m['2020-11-01':'2021-01-31']
    test_6 = m['2021-02-01':'2021-04-30']
    test_7 = m['2021-05-01':]
    # print(train_1)
    # import os
    # os.chdir('D:/data-vietquant/fundamental/fundamental_data')
    #
    # train_1.to_csv('train1.csv')
    # train_2.to_csv('train2.csv')
    # train_3.to_csv('train3.csv')
    # train_4.to_csv('train4.csv')
    # train_5.to_csv('train5.csv')
    # train_6.to_csv('train6.csv')
    # train_7.to_csv('train7.csv')
    # test_1.to_csv('test1.csv')
    # test_2.to_csv('test2.csv')
    # test_3.to_csv('test3.csv')
    # test_4.to_csv('test4.csv')
    # test_5.to_csv('test5.csv')
    # test_6.to_csv('test6.csv')
    # test_7.to_csv('test7.csv')

    #parameter searching
    # ranging = np.linspace(0.06, 0.08, 20)
    # alpha_range = np.linspace(0.85, 0.95, 10)
    # train_set = [train_7]
    # best_list = []
    # best_ilist =[]
    # for j in train_set:
    #     v_list =[]
    #     i_list =[]
    #     for k in alpha_range:
    #         for i in (ranging):
    #             try:
    #                 # dataframe, list_num, bound_group, bound_alpha, alpha, v3
    #                 new_ratio = MDD_constrained_fudamental_with_bounded_pc(j,6, 0.06, 0.1, k, i)
    #                 v_list.append(new_ratio)
    #                 i_list.append([k, i])
    #             except:
    #                 pass
    #     max_index = np.argmax(v_list)
    #     best_ilist.append(i_list[max_index])
    #     best_list.append(v_list[max_index])
    # print(best_list)
    # print(best_ilist)

    # w7= MDD_constrained_fudamental_with_bounded_pc(train_7, 6, 0.06, 0.1, 0.95, 0.08)
    # print(w7)


