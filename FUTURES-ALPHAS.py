import os
# os.chdir('D:/data-vietquant/futures-alpha-rolling')
os.chdir('D:/data-vietquant/10-futures-alpha-rolling')
# os.chdir('/Users/tanvu10/Downloads/data-vietquant/futures-alpha-rolling')
# os.chdir('/Users/tanvu10/Downloads/data-vietquant/10-futures-alpha-rolling')



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

#import function from CDAR file
from CDaR_optimization import MDD_constrained_futures
from CDaR_optimization import copula_futures
from CDaR_optimization import trad_futures
from CDaR_optimization import infoTest
from CDaR_optimization import cum_ret1

it  = infoTest()


#MDD drawing
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
            # print(mdd)
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
#
# #train:
# train1 = pd.read_csv('train1.csv',parse_dates=['datetime'])
# train2 = pd.read_csv('train2.csv',parse_dates=['datetime'])
# train3 = pd.read_csv('train3.csv',parse_dates=['datetime'])
# train4 = pd.read_csv('train4.csv',parse_dates=['datetime'])
# train5 = pd.read_csv('train5.csv',parse_dates=['datetime'])
# train6 = pd.read_csv('train6.csv',parse_dates=['datetime'])
# train7 = pd.read_csv('train7.csv',parse_dates=['datetime'])
#
# #test:
#
# test1 = pd.read_csv('test1.csv',parse_dates=['datetime'],index_col=['datetime'])
# test2 = pd.read_csv('test2.csv',parse_dates=['datetime'],index_col=['datetime'])
# test3 = pd.read_csv('test3.csv',parse_dates=['datetime'],index_col=['datetime'])
# test4 = pd.read_csv('test4.csv',parse_dates=['datetime'],index_col=['datetime'])
# test5 = pd.read_csv('test5.csv',parse_dates=['datetime'],index_col=['datetime'])
# test6 = pd.read_csv('test6.csv',parse_dates=['datetime'],index_col=['datetime'])
# test7 = pd.read_csv('test7.csv',parse_dates=['datetime'],index_col=['datetime'])


#train:
train1 = pd.read_csv('future_train1.csv',parse_dates=['datetime'])
train2 = pd.read_csv('future_train2.csv',parse_dates=['datetime'])
train3 = pd.read_csv('future_train3.csv',parse_dates=['datetime'])
train4 = pd.read_csv('future_train4.csv',parse_dates=['datetime'])
train5 = pd.read_csv('future_train5.csv',parse_dates=['datetime'])
train6 = pd.read_csv('future_train6.csv',parse_dates=['datetime'])
train7 = pd.read_csv('future_train7.csv',parse_dates=['datetime'])

#test:

test1 = pd.read_csv('future_test1.csv',parse_dates=['datetime'],index_col=['datetime'])
test2 = pd.read_csv('future_test2.csv',parse_dates=['datetime'],index_col=['datetime'])
test3 = pd.read_csv('future_test3.csv',parse_dates=['datetime'],index_col=['datetime'])
test4 = pd.read_csv('future_test4.csv',parse_dates=['datetime'],index_col=['datetime'])
test5 = pd.read_csv('future_test5.csv',parse_dates=['datetime'],index_col=['datetime'])
test6 = pd.read_csv('future_test6.csv',parse_dates=['datetime'],index_col=['datetime'])
test7 = pd.read_csv('future_test7.csv',parse_dates=['datetime'],index_col=['datetime'])




# #Clayton covariance matrix:
# cov1_C = pd.read_csv('train1_covC.csv', header= None,skiprows = 1 )
# cov1_C = cov1_C.iloc[:,1:]
#
# cov2_C = pd.read_csv('train2_covC.csv', header= None,skiprows = 1 )
# cov2_C = cov2_C.iloc[:,1:]
#
# cov3_C = pd.read_csv('train3_covC.csv', header= None,skiprows = 1 )
# cov3_C = cov3_C.iloc[:,1:]
#
# cov4_C = pd.read_csv('train4_covC.csv', header= None,skiprows = 1 )
# cov4_C = cov4_C.iloc[:,1:]
#
# cov5_C = pd.read_csv('train5_covC.csv', header= None,skiprows = 1 )
# cov5_C = cov5_C.iloc[:,1:]
#
# cov6_C = pd.read_csv('train6_covC.csv', header= None,skiprows = 1 )
# cov6_C = cov6_C.iloc[:,1:]
#
# cov7_C = pd.read_csv('train7_covC.csv', header= None,skiprows = 1 )
# cov7_C = cov7_C.iloc[:,1:]
#
#
# #Gaussian covariance matrix
# cov1_G = pd.read_csv('train1_covG.csv', header= None,skiprows = 1 )
# cov1_G = cov1_G.iloc[:,1:]
#
# cov2_G = pd.read_csv('train2_covG.csv', header= None,skiprows = 1 )
# cov2_G = cov2_G.iloc[:,1:]
#
# cov3_G = pd.read_csv('train3_covG.csv', header= None,skiprows = 1 )
# cov3_G = cov3_G.iloc[:,1:]
#
# cov4_G = pd.read_csv('train4_covG.csv', header= None,skiprows = 1 )
# cov4_G = cov4_G.iloc[:,1:]
#
# cov5_G = pd.read_csv('train5_covG.csv', header= None,skiprows = 1 )
# cov5_G = cov5_G.iloc[:,1:]
#
# cov6_G = pd.read_csv('train6_covG.csv', header= None,skiprows = 1 )
# cov6_G = cov6_G.iloc[:,1:]
#
# cov7_G = pd.read_csv('train7_covG.csv', header= None,skiprows = 1 )
# cov7_G = cov7_G.iloc[:,1:]



# #max SHARPE: decent return - good dd - count = 9
# MDDw1 = MDD_constrained_futures(train1,0.36, 0.16, 0.8833333333333333, 0.06421052631578947)
# MDDw2 = MDD_constrained_futures(train2,0.36, 0.16, 0.9277777777777777, 0.06)
# MDDw3 = MDD_constrained_futures(train3,0.36, 0.16, 0.9277777777777777, 0.061052631578947365)
# MDDw4 = MDD_constrained_futures(train4,0.36, 0.16, 0.8944444444444444, 0.06210526315789473)
# MDDw5 = MDD_constrained_futures(train5,0.36, 0.16, 0.95, 0.07157894736842105)
# MDDw6 = MDD_constrained_futures(train6,0.36, 0.16, 0.9055555555555556, 0.06842105263157895)
# MDDw7 = MDD_constrained_futures(train7,0.36, 0.16, 0.8944444444444444, 0.06736842105263158)

# print(MDDw1)
# print(MDDw2)
# print(MDDw3)
# print(MDDw4)
# print(MDDw5)
# print(MDDw6)
# print(MDDw7)


# #max SHARPE/CDAR: decent return - good dd - count = 9
# MDDw1 = MDD_constrained_futures(train1,0.36, 0.16, 0.85, 0.06)
# MDDw2 = MDD_constrained_futures(train2,0.36, 0.16, 0.9277777777777777, 0.06)
# MDDw3 = MDD_constrained_futures(train3,0.36, 0.16, 0.9166666666666666, 0.06)
# MDDw4 = MDD_constrained_futures(train4,0.36, 0.16, 0.8833333333333333, 0.06)
# MDDw5 = MDD_constrained_futures(train5,0.36, 0.16, 0.9055555555555556, 0.06)
# MDDw6 = MDD_constrained_futures(train6,0.36, 0.16, 0.85, 0.06)
# MDDw7 = MDD_constrained_futures(train7,0.36, 0.16, 0.861111111111111, 0.06)

#:MANUAL
h = 0.95
v = 0.06
# MDDw1 = MDD_constrained_futures(train1,0.36, 0.16, 0.95, 0.06)
# MDDw2 = MDD_constrained_futures(train2,0.36, 0.16, 0.95, 0.06)
# MDDw3 = MDD_constrained_futures(train3,0.36, 0.16, 0.95, 0.06)
# MDDw4 = MDD_constrained_futures(train4,0.36, 0.16, 0.95, 0.06)
# MDDw5 = MDD_constrained_futures(train5,0.36, 0.16, 0.95, 0.06)
# MDDw6 = MDD_constrained_futures(train6,0.36, 0.16, 0.95, 0.06)
# MDDw7 = MDD_constrained_futures(train7,0.36, 0.16, 0.95, 0.06)

#max sharpe/meandd: (good sharpe - good drawdown - not good return)
MDDw1 = MDD_constrained_futures(train1,0.36, 0.16, 0.8833333333333333, 0.06421052631578947)
MDDw2 = MDD_constrained_futures(train2,0.36, 0.16, 0.8833333333333333, 0.06)
MDDw3 = MDD_constrained_futures(train3,0.36, 0.16, 0.9388888888888889, 0.06210526315789473)
MDDw4 = MDD_constrained_futures(train4,0.36, 0.16, 0.8833333333333333, 0.06)
MDDw5 = MDD_constrained_futures(train5,0.36, 0.16, 0.9166666666666666, 0.061052631578947365)
MDDw6 = MDD_constrained_futures(train6,0.36, 0.16, 0.9277777777777777, 0.06736842105263158)
MDDw7 = MDD_constrained_futures(train7,0.36, 0.16, 0.9388888888888889, 0.06736842105263158)


#max sharpe/maxdd: (good sharpe - good drawdown - not good return)
# MDDw1 = MDD_constrained_futures(train1,0.36, 0.16, 0.95, 0.06210526315789473)
# MDDw2 = MDD_constrained_futures(train2,0.36, 0.16, 0.95, 0.06210526315789473)
# MDDw3 = MDD_constrained_futures(train3,0.36, 0.16, 0.95, 0.06210526315789473)
# MDDw4 = MDD_constrained_futures(train4,0.36, 0.16, 0.95, 0.06315789473684211)
# MDDw5 = MDD_constrained_futures(train5,0.36, 0.16, 0.9388888888888889, 0.06210526315789473)
# MDDw6 = MDD_constrained_futures(train6,0.36, 0.16, 0.9055555555555556, 0.06315789473684211)
# MDDw7 = MDD_constrained_futures(train7,0.36, 0.16, 0.8944444444444444, 0.061052631578947365)

#max return/meamdd: (count = 8)
# MDDw1 = MDD_constrained_futures(train1,0.36, 0.16, 0.95, 0.06210526315789473)
# MDDw2 = MDD_constrained_futures(train2,0.36, 0.16, 0.95, 0.06210526315789473)
# MDDw3 = MDD_constrained_futures(train3,0.36, 0.16, 0.95, 0.06)
# MDDw4 = MDD_constrained_futures(train4,0.36, 0.16, 0.95, 0.06)
# MDDw5 = MDD_constrained_futures(train5,0.36, 0.16, 0.95, 0.06)
# MDDw6 = MDD_constrained_futures(train6,0.36, 0.16, 0.9388888888888889, 0.06)
# MDDw7 = MDD_constrained_futures(train7,0.36, 0.16, 0.9166666666666666, 0.06)

#max return/maxdd (count = 7, mdd good, return good)
# MDDw1 = MDD_constrained_futures(train1,0.36, 0.16, 0.9277777777777777, 0.06)
# MDDw2 = MDD_constrained_futures(train2,0.36, 0.16, 0.9277777777777777, 0.06)
# MDDw3 = MDD_constrained_futures(train3,0.36, 0.16, 0.95, 0.06210526315789473)
# MDDw4 = MDD_constrained_futures(train4,0.36, 0.16, 0.95, 0.06315789473684211)
# MDDw5 = MDD_constrained_futures(train5,0.36, 0.16, 0.9388888888888889, 0.06210526315789473)
# MDDw6 = MDD_constrained_futures(train6,0.36, 0.16, 0.8722222222222222, 0.06)
# MDDw7 = MDD_constrained_futures(train7,0.36, 0.16, 0.8944444444444444, 0.061052631578947365)


trad_w1 = trad_futures(4,6,train1, 10**9,0.16,0.36,test1)
# Clayton_w1 = copula_futures(4,5, train1, 10**9, 0.16, 0.36, test1, cov1_C)
# Gaussian_w1 = copula_futures(4,5,train1, 10**9, 0.16, 0.36, test1, cov1_G)

trad_w2 = trad_futures(4,6,train2, 10**9,0.16,0.36,test2)
# Clayton_w2 = copula_futures(4,5, train2, 10**9, 0.16, 0.36, test2, cov2_C)
# Gaussian_w2 = copula_futures(4,5,train2, 10**9, 0.16, 0.36, test2, cov2_G)

trad_w3 = trad_futures(4,6,train3, 10**9,0.16,0.36,test3)
# Clayton_w3 = copula_futures(4,5, train3, 10**9, 0.16, 0.36, test3, cov3_C)
# Gaussian_w3 = copula_futures(4,5,train3, 10**9, 0.16, 0.36, test3, cov3_G)

trad_w4 = trad_futures(4,6,train4, 10**9,0.16,0.36,test4)
# Clayton_w4 = copula_futures(4,5, train4, 10**9, 0.16, 0.36, test4, cov4_C)
# Gaussian_w4 = copula_futures(4,5,train4, 10**9, 0.16, 0.36, test4, cov4_G)

trad_w5 = trad_futures(4,6,train5, 10**9,0.16,0.36,test5)
# Clayton_w5 = copula_futures(4,5, train5, 10**9, 0.16, 0.36, test5, cov5_C)
# Gaussian_w5 = copula_futures(4,5,train5, 10**9, 0.16, 0.36, test5, cov5_G)

trad_w6 = trad_futures(4,6,train6, 10**9,0.16,0.36,test6)
# Clayton_w6 = copula_futures(4,5, train6, 10**9, 0.16, 0.36, test6, cov6_C)
# Gaussian_w6 = copula_futures(4,5,train6, 10**9, 0.16, 0.36, test6, cov6_G)

trad_w7 = trad_futures(4,6,train7, 10**9,0.16,0.36,test7)
# Clayton_w7 = copula_futures(4,5, train7, 10**9, 0.16, 0.36, test7, cov7_C)
# Gaussian_w7 = copula_futures(4,5,train7, 10**9, 0.16, 0.36, test7, cov7_G)

#MDD:
print('MDD1')
gs = gridspec.GridSpec(3, 3)
md_calculator(10**3, trad_w1, test1)
# md_calculator(10**3, Clayton_w1, test1)
# md_calculator(10**3,Gaussian_w1, test1)
md_calculator(10**3,MDDw1, test1)
plt.show()



print('MDD2')
gs = gridspec.GridSpec(3, 3)
md_calculator(10**3, trad_w2, test2)
# md_calculator(10**3, Clayton_w2, test2)
# md_calculator(10**3,Gaussian_w2, test2)
md_calculator(10**3,MDDw2, test2)
plt.show()



print('MDD3')
gs = gridspec.GridSpec(3, 3)
md_calculator(10**3, trad_w3, test3)
# md_calculator(10**3, Clayton_w3, test3)
# md_calculator(10**3,Gaussian_w3, test3)
md_calculator(10**3,MDDw3, test3)
plt.show()

print('MDD4')
gs = gridspec.GridSpec(3, 3)
md_calculator(10**3, trad_w4, test4)
# md_calculator(10**3, Clayton_w4, test4)
# md_calculator(10**3,Gaussian_w4, test4)
md_calculator(10**3,MDDw4, test4)
plt.show()

print('MDD5')
gs = gridspec.GridSpec(3, 3)
md_calculator(10**3, trad_w5, test5)
# md_calculator(10**3, Clayton_w5, test5)
# md_calculator(10**3,Gaussian_w5, test5)
md_calculator(10**3,MDDw5, test5)
plt.show()

print('MDD6')
gs = gridspec.GridSpec(3, 3)
md_calculator(10**3, trad_w6, test6)
# md_calculator(10**3, Clayton_w6, test6)
# md_calculator(10**3,Gaussian_w6, test6)
md_calculator(10**3,MDDw6, test6)
plt.show()

print('MDD7')
gs = gridspec.GridSpec(3, 3)
md_calculator(10**3, trad_w7, test7)
# md_calculator(10**3, Clayton_w7, test7)
# md_calculator(10**3,Gaussian_w7, test7)
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
print("WHOLE PERIOD")
gs = gridspec.GridSpec(3, 3)
cum_ret(test1,trad_w1, test2, trad_w2, test3, trad_w3,test4,
        trad_w4, test5, trad_w5, test6, trad_w6, test7,trad_w7, 10**3)
# cum_ret(test1,Clayton_w1, test2, Clayton_w2, test3, Clayton_w3,
#         test4,Clayton_w4, test5, Clayton_w5, test6, Clayton_w6,test7, Clayton_w7,10**3)
# cum_ret(test1,Gaussian_w1, test2, Gaussian_w2, test3, Gaussian_w3,
#         test4,Gaussian_w4, test5, Gaussian_w5, test6, Gaussian_w6,test7, Gaussian_w7,10**3)
cum_ret(test1,MDDw1, test2, MDDw2, test3, MDDw3, test4, MDDw4, test5, MDDw5, test6, MDDw6, test7, MDDw7,10**3)
plt.show()
#

Trad = [cum_ret1(test1,trad_w1),cum_ret1(test2,trad_w2),
        cum_ret1(test3,trad_w3),cum_ret1(test4,trad_w4),
        cum_ret1(test5,trad_w5),cum_ret1(test6,trad_w6),cum_ret1(test7,trad_w7)]
#
# Clay = [cum_ret1(test1,Clayton_w1),cum_ret1(test2,Clayton_w2),cum_ret1(test3,Clayton_w3),
#         cum_ret1(test4,Clayton_w4),cum_ret1(test5,Clayton_w5),cum_ret1(test6,Clayton_w6),
#         cum_ret1(test7,Clayton_w7)]
#
#
# Gauss = [cum_ret1(test1,Gaussian_w1),cum_ret1(test2,Gaussian_w2),cum_ret1(test3,Gaussian_w3)
#                 ,cum_ret1(test4,Gaussian_w4),cum_ret1(test5,Gaussian_w5),cum_ret1(test6,Gaussian_w6),
#                                                                           cum_ret1(test7,Gaussian_w7)]


CDaR = [cum_ret1(test1,MDDw1),cum_ret1(test2,MDDw2),cum_ret1(test3,MDDw3)
                ,cum_ret1(test4,MDDw4),cum_ret1(test5,MDDw5),cum_ret1(test6,MDDw6),
                    cum_ret1(test7, MDDw7)]

# Sharpe = {'trad' : Trad, 'Clay' : Clay, 'Gauss' : Gauss, 'CDaR': CDaR}
Sharpe = {'trad' : Trad, 'CDaR': CDaR}

Sharpe = pd.DataFrame(Sharpe)

print(Sharpe)


#PERIOD 7
port7 = {'trad':trad_w7,'MDD' : MDDw7}
port7 = pd.DataFrame(port7)
port7.index = train1.columns[1:]
print(port7)

# port7.to_csv('Port_test_7_return_over_mdd.csv')

# plt.plot(pd.Series(np.cumsum(test7.iloc[:, 4])))
# plt.plot(pd.Series(np.cumsum(test7.iloc[:, 9])))
# plt.show()