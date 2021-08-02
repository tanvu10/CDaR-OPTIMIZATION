import os
# os.chdir('D:/data-vietquant/futures-alpha-rolling')
# os.chdir('D:/data-vietquant/10-futures-alpha-rolling')
# os.chdir('/Users/tanvu10/Downloads/data-vietquant/futures-alpha-rolling')
# os.chdir('/Users/tanvu10/Downloads/data-vietquant/10-futures-alpha-rolling')
# os.chdir('D:/data-vietquant/fundamental/fundamental_data)



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

from CDaR_optimization_funda import trad_sharpe_with_bounded
from CDaR_optimization_funda import trad_sharpe_with_unbounded
from CDaR_optimization_funda import copula_sharpe_with_bounded
from CDaR_optimization_funda import copula_sharpe_with_unbounded
from CDaR_optimization_funda import infoTest
from CDaR_optimization_funda import cum_ret1
it = infoTest()


def md_calculator(booksize, weight, dataframe):
    dis_ret = np.dot(dataframe, weight)
    cum_ret = np.cumsum(dis_ret)*(booksize)
    mdd = 0
    X = cum_ret + 10**10
    peak = X[0]
    ddsr = []
    #calculate dd
    for x in X:
        if x > peak:
            peak = x
        dd = (peak - x) / 10**10
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

def cum_ret(df1, weight1, df2, weight2, df3, weight3,df4, weight4, df5, weight5, df6, weight6,df7, weight7, booksize):
    return_1 = np.dot(df1, weight1).reshape(-1,1)
    return_2 = np.dot(df2, weight2).reshape(-1,1)
    return_3 = np.dot(df3, weight3).reshape(-1,1)
    return_4 = np.dot(df4, weight4).reshape(-1,1)
    return_5 = np.dot(df5, weight5).reshape(-1,1)
    return_6 = np.dot(df6, weight6).reshape(-1,1)
    return_7 = np.dot(df7, weight7).reshape(-1,1)
    dis_ret = np.vstack((return_1, return_2,return_3,return_4,return_5,return_6,return_7))
    cum_ret = np.cumsum(dis_ret)*(booksize)
    sharpe = it.calculateSharpe(dis_ret)

    #mdd plot
    mdd = 0
    X = cum_ret + 10**10
    peak = X[0]
    ddsr = []
    count =0
    # calculate dd
    for x in X:
        if x > peak:
            peak = x
        dd = (peak - x) / 10**10
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



#IMPORT
os.chdir('D:/data-vietquant/fundamental/fundamental_data')
train1 = pd.read_csv('train1.csv',parse_dates=['datetime'],index_col=['datetime'])
train2 = pd.read_csv('train2.csv',parse_dates=['datetime'],index_col=['datetime'])
train3 = pd.read_csv('train3.csv',parse_dates=['datetime'],index_col=['datetime'])
train4 = pd.read_csv('train4.csv',parse_dates=['datetime'],index_col=['datetime'])
train5 = pd.read_csv('train5.csv',parse_dates=['datetime'],index_col=['datetime'])
train6 = pd.read_csv('train6.csv',parse_dates=['datetime'],index_col=['datetime'])
train7 = pd.read_csv('train7.csv',parse_dates=['datetime'],index_col=['datetime'])
train8 = pd.read_csv('train8.csv',parse_dates=['datetime'],index_col=['datetime'])
print(train8)
#test:

test1 = pd.read_csv('test1.csv',parse_dates=['datetime'],index_col=['datetime'])
test2 = pd.read_csv('test2.csv',parse_dates=['datetime'],index_col=['datetime'])
test3 = pd.read_csv('test3.csv',parse_dates=['datetime'],index_col=['datetime'])
test4 = pd.read_csv('test4.csv',parse_dates=['datetime'],index_col=['datetime'])
test5 = pd.read_csv('test5.csv',parse_dates=['datetime'],index_col=['datetime'])
test6 = pd.read_csv('test6.csv',parse_dates=['datetime'],index_col=['datetime'])
test7 = pd.read_csv('test7.csv',parse_dates=['datetime'],index_col=['datetime'])


#import simulation data
os.chdir('D:/data-vietquant/fundamental/fundamental_simulation')
simu_G1 = pd.read_csv('X1_G.csv')
simu_G1 = simu_G1.iloc[:,1:]

simu_G2 = pd.read_csv('X2_G.csv')
simu_G2 = simu_G2.iloc[:,1:]

simu_G3 = pd.read_csv('X3_G.csv')
simu_G3 = simu_G3.iloc[:,1:]

simu_G4 = pd.read_csv('X4_G.csv')
simu_G4 = simu_G4.iloc[:,1:]

simu_G5 = pd.read_csv('X5_G.csv')
simu_G5 = simu_G5.iloc[:,1:]

simu_G6 = pd.read_csv('X6_G.csv')
simu_G6 = simu_G6.iloc[:,1:]

simu_G7 = pd.read_csv('X7_G.csv')
simu_G7 = simu_G7.iloc[:,1:]
# print(simu_G7)

simu_G8 = pd.read_csv('X8_G.csv')
simu_G8 = simu_G8.iloc[:,1:]


os.chdir('D:/data-vietquant/fundamental/fundamental_cov')
cov1_G = pd.read_csv('cov1_G.csv')
cov1_G = cov1_G.iloc[:,1:]

cov2_G = pd.read_csv('cov2_G.csv')
cov2_G = cov2_G.iloc[:,1:]

cov3_G = pd.read_csv('cov3_G.csv')
cov3_G = cov3_G.iloc[:,1:]

cov4_G = pd.read_csv('cov4_G.csv')
cov4_G = cov4_G.iloc[:,1:]

cov5_G = pd.read_csv('cov5_G.csv')
cov5_G = cov5_G.iloc[:,1:]

cov6_G = pd.read_csv('cov6_G.csv')
cov6_G = cov6_G.iloc[:,1:]

cov7_G = pd.read_csv('cov7_G.csv')
cov7_G = cov7_G.iloc[:,1:]
# print(cov7_G)

cov8_G = pd.read_csv('cov8_G.csv')
cov8_G = cov8_G.iloc[:,1:]

#traditional
tradw1_bounded = trad_sharpe_with_bounded(train1, 7,0.06, 0.1, test1)
tradw1_unbounded = trad_sharpe_with_unbounded(train1, 7,0.06, test1)

tradw2_bounded = trad_sharpe_with_bounded(train2, 7,0.06, 0.1, test2)
tradw2_unbounded = trad_sharpe_with_unbounded(train2, 7,0.06, test2)

tradw3_bounded = trad_sharpe_with_bounded(train3, 7,0.06, 0.1, test3)
tradw3_unbounded = trad_sharpe_with_unbounded(train3, 7,0.06, test3)

tradw4_bounded = trad_sharpe_with_bounded(train4, 7,0.06, 0.1, test3)
tradw4_unbounded = trad_sharpe_with_unbounded(train4,7, 0.06, test4)

tradw5_bounded = trad_sharpe_with_bounded(train5, 7,0.06, 0.1, test5)
tradw5_unbounded = trad_sharpe_with_unbounded(train5, 7,0.06, test5)

tradw6_bounded = trad_sharpe_with_bounded(train6, 7,0.06, 0.1, test6)
tradw6_unbounded = trad_sharpe_with_unbounded(train6, 7,0.06, test6)
#
tradw7_bounded = trad_sharpe_with_bounded(train7, 7,0.06, 0.1, test7)
tradw7_unbounded = trad_sharpe_with_unbounded(train7,7, 0.06, test7)


tradw8_bounded = trad_sharpe_with_bounded(train8, 7,0.06, 0.1, train8)
tradw8_unbounded = trad_sharpe_with_unbounded(train8,7, 0.06, train8)
print(tradw7_bounded)
print(tradw7_unbounded)
print(tradw8_bounded)
print(tradw8_unbounded)


#copula
Gw1_bounded = copula_sharpe_with_bounded(train1,7, 0.06, 0.1,test1, cov1_G)
# print(Gw1_bounded)
Gw1_unbounded = copula_sharpe_with_unbounded(train1,7, 0.06,test1, cov1_G)
# print((Gw1_unbounded))
# Gw1_bounded = trad_sharpe_with_bounded(simu_G1, 7,0.06, 0.1, test1)
# Gw1_unbounded = trad_sharpe_with_unbounded(simu_G1, 7,0.06, test1)
# #)

Gw2_bounded = copula_sharpe_with_bounded(simu_G2,7, 0.06, 0.1,test2, cov2_G)
Gw2_unbounded = copula_sharpe_with_unbounded(simu_G2,7, 0.06,test2, cov2_G)
# Gw2_bounded = trad_sharpe_with_bounded(simu_G2, 7,0.06, 0.1, test2)
# Gw2_unbounded = trad_sharpe_with_unbounded(simu_G2, 7,0.06, test2)

#
Gw3_bounded = copula_sharpe_with_bounded(simu_G3,7, 0.06, 0.1,test3, cov3_G)
Gw3_unbounded = copula_sharpe_with_unbounded(simu_G3,7, 0.06,test3, cov3_G)

#
Gw4_bounded = copula_sharpe_with_bounded(simu_G4,7, 0.06, 0.1,test4, cov4_G)
Gw4_unbounded = copula_sharpe_with_unbounded(simu_G4,7, 0.06,test4, cov4_G)
#
Gw5_bounded = copula_sharpe_with_bounded(simu_G5,7, 0.06, 0.1,test5, cov5_G)
Gw5_unbounded = copula_sharpe_with_unbounded(simu_G5,7, 0.06,test5, cov5_G)
#
Gw6_bounded = copula_sharpe_with_bounded(simu_G6,7, 0.06, 0.1,test6, cov6_G)
Gw6_unbounded = copula_sharpe_with_unbounded(simu_G6,7, 0.06,test6, cov6_G)
#
Gw7_bounded = copula_sharpe_with_bounded(simu_G7,7, 0.06, 0.1,test7, cov7_G)
Gw7_unbounded = copula_sharpe_with_unbounded(simu_G7,7, 0.06,test7, cov7_G)


Gw8_bounded = copula_sharpe_with_bounded(simu_G8,7, 0.06, 0.1,train8, cov8_G)
Gw8_unbounded = copula_sharpe_with_unbounded(simu_G8,7, 0.06,train8, cov8_G)
print(Gw7_bounded)
print(Gw7_unbounded)

print(np.sum(Gw8_bounded))

print(np.sum(Gw8_unbounded))



print('MDD1')
gs = gridspec.GridSpec(3, 3)
md_calculator(10**9, tradw1_bounded, test1)
md_calculator(10**9, tradw1_unbounded, test1)
md_calculator(10**9,Gw1_bounded, test1)
md_calculator(10**9,Gw1_unbounded, test1)
plt.show()

print('MDD2')
gs = gridspec.GridSpec(3, 3)
md_calculator(10**9, tradw2_bounded, test2)
md_calculator(10**9, tradw2_unbounded, test2)
md_calculator(10**9,Gw2_bounded, test2)
md_calculator(10**9,Gw2_unbounded, test2)
plt.show()

print('MDD3')
gs = gridspec.GridSpec(3, 3)
md_calculator(10**9, tradw3_bounded, test3)
md_calculator(10**9, tradw3_unbounded, test3)
md_calculator(10**9,Gw3_bounded, test3)
md_calculator(10**9,Gw3_unbounded, test3)
plt.show()

print('MDD4')
gs = gridspec.GridSpec(3, 3)
md_calculator(10**9, tradw4_bounded, test4)
md_calculator(10**9, tradw4_unbounded, test4)
md_calculator(10**9,Gw4_bounded, test4)
md_calculator(10**9,Gw4_unbounded, test4)
plt.show()


print('MDD5')
gs = gridspec.GridSpec(3, 3)
md_calculator(10**9, tradw5_bounded, test5)
md_calculator(10**9, tradw5_unbounded, test5)
md_calculator(10**9,Gw5_bounded, test5)
md_calculator(10**9,Gw5_unbounded, test5)
plt.show()

print('MDD6')
gs = gridspec.GridSpec(3, 3)
md_calculator(10**9, tradw6_bounded, test6)
md_calculator(10**9, tradw6_unbounded, test6)
md_calculator(10**9,Gw6_bounded, test6)
md_calculator(10**9,Gw6_unbounded, test6)
plt.show()


print('MDD7')
gs = gridspec.GridSpec(3, 3)
md_calculator(10**9, tradw7_bounded, test7)
md_calculator(10**9, tradw7_unbounded, test7)
md_calculator(10**9,Gw7_bounded, test7)
md_calculator(10**9,Gw7_unbounded, test7)
plt.show()


print("WHOLE PERIOD")
gs = gridspec.GridSpec(3, 3)
cum_ret(test1,tradw1_bounded, test2, tradw2_bounded, test3, tradw3_bounded,test4,
        tradw4_bounded, test5, tradw5_bounded, test6, tradw6_bounded, test7,tradw7_bounded, 10**9)
cum_ret(test1,tradw1_unbounded, test2, tradw2_unbounded, test3, tradw3_unbounded,test4,
        tradw4_unbounded, test5, tradw5_unbounded, test6, tradw6_unbounded, test7,tradw7_unbounded, 10**9)
cum_ret(test1,Gw1_bounded, test2, Gw2_bounded, test3, Gw3_bounded,
        test4,Gw4_bounded, test5, Gw5_bounded, test6, Gw6_bounded,test7, Gw7_bounded,10**9)
cum_ret(test1,Gw1_unbounded, test2, Gw2_unbounded, test3, Gw3_unbounded,
        test4,Gw4_unbounded, test5, Gw5_unbounded, test6, Gw6_unbounded,test7, Gw7_unbounded,10**9)
plt.show()
#


Trad_bounded = [cum_ret1(test1,tradw1_bounded),cum_ret1(test2,tradw2_bounded),
        cum_ret1(test3,tradw3_bounded),cum_ret1(test4,tradw4_bounded),
        cum_ret1(test5,tradw5_bounded),cum_ret1(test6,tradw6_bounded),cum_ret1(test7,tradw7_bounded)]
Trad_unbounded = [cum_ret1(test1,tradw1_unbounded),cum_ret1(test2,tradw2_unbounded),
        cum_ret1(test3,tradw3_unbounded),cum_ret1(test4,tradw4_unbounded),
        cum_ret1(test5,tradw5_unbounded),cum_ret1(test6,tradw6_unbounded),cum_ret1(test7,tradw7_unbounded)]

Gauss_bounded = [cum_ret1(test1,Gw1_bounded),cum_ret1(test2,Gw2_bounded),cum_ret1(test3,Gw3_bounded)
                ,cum_ret1(test4,Gw4_bounded),cum_ret1(test5,Gw5_bounded),cum_ret1(test6,Gw6_bounded),
                                                                          cum_ret1(test7,Gw7_bounded)]

Gauss_unbounded = [cum_ret1(test1,Gw1_unbounded),cum_ret1(test2,Gw2_unbounded),cum_ret1(test3,Gw3_unbounded)
                ,cum_ret1(test4,Gw4_unbounded),cum_ret1(test5,Gw5_unbounded),cum_ret1(test6,Gw6_unbounded),
                                                                          cum_ret1(test7,Gw7_unbounded)]


Sharpe = {'trad_bounded' : Trad_bounded, 'trad_unbounded': Trad_unbounded, 'G_bounded': Gauss_bounded, 'G_unbounded': Gauss_unbounded}
Sharpe = pd.DataFrame(Sharpe)

print(Sharpe)

#PERIOD 1
port1 = {'trad_bounded':tradw1_bounded,'trad_unbounded' : tradw1_unbounded, 'Gauss_bounded': Gw1_bounded, 'Gauss_unbounded': Gw1_unbounded}
port1 = pd.DataFrame(port1)
port1.index = train1.columns
print(port1)

#PERIOD 2
port2 = {'trad_bounded':tradw2_bounded,'trad_unbounded' : tradw2_unbounded, 'Gauss_bounded': Gw2_bounded, 'Gauss_unbounded': Gw2_unbounded}
port2 = pd.DataFrame(port2)
port2.index = train1.columns
print(port2)

#PERIOD 3
port3 = {'trad_bounded':tradw3_bounded,'trad_unbounded' : tradw3_unbounded, 'Gauss_bounded': Gw3_bounded, 'Gauss_unbounded': Gw3_unbounded}
port3 = pd.DataFrame(port3)
port3.index = train1.columns
print(port3)

#PERIOD 4
port4 = {'trad_bounded':tradw4_bounded,'trad_unbounded' : tradw4_unbounded, 'Gauss_bounded': Gw4_bounded, 'Gauss_unbounded': Gw4_unbounded}
port4 = pd.DataFrame(port4)
port4.index = train1.columns
print(port4)

#PERIOD 5
port5 = {'trad_bounded':tradw5_bounded,'trad_unbounded' : tradw5_unbounded, 'Gauss_bounded': Gw5_bounded, 'Gauss_unbounded': Gw5_unbounded}
port5 = pd.DataFrame(port5)
port5.index = train1.columns
print(port5)

#PERIOD 6
port6 = {'trad_bounded':tradw6_bounded,'trad_unbounded' : tradw6_unbounded, 'Gauss_bounded': Gw6_bounded, 'Gauss_unbounded': Gw6_unbounded}
port6 = pd.DataFrame(port6)
port6.index = train1.columns
print(port6)

#PERIOD 7
port7 = {'trad_bounded':tradw7_bounded,'trad_unbounded' : tradw7_unbounded, 'Gauss_bounded': Gw7_bounded, 'Gauss_unbounded': Gw7_unbounded}
port7 = pd.DataFrame(port7)
port7.index = train1.columns
print(port7)

#PERIOD 8
port8 = {'trad_bounded':tradw8_bounded,'trad_unbounded' : tradw8_unbounded, 'Gauss_bounded': Gw8_bounded, 'Gauss_unbounded': Gw8_unbounded}
port8 = pd.DataFrame(port8)
port8.index = train1.columns
print(port8)

os.chdir('D:/data-vietquant/fundamental')

# port1.to_csv('Port_test_1.csv')
# port2.to_csv('Port_test_2.csv')
# port3.to_csv('Port_test_3.csv')
# port4.to_csv('Port_test_4.csv')
# port5.to_csv('Port_test_5.csv')
# port6.to_csv('Port_test_6.csv')
# port7.to_csv('Port_test_7.csv')
# port8.to_csv('Port_test_8.csv')
# #
# return_0 = np.dot(train1,Gw1_bounded).reshape(-1,1)
# return_1 = np.dot(test1, Gw1_bounded).reshape(-1,1)
# return_2 = np.dot(test2, Gw2_bounded).reshape(-1,1)
# return_3 = np.dot(test3, Gw3_bounded).reshape(-1,1)
# return_4 = np.dot(test4, Gw4_bounded).reshape(-1,1)
# return_5 = np.dot(test5, Gw5_bounded).reshape(-1,1)
# return_6 = np.dot(test6, Gw6_bounded).reshape(-1,1)
# return_7 = np.dot(test7, Gw7_bounded).reshape(-1,1)
# dis_ret = np.vstack((return_0, return_1, return_2,return_3,return_4,return_5,return_6,return_7))
# dis_ret = pd.DataFrame(dis_ret, index= train8.index)
# print(dis_ret)
# os.chdir('D:/data-vietquant/fundamental')
# dis_ret.to_csv('daily_return.csv')



