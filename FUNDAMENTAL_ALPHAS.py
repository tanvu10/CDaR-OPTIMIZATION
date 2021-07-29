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

def md_calculator(booksize, weight, dataframe):
    dis_ret = np.dot(dataframe, weight)
    cum_ret = np.cumsum(dis_ret)*(10**10)
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




#IMPORT
os.chdir('D:/data-vietquant/fundamental/fundamental_data')
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


#import simulation data
os.chdir('D:/data-vietquant/fundamental/fundamental_simulation')
simu_G5 = pd.read_csv('X5_G.csv')
simu_G5 = simu_G5.iloc[:,1:]

simu_G6 = pd.read_csv('X6_G.csv')
simu_G6 = simu_G6.iloc[:,1:]

simu_G7 = pd.read_csv('X7_G.csv')
simu_G7 = simu_G7.iloc[:,1:]
# print(simu_G7)


os.chdir('D:/data-vietquant/fundamental/fundamental_cov')
cov5_G = pd.read_csv('cov5_G.csv')
cov5_G = cov5_G.iloc[:,1:]

cov6_G = pd.read_csv('cov6_G.csv')
cov6_G = cov6_G.iloc[:,1:]

cov7_G = pd.read_csv('cov7_G.csv')
cov7_G = cov7_G.iloc[:,1:]
# print(cov7_G)

#traditional
tradw5_bounded = trad_sharpe_with_bounded(train5, 6,0.06, 0.1, test5)
tradw5_unbounded = trad_sharpe_with_unbounded(train5, 6,0.06, test5)

tradw6_bounded = trad_sharpe_with_bounded(train6, 6,0.06, 0.1, test6)
tradw6_unbounded = trad_sharpe_with_unbounded(train6, 6,0.06, test6)


tradw7_bounded = trad_sharpe_with_bounded(train7, 6,0.06, 0.1, test7)
tradw7_unbounded = trad_sharpe_with_unbounded(train7,6, 0.06, test7)
# print(tradw7_bounded)
# print(tradw7_unbounded)
#copula
Gw5_bounded = copula_sharpe_with_bounded(simu_G5,6, 0.06, 0.1,test1, cov5_G)
Gw5_unbounded = copula_sharpe_with_unbounded(simu_G5,6, 0.06,test1, cov5_G)

Gw6_bounded = copula_sharpe_with_bounded(simu_G6,6, 0.06, 0.1,test2, cov6_G)
Gw6_unbounded = copula_sharpe_with_unbounded(simu_G6,6, 0.06,test2, cov6_G)

Gw7_bounded = copula_sharpe_with_bounded(simu_G7,6, 0.06, 0.1,test7, cov7_G)
Gw7_unbounded = copula_sharpe_with_unbounded(simu_G7,6, 0.06,test7, cov7_G)
# print(Gw7_bounded)
# print(Gw7_unbounded)


#PERIOD 7
port7 = {'trad_bounded':tradw7_bounded,'trad_unbounded' : tradw7_unbounded, 'Gauss_bounded': Gw7_bounded, 'Gauss_unbounded': Gw7_unbounded}
port7 = pd.DataFrame(port7)
port7.index = train1.columns[1:]
print(port7)
os.chdir('D:/data-vietquant/fundamental')
# port7.to_csv('Port_test_7.csv')

print('MDD5')
gs = gridspec.GridSpec(3, 3)
md_calculator(10**10, tradw5_bounded, test5)
md_calculator(10**10, tradw5_unbounded, test5)
md_calculator(10**10,Gw5_bounded, test5)
md_calculator(10**10,Gw5_unbounded, test5)
plt.show()

print('MDD6')
gs = gridspec.GridSpec(3, 3)
md_calculator(10**10, tradw6_bounded, test6)
md_calculator(10**10, tradw6_unbounded, test6)
md_calculator(10**10,Gw6_bounded, test6)
md_calculator(10**10,Gw6_unbounded, test6)
plt.show()


print('MDD7')
gs = gridspec.GridSpec(3, 3)
md_calculator(10**10, tradw7_bounded, test7)
md_calculator(10**10, tradw7_unbounded, test7)
md_calculator(10**10,Gw7_bounded, test7)
md_calculator(10**10,Gw7_unbounded, test7)
plt.show()
