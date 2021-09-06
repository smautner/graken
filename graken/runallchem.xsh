import main as m
from collections import Counter
from lmz import *
import structout as so
import os
import sys
import numpy as np

'''
# I NEED TO GET  THEESE TO WORK
set -x 'MKL_NUM_THREADS' 1
set -x 'NUMBA_NUM_THREADS' 1
set -x 'OMP_NUM_THREADS' 1
set -x 'OPENBLAS_NUM_THREADS' 1
'''

#args = {'aid': '119  1345082 624202  977611', 'grammar': 'eden nonlookahead', 'taskid' }

paraargs  = " -j 32 --bar  --joblog chemlog.txt -j 32 --bar python main.py "

static1 = '''--n_train 2000 --n_iter 5 --cipselector graph --cipselector_k 10 --removedups False
	--filter_min_cip 1
    --size_limiter "lambda x:int(x.mean()*1.25)"
    --contextsize 2
	--n_neighbors {4}
	--grammarname {2} '''

prog  = "--shuffle 4 --i chemtasks/{1} --taskid {3} --out reschem/{2}_{4}_{3}_{1} --pareto greedy "


toargs = lambda x : " ".join(map(str,x))
argvalues  = " :::  119  1345082 624202  977611"
argvalues += " ::: eden nonlookahead"
argvalues += " ::: " + toargs(Range(50))
argvalues += " ::: " + toargs(range(50,1000,100))


if False:
    # recalculate
    rm reschem/*
    parallel @( (paraargs+static1+prog+argvalues).split()  )
    exit()



##############
# collect data
##################

numneighs = Range(50,1000,100)
grammarsize = []
timeperstep =[]
steptime_norm = []
r1,r2 = [],[]

# lookahead grammar
for numneigh in numneighs:
    $num = numneigh
    f = Map(lambda x: m.loadfile(x), `reschem/eden_$num.*`)
    res, steps, optitime , timetotal, productions =  Transpose(f)
    grammarsize.append( np.mean(productions)  )
    timeperstep.append( sum(optitime)/sum(steps)  )
    r1.append( sum(res)/ len(res))


# expand all grammar
for numneigh in numneighs:
    $num = numneigh
    f = Map(lambda x: m.loadfile(x), `reschem/nonlookahead_$num.*`)
    res, steps, optitime , timetotal, productions =  Transpose(f)
    steptime_norm.append( sum(optitime)/sum(steps)  )
    r2.append( sum(res)/ len(res))






##########
# plotting function  and a smooth curve projecting into the future
###########

import matplotlib
matplotlib.use("module://matplotlib-sixel")
import matplotlib.pyplot as plt
plt.set_loglevel("info")
from scipy.optimize import curve_fit
def f(x, a, b,c):
    return a* np.log(x+c) +b


def plot12(x,y, xprojection):
    plt.plot(x, y )
    popt, pcov = curve_fit(f, x ,y)
    print(f" {popt=}")
    X = np.linspace(0, xprojection, 100)
    plt.plot(X, Map(lambda x:f(x,*popt), X))
    #plt.yscale("log")



##########
# plot speedup ratio
#############
datay = [b/a for (a,b) in  zip(timeperstep, steptime_norm)]
plot12(numneighs,datay, 1000)

plt.plot(numneighs,[g/10000 for g in grammarsize], label = 'grammarsize /e4')
plt.plot(numneighs,r1, label = 'correct lookahead')
plt.plot(numneighs,r2, label = 'correct default')



#####
# finish plot
#########
plt.grid()
plt.legend()
plt.show()
plt.close()


print(f" {numneighs=}")
print(f" {grammarsize=}")
print(f" {timeperstep=}")
print(f" {r1=}")
print(f" {r2=}")



######
# graphsizemax increase -> better speedup


# alternative to show res is to grep the results from the logfile
#grep tasks82 lol.txt | awk '{total+=$7} END {print total}'
#for i in (seq 1 5); grep tasks8$i log.txt | awk '{total+=$7} END {print total}') ; end
# greedy for 1..5  steps of grammar mixing:
# 29 26 24 18 23 # run again with ..99
