set -x 'MKL_NUM_THREADS' 1
set -x 'NUMBA_NUM_THREADS' 1
set -x 'OMP_NUM_THREADS' 1
set -x 'OPENBLAS_NUM_THREADS' 1



########
#  normal run...
######
set paraargs  -j 32 --bar  --joblog log.txt   python main.py  --specialset True
set params --n_train 1000 --n_iter 10 --cipselector graph --cipselector_k 10 \
--removedups True --filter_min_cip 1 --keepgraphs 30 --n_neighbors {2} \
--size_limiter 'lambda x:int\(x.mean\(\)\*1.4+.5\)' \
--task_id 4 --v_radius 2 --v_distance 1  \
--shuffle -1 --i tasks83/task_{1} --out res_k10/{2}_{1} --pareto greedy  --contextsize 1
set argvalues ::: (seq 0 5) ::: (seq 50 50 950)


##########
# expand all options per round
##########
set paraargs  -j 32 --bar  --joblog log.txt   python main.py  --specialset True
set params --n_train 1000 --n_iter 10 --cipselector graph --cipselector_k 5000 \
--removedups True --filter_min_cip 1 --keepgraphs 30 --n_neighbors {2} \
--size_limiter 'lambda x:int\(x.mean\(\)\*1.4+.5\)' \
--task_id 4 --v_radius 2 --v_distance 1  --grammarname blablabla\
--shuffle -1 --i tasks83/task_{1} --out res_k5k/{2}_{1} --pareto greedy  --contextsize 1
set argvalues ::: (seq 0 5) ::: (seq 50 50 950)



#parallel $paraargs $params $argvalues
#for x in (seq 1 5); set c (find . -name {$x}_\*); showres $c end
#showres (find res -type f)












'
xonsh -c
"""
import main as m
from collections import counter
from lmz import *
import structout as so
import os
import sys

f = map(lambda x: m.loadfile(x), `res/.*`)
res, dur, time , timetotal, productions =  transpose(f)

# success:
succ = sum(res)
print (f"succ: {succ}/{len(res)} ({succ/len(res):.2f}) ")

# numper of steps when successfull
sucstep = [d for r,d in zip(res,dur) if r ]
if sucstep:
	sucstep.sort()
	print (f"sucstep: median:{sucstep[len(sucstep)//2]}  max:{max(sucstep)}")

# time total and per step
steptime = sum(time) / sum(dur)
totaltime = sum(timetotal)
print (f"time: {totaltime:.2f} (step: {steptime:.2f})  (prep_avg: {totaltime/sum(time):.2f})")
print (f"r/h: {3600*(sum(res)/totaltime):.2f}")
"""
'



# alternative to show res is to grep the results from the logfile
#grep tasks82 lol.txt | awk '{total+=$7} END {print total}'


#for i in (seq 1 5); grep tasks8$i log.txt | awk '{total+=$7} END {print total}') ; end


# greedy for 1..5  steps of grammar mixing:
# 29 26 24 18 23 # run again with ..99
