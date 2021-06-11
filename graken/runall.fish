set -x 'MKL_NUM_THREADS' 1 
set -x 'NUMBA_NUM_THREADS' 1
set -x 'OMP_NUM_THREADS' 1
set -x 'OPENBLAS_NUM_THREADS' 1


function showres
python -c 'import main as m
from collections import Counter
from lmz import *
import structout as so
import os

DIR = "res"

f = map(lambda x: m.loadfile(f"{DIR}/{x}"), os.listdir(DIR))
res, dur, time =  Transpose(f)
succ = sum(res)
rtime = sum(time)
print (f"succ: {succ}/{len(res)} ({succ/len(res):.2f}) ")
print (f"time: {rtime:.2f} ({rtime/len(res):.2f}) ")

sucstep = [d for r,d in zip(res,dur) if r ]
if sucstep:
    sucstep.sort()
    print (f"sucstep: median:{sucstep[len(sucstep)//2]}  max:{max(sucstep)}")
    #so.dprint( Counter(sucstep),length = max(sucstep))
'
end 




set paraargs  -j 32 --bar  --joblog lol.txt -j 32 --bar python main.py 
set static1 --n_train 500 --n_iter 10 --cipselector graph --cipselector_k 10  --removedups True --filter_min_cip 2 --keepgraphs 30
set static2 --size_limiter 'lambda x:int\(x.mean\(\)\)+6' # i should use theese: " 
set prog --shuffle 13 --i tasks84/task_{2} --out res/{1}_{2} --pareto {1}
set pareto greedy #geometric paretogreed greedy pareto_only default
set argvalues ::: $pareto ::: (seq 0 31)  

rm res/*
parallel $paraargs $static1 $static2 $prog $argvalues

showres


