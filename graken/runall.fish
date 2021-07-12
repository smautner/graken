set -x 'MKL_NUM_THREADS' 1 
set -x 'NUMBA_NUM_THREADS' 1
set -x 'OMP_NUM_THREADS' 1
set -x 'OPENBLAS_NUM_THREADS' 1


function showres
python -c "ARGS='$argv';"'import main as m
from collections import Counter
from lmz import *
import structout as so
import os
import sys

DIR = "res"
#f = map(lambda x: m.loadfile(f"{DIR}/{x}"), sys.argv[1:]) 
f = Map(lambda x: m.loadfile(x), ARGS.split()) 
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




set paraargs  -j 32 --bar  --joblog log.txt   python main.py  --specialset True
set static1 --n_train 500 --n_iter 10 --cipselector cip --cipselector_k 1  --removedups True --filter_min_cip 2 --keepgraphs 30
set static2 --size_limiter 'lambda x:int\(x.mean\(\)\*1.4+.5\)' # i should use theese: " 
set prog --shuffle -1 --i tasks8{2}/task_{1} --out res/{2}_{1} --pareto greedy  --contextsize 1

set argvalues  :::  (seq 0 99) ::: 4 5 

rm res/*
parallel $paraargs $static1 $static2 $prog $argvalues

#for x in (seq 1 5); set c (find . -name {$x}_\*); showres $c end
showres (find res -type f)

# alternative to show res is to grep the results from the logfile
#grep tasks82 lol.txt | awk '{total+=$7} END {print total}'


#for i in (seq 1 5); grep tasks8$i log.txt | awk '{total+=$7} END {print total}') ; end 


# greedy for 1..5  steps of grammar mixing: 
# 29 26 24 18 23 # run again with ..99
