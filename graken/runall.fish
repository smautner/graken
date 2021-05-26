source setvar.fish 

set paraargs  -j 32 --bar --results LOL  -j 32 --bar  python main.py   
set static --n_train 500 --n_iter 10 --cipselector 2 --cipselector_k 1

set prog --shuffle {3} --i tasks/task_{2} --out res/{1}_{2}_{3} --pareto {1}
# ARGS
set pareto greedy paretogreed pareto_only default
set argvalues ::: $pareto ::: (seq 0 31) ::: (seq 0 0)

#parallel $paraargs $static $prog $argvalues



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

showres

