source setvar.fish 

set paraargs  -j 32 --bar --results LOL  -j 32 --bar --results LOL python main.py   
set static --n_train 500 --n_iter 10 --cipselector 2 --cipselector_k 1



set prog --shuffle {3} --i tasks/task_{2} --out res/{1}_{2}_{3} --pareto {1}
# ARGS
set pareto greedy paretogreed pareto_only default
set argvalues ::: $pareto ::: (seq 0 49) ::: (seq 0 2)


parallel $paraargs $static $prog $argvalues
