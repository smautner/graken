source setvar.fish 

set stuff  -j 32 --bar --results LOL  -j 32 --bar --results LOL python main.py   

set static --n_train 500 --n_iter 3 --cipselector 1 --cipselector_k 66
set prog --shuffle {3} --i tasks/task_{2} --out res/{1}_{2}_{3} --pareto {1}

# ARGS
set pareto default greedy paretogreed pareto_only 
set argvalues ::: $pareto ::: (seq 0 99) ::: (seq 0 5)


parallel $stuff $prog $static $argvalues
