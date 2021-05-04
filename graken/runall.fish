source setvar.fish 

set i  default greedy paretogreed pareto_only # random 
set stuff  -j 32 --bar --results LOL  -j 32 --bar --results LOL python main.py   
set prog --shuffle {3} --i tasks/task_{2} --n_train 500 --n_iter 3 --out res/{1}_{2}_{3} --pareto {1}
set args ::: $i ::: (seq 0 99) ::: (seq 0 5)


parallel $stuff $prog $args
