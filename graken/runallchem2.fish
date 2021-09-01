set -x 'MKL_NUM_THREADS' 1
set -x 'NUMBA_NUM_THREADS' 1
set -x 'OMP_NUM_THREADS' 1
set -x 'OPENBLAS_NUM_THREADS' 1

###############
##t this copy of runallchem is for the speedup comparison...
############

set paraargs  -j 32 --bar  --joblog chemlog.txt -j 32 --bar python main.py

set static1 --n_train 2000 --n_iter 5 --cipselector graph --cipselector_k 10 --removedups False \
	--filter_min_cip 1 \
    --size_limiter 'lambda x:int\(x.mean\(\)\*1.25\)' \
	--contextsize 2 \
	--n_neighbors {4} \
	--grammarname {2}

set prog --shuffle 4 --i chemtasks/{1} --taskid {3} --out reschem/{1}_{2}_{3} --pareto greedy
set argvalues :::  119  1345082 624202  977611 ::: eden nonlookahead ::: (seq 0 2) ::: (seq 50 50 950)

rm reschem/*
parallel $paraargs $static1 $static2 $prog $argvalues

