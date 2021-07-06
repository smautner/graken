 mkdir tasks81
 mkdir tasks82
 mkdir tasks83
 mkdir tasks84
 mkdir tasks85
 parallel -j 32 --bar  python maketasks/maketasks.py --iter {2} --numgr 6000 --out tasks8{2}/task_{1} ::: (seq 0 99) ::: 1 2 3 4 5 
 #parallel -j 32 --bar  python maketasks/maketasks.py --iter 3 --numgr 6000 --out tasks83/task_{1} ::: (seq 0 99)
 #parallel -j 32 --bar  python maketasks/maketasks.py --iter 4 --numgr 6000 --out tasks84/task_{1} ::: (seq 0 99)
