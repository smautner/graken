 parallel -j 32 --bar  python maketasks/maketasks.py --iter 4 --numgr 6000 --out tasks84/task_{1} ::: (seq 0 99)
 parallel -j 32 --bar  python maketasks/maketasks.py --iter 3 --numgr 6000 --out tasks83/task_{1} ::: (seq 0 99)
 parallel -j 32 --bar  python maketasks/maketasks.py --iter 2 --numgr 6000 --out tasks82/task_{1} ::: (seq 0 99)
