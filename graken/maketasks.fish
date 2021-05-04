 parallel -j 32 --bar  python maketasks/maketasks.py --numgr 10000 --out tasks/task_{1} ::: (seq 0 99)
