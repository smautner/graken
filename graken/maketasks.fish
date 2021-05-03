
for rep in (seq 0 99)
    echo tasks/task_$rep
end | parallel -j 32 --bar  python maketasks/maketasks.py --numgr 10000 --out 
