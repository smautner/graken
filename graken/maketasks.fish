 mkdir tasks2_3
 mkdir tasks2_4
 mkdir tasks2_5


 parallel -j 32 --bar  python maketasks/maketasks.py --ngraphs 30\
 --node_labels 4 --edge_labels 2 --bottleneck 2000\
 --iter {2} --numgr 3000  --graphsize 8 --out tasks2_{2}/task_{1} ::: (seq 0 31) ::: 4


 #parallel -j 32 --bar  python maketasks/maketasks.py --iter 3 --numgr 6000 --out tasks83/task_{1} ::: (seq 0 99)
 #parallel -j 32 --bar  python maketasks/maketasks.py --iter 4 --numgr 6000 --out tasks84/task_{1} ::: (seq 0 99)


xonsh -c '
from main import loadfile as load
files = `tasks2_3/.*`
print ("############# AVG SIZE ##########")
print(sum([len(load(e)) for e in files])/len(files))
'
