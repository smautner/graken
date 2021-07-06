import random
from graken.main import dumpfile
import dirtyopts as opts
import structout as so
import random
import json
import networkx.readwrite.json_graph as sg
import networkx as nx
from graken.ml import vector

chemfiles='''
# start graphs
--json str ''     # we want theese: "AID119.sdf.json AID1345082.sdf.json AID588590.sdf.json AID624202.sdf.json AID977611.sdf.json"
--out str ''
'''

############
#  CHEM STUFF COMES LATER 
############# 

def prep(aidfile): 
        stuff =load_chem(aidfile)
        random.shuffle(stuff)
        return stuff
    
    
def load_chem(AID):
    with open(AID, 'r') as handle:
        js = json.load(handle)
        res = [sg.node_link_graph(jsg) for jsg in js]
        res = [g for g in res if len(g)> 2]
        res = [g for g in res if nx.is_connected(g)]  # rm not connected crap
        for g in res:g.graph={}

        zz = vector.Vectorizer()
        res2 = list(zz._duplicate_rm(res, {}))

        print ("duplicates in chem files:%d"% (len(res)-len(res2)))
        zomg = [(len(g),g) for g in res]
        zomg.sort(key=lambda x:x[0])
        cut = int(len(res)*.1)
        res2 = [b for l,b in zomg[cut:-cut]]
    return res2

if __name__=="__main__":
    args = opts.parse(chemfiles)
    graphs = prep(args.json) 
    dumpfile(graphs, args.out)


