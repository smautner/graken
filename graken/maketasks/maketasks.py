import random
from graken.maketasks import random_graphs as rg, rule_rand_graphs as rrg
from graken.main import dumpfile
import dirtyopts as opts
import structout as so

docStartgraphs='''
# start graphs
--nocycles 
--ngraphs int default:30
--graphsize int default:8
--node_labels int default:4
--edge_labels int default:2
--labeldistribution str default:uniform
--maxdeg int default:3
'''

docIter='''
--numgr int default:501
--iter int default:3
--out str default:nugraphs
'''


if __name__=="__main__":
    args = opts.parse(docStartgraphs)
    startgraphs = rg.make_graphs_static(**args.__dict__)
    so.gprint(startgraphs[:4])

    args=opts.parse(docIter)
    out  = args.__dict__.pop('out')
    graphs = rrg.rule_rand_graphs(startgraphs, **args.__dict__)

    dumpfile(graphs, out)


############
#  CHEM STUFF COMES LATER 
############# 
'''
def get_chem_filenames():
    # these are size ~500
    files="""AID1224837.sdf.json  AID1454.sdf.json  AID1987.sdf.json  AID618.sdf.json     AID731.sdf.json     AID743218.sdf.json  AID904.sdf.json AID1224840.sdf.json  AID1554.sdf.json  AID2073.sdf.json  AID720709.sdf.json  AID743202.sdf.json  AID828.sdf.json"""
    # these are size ~4000
    files="""AID119.sdf.json
            AID1345082.sdf.json
            AID588590.sdf.json
            AID624202.sdf.json
            AID977611.sdf.json"""
    files = files.split()
    return files


def make_chem_task_file():
    files = get_chem_filenames()
    res=[]
    for f in files:
        stuff =load_chem("chemsets/"+f)
        random.shuffle(stuff)
        res.append(stuff)
    dumpfile(res, ".chemtasks"
    
    
def load_chem(AID):
    import json
    import networkx.readwrite.json_graph as sg
    import networkx as nx
    import exploration.pareto as pp
    from structout import gprint
    with open(AID, 'r') as handle:
        js = json.load(handle)
        res = [sg.node_link_graph(jsg) for jsg in js]
        res = [g for g in res if len(g)> 2]
        res = [g for g in res if nx.is_connected(g)]  # rm not connected crap
        for g in res:g.graph={}
        zz=pp.MYOPTIMIZER()
        res2 = list(zz._duplicate_rm(res))
        print ("duplicates in chem files:%d"% (len(res)-len(res2)))
        print (zz.collisionlist)
        #for a,b in zz.collisionlist:
        #    gprint([res[a],res[b]])
        zomg = [(len(g),g) for g in res]
        zomg.sort(key=lambda x:x[0])
        cut = int(len(res)*.1)
        res2 = [b for l,b in zomg[cut:-cut]]
    return res2

'''
