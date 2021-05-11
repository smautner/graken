import time
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix
import logging
from toolz import concat
from graphlearn.cipcorevector import LsggCoreVec as lsgg

class gradigrammar(lsgg):
    '''
    adapting the grammar :)
    - cip selection:
        - graphsize limiter (also needs fitting)
        - cip selection: 3 levels and we select the k best
    - init now needs to do more
    '''

    def __init__(self,graphsizelimiter= None,
                 vectorizer=None,
                 selector =2,
                 selektor_k=1, **kwargs): # expected kwargs: radii thickness filter_min_cip
        kwargs['core_vec_decomposer']=vectorizer.decompose
        self.vectorizer = vectorizer
        super(gradigrammar,self).__init__(**kwargs)
        self.selelector = selector
        self.selelector_k = selektor_k
        self.graphsizelimiter = graphsizelimiter


    def fit(self,graphs, target):
        timenow = time.time()
        super(gradigrammar, self).fit(graphs)
        self.genmaxsize = self.graphsizelimiter(np.array([len(g) + g.size() for g in graphs]))
        self.target= target.toarray().T
        logging.debug(f"grammar generation: %.2fs ({len(graphs)} graphs)" % (time.time() - timenow))
        logging.debug(f"graphsizelimit: {self.genmaxsize}")
    

    def expand_neighbors(self, graphs):
        '''
        the plan: 
        - there are 3 cip selectors:
        - 0: [graph, cip_congr]
        - 1: [graph,cc][graph,cc]
        - 2: same as one but there is a list for each start-cip now
        once i have the lists i can select the select_k best ones
        i guess i can just throw everything on a default dict with the key as something that ids the list  
        '''
        timestart = time.time()
        proddict = defaultdict(list)
        for i,graph in enumerate(graphs):
            self.get_productions(proddict, graph, i)

        sumprod = sum([len(x) for x in proddict.values()])
        productions = list(concat(map(self.selectbest, proddict.values())))
        logging.debug(f"expand neighbors: {sumprod} productions reduced to {len(productions)}  ({time.time() - timestart:.3}s )")

        timemid = time.time()
        graphs = [ self._substitute_core(*prod) for prod in productions ]
        graphs = [ g for g in graphs if g ]
        logging.debug(f"expand neighbors: generating graphs  ({time.time() - timemid:.3}s )")
        return graphs

    def get_productions(self, pdict, graph,grid):
        grlen = len(graph)
        vec = self.vectorizer.raw(graph)
        current_cips = self._get_cips(graph)
        for current_cip in current_cips:
            if productions := [(graph, vec, current_cip,concip) for concip in self._get_congruent_cips(current_cip) 
                    if len(concip.core_nodes) + grlen - len(current_cip.core_nodes) <= self.genmaxsize]:
                pdict[self.mkkey(grid, current_cip.interface_hash)]+= productions

    def mkkey(self, grid, ihash):
        if self.selelector == 0:
            return 0
        elif self.selelector ==1:
            return grid
        return f"{grid}_{ihash}"

    def selectbest(self,stuff):
        '''
            stuff is a list:[startgraph,startgraph_vec, cip_con, cip_congru]
            return: selctor_k best productions in the list
        '''
        # score productions:
        #from graken.main import dumpfile


        myvectors = np.empty((len(stuff),stuff[0][1].shape[1]))
        for i,(gra,vec,cur,con) in enumerate(stuff):
            myvectors[i] = vec  - cur.core_vec + con.core_vec

        #myvectors = np.vstack([vec - cur.core_vec + con.core_vec for gra,vec,cur,con in stuff]) 
        myvectors = self.vectorizer.normalize(myvectors)
        scores = np.dot(myvectors, self.target)
        goodindex = np.argsort(scores.T)[0][-self.selelector_k:]
        return [ (stuff[i][0],stuff[i][2], stuff[i][3]) for i in goodindex]
