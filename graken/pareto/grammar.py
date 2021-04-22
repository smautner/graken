import time
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix
import logging
from toolz import concat

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
        super(gradigrammar,self).__init__(**kwargs)
        self.selelector = selector
        self.selelector_k = selektor_k
        self.graphsizelimiter = graphsizelimiter


    def fit(self,graphs, target):
        timenow = time.time()
        super(gradigrammar, self).fit(graphs)
        self.genmaxsize = self.graphsizelimiter(graphs)
        self.target= target
        logging.debug("graph generation: %.2fs" % (time.time() - timenow))
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
        timenow = time.time()
        proddict = defaultdict(list)
        for i,graph in enumerate(graphs):
            self.get_productions(proddict, graph, i)

        productions = list(concat(map(self.selectbest, proddict.values())))
        graphs = [ self.substitute_core(*prod) for prod in productions ]
        graphs = [ g for g in graphs if g ]

        logging.debug("neighbor generation: %.2fs" % (time.time() - timenow))
        return graphs

    def get_productions(self, pdict, graph,grid):
        grlen = len(graph)
        grvec = self.vectorizer.raw(graph)
        current_cips = self._get_cips(graph)
        for current_cip in current_cips:
            pdict[self.mkkey(grid, current_cip.interface_hash)]+= [(graph,grvec, current_cip,concip)
                                for concip in self._get_congruent_cips(current_cip) if len(concip.core_nodes) + grlen - len(current_cip.core_nodes) <= self.genmaxsize]

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
        predicted_vectors = np.vstack([self.vectorizer.normalize(csr_matrix(vec - cur.core_vec + con.core_vec).T) for gra,vec,cur,con in stuff])
        scores = np.dot(self.target, predicted_vectors)
        goodindex = np.argsort(scores)[-self.selelector_k:]
        return [ (stuff[i][0],stuff[i][2], stuff[i][3]) for i in goodindex]