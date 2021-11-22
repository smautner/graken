import time
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix, vstack, dok_matrix
import logging
logger=logging.getLogger(__name__)
from toolz import concat
from graphlearn.cipcorevector import LsggCoreVec as lsgg
from graphlearn import lsgg_core_interface_pair as lsggcip
import eden.graph as eg


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
                 selector = 'cip',
                 selektor_k=1, **kwargs): # expected kwargs: radii thickness filter_min_cip
        kwargs['core_vec_decomposer']=vectorizer.decompose
        self.vectorizer = vectorizer
        super(gradigrammar,self).__init__(**kwargs)
        self.selelector = selector
        self.selelector_k = selektor_k
        self.graphsizelimiter = graphsizelimiter
        self.isfit = False


    def fit(self,graphs,landmarks, target):
        timenow = time.time()
        super(gradigrammar, self).fit(graphs)
        self.genmaxsize = self.graphsizelimiter(np.array([len(g) + g.size() for g in landmarks]))
        self.target= target.T# target.toarray().T
        logger.debug(f"grammar generation: %.2fs ({len(graphs)} graphs)" % (time.time() - timenow))
        logger.debug(f"graphsizelimit: {self.genmaxsize}")
        self.isfit=True


    def _get_cores(self, graph):
        cores = super(gradigrammar,self)._get_cores(graph)
        if self.isfit:
           cores += [core for core in lsggcip.get_cores_closeloop(graph, self.radii) if core]
        return cores

    def expand_neighbors(self, graphs):
        '''
        the plan:
        - there are 3 cip selectors:
        - pop: [graph, cip_congr]
        - graph: [graph,cc][graph,cc]
        - cip: same as one but there is a list for each start-cip now
        once i have the lists i can select the select_k best ones
        i guess i can just throw everything on a default dict with the key as something that ids the list
        '''
        timestart = time.time()
        proddict = defaultdict(list)
        for i,graph in enumerate(graphs):
            self.get_productions(proddict, graph, i)

        sumprod = sum([len(x) for x in proddict.values()])
        self.predscores = []
        productions = list(concat(map(self.selectbest, proddict.values())))
        logger.debug(f"expand neighbors: {sumprod} productions reduced to {len(productions)}  ({time.time() - timestart:.3}s )")

        timemid = time.time()
        graphs = [ self._substitute_core(*prod) for prod in productions ]
        graphs = [ g for g in graphs if g ]
        logger.debug(f"expand neighbors: generating graphs  ({time.time() - timemid:.3}s )")
        return graphs


    def vectorize(self, graph):
        return dok_matrix(self.vectorizer.raw(graph))

    def get_productions(self, pdict, graph,grid):
        grlen = len(graph) + graph.size()
        vec = self.vectorize(graph)
        current_cips = self._get_cips(graph)

        for current_cip in current_cips:
            if productions := [(graph, vec, current_cip,concip) for concip in self._get_congruent_cips(current_cip)
                    if len(concip.core_nodes) + grlen - len(current_cip.core_nodes) <= self.genmaxsize]:
                pdict[self.mkkey(grid, current_cip.interface_hash)]+= productions

    def mkkey(self, grid, ihash):
        if self.selelector == 'pop':
            return 0
        elif self.selelector =='graph':
            return grid
        return f"{grid}_{ihash}"

    def selectbest(self,stuff):
        '''
            stuff is a list:[startgraph,startgraph_vec, cip_con, cip_congru]
            return: selctor_k best productions in the list
        '''
        # score productions:
        #from graken.main import dumpfile
        #myvectors = np.empty((len(stuff),stuff[0][1].shape[1]))
        #for i,(gra,vec,cur,con) in enumerate(stuff):
        #    myvectors[i] = vec  - cur.core_vec + con.core_vec
        myvectors = [vec - cur.core_vec + con.core_vec for gra,vec,cur,con in stuff]
        #myvectors = [vec + ( con.core_vec - cur.core_vec  ) for gra,vec,cur,con in stuff]
        myvectors = vstack(myvectors)
        myvectors = self.vectorizer.normalize(myvectors)
        #print("lenv:", len(myvectors.indices))
        #scores = np.dot(myvectors, self.target)
        scores =myvectors.dot(self.target)


        # the datatype here changes randomly? wtf? TODO TODO
        if type(scores) == np.ndarray:
            scores = scores.ravel()
        else:
            scores  = scores.todenze().ravel()
        # if not 'ravel' in scores.__dict__:
        #     scores = scores.todense().A1
        # else:
        #     scores = scores.ravel()

        goodindex = np.argsort(scores)
        #print("gind", goodindex)
        goodindex = goodindex[-self.selelector_k:]
        self.predscores+=[scores[i] for i in goodindex]
        return [ (stuff[i][0],stuff[i][2], stuff[i][3]) for i in goodindex]


class edengrammar(gradigrammar):
    def make_core_vector(self, core, graph, node_vectors):
        c_set = set(core.nodes())
        # TODO check if edge check is necessary...
        core_ids = [i for i,n in enumerate(graph.nodes()) if n in c_set and not graph.nodes[n].get('edge',False)  ]
        return dok_matrix(node_vectors[core_ids,:].sum(axis=0))

    def vertex_vectorizer(self,exgraph):
        # TODO vectex vectorization should be part of the vectorizer class, hoever this abstraction needs to also be done in graphlearn..
        return  eg.vertex_vectorize([exgraph], d = self.eden_d, r=self.eden_r, normalization = False,nbits= 16,inner_normalization= False)[0]

    def vectorize(self,graph):
        return eg.vectorize([graph], d = self.eden_d, r=self.eden_r, normalization = False,nbits=16,inner_normalization= False)[0]


from graphlearn import local_substitution_graph_grammar as grammarcore

class sizecutgrammar(grammarcore.LocalSubstitutionGraphGrammar):

    def __init__(self,graphsizelimiter, **kwargs):
        super(sizecutgrammar,self).__init__(**kwargs)
        self.graphsizelimiter = graphsizelimiter
        self.isfit= False


    def fit(self,graphs,landmarks):
        self.genmaxsize = self.graphsizelimiter(np.array([len(g) + g.size() for g in landmarks]))
        super(sizecutgrammar,self).fit(graphs)
        self.isfit = True


    def _get_cores(self, graph):
        cores = super(sizecutgrammar,self)._get_cores(graph)
        if self.isfit:
           cores += [core for core in lsggcip.get_cores_closeloop(graph, self.radii) if core]
        return cores

    def expand_neighbors(self, graphs):
        timemid = time.time()
        r =  [ n for graph in graphs for  n in self.neighbors(graph)]
        logger.debug(f"expand neighbors: generating graphs  ({time.time() - timemid:.3}s )")
        return r
    def neighbors(self, graph):
        grsize = len(graph) + graph.size()
        for cip in self._get_cips(graph):
            for congruent_cip in self._get_congruent_cips(cip):
                if grsize - len(cip.core_nodes) + len(congruent_cip.core_nodes) <= self.genmaxsize:
                    graph_ = self._substitute_core(graph, cip, congruent_cip)
                    if graph_ is not None:
                        yield graph_
