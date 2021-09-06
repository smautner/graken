from lmz import *
import time
from ego import real_vectorize as rv
from ego.decomposition.paired_neighborhoods import decompose_paired_neighborhoods
from sklearn.preprocessing import normalize as sknormalize
import numpy as np
from scipy.sparse import csr_matrix

import logging
logger=logging.getLogger(__name__)
from eden.graph import Vectorizer as edenvec

class Vectorizer():

    def __init__(self, radius=3, distance=3, normalize=True, eden = True):
        self.decompose = lambda gr: decompose_paired_neighborhoods(gr, max_radius=radius, max_distance=distance)
        self.edenvec = edenvec(r=radius,d=distance, normalization = False, inner_normalization= False)
        self.norm = normalize
        self.hasher = edenvec(r=2,d=1, normalization = False, inner_normalization= False)
        self.eden=eden

    


    def vectorize(self, graph):
        #  vextorize dioes as grammar does and then sum axis 0 
        if not self.eden:
            encoding, node_ids = rv.node_encode_graph(graph, rv.tz.compose(
                    rv.get_subgraphs_from_graph_component, self.decompose, rv.convert),
                            bitmask= 2**16-1)
            data_matrix = rv._to_sparse_matrix(encoding, node_ids, 2**16+1)
            r = data_matrix.sum(axis=0)
            return r 
        return self.edenvec.transform([graph])[0] 

    def transform(self,graphs):
        # should i use this instead? from ego.vectorize import vectorize
        r = np.vstack(Map(self.vectorize,graphs)) if not self.eden else self.edenvec.transform(graphs)
        re = self.normalize(r)
        re = csr_matrix(re)
        return re

    def normalize(self, ndarr):
        if self.norm:
            ndarr = sknormalize(ndarr, axis = 1 )
        return ndarr


   

    ##############
    # hashing 
    ##############
    def hashvec(self,graphs):
        csrs =  self.hasher.transform(graphs)
        return Map(lambda row: hash(tuple(row.indices)+tuple(row.data)), csrs)

    def duplicate_rm(self, graphs, seen_graphs):
        timenow = time.time()
        count = len(graphs)
        graphs = list(self._duplicate_rm(graphs,seen_graphs))
        logger.debug("duplicate_rm: %d -> %d graphs (%.2fs)" % (count, len(graphs), time.time() - timenow))
        return graphs

    def _duplicate_rm(self, graphs, seen_graphs):
        hashes = self.hashvec(graphs)
        for i, (ha, gr) in enumerate(zip(hashes, graphs)):
            if ha not in seen_graphs:
                seen_graphs[ha] = i
                yield gr



'''
class hashvec(object):

    def __init__(self, vectorizer, multiproc = 1):
        self.vectorizer = vectorizer
        self.multiproc = multiproc


    def grouper(self, n, iterable):
        it = iter(iterable)
        while True:
            chunk = tuple(itertools.islice(it, n))
            if not chunk:
                return
            yield chunk

    def vectorize_chunk(self,chunk):
        feature_dicts = [ self.vectorizer._transform(graph) for graph in chunk]
        def hashor(fd):
            k= sorted(fd.keys())
            v = [fd[kk] for kk in k]
            return hash(tuple(k+v))
        hashed_features = [ hashor(fd) for fd in feature_dicts]
        return hashed_features
    
    def vectorize_chunk_glhash(self,chunk):
        return [glcip.graph_hash(eden.graph._edge_to_vertex_transform(g),2**20-1,node_name_label=lambda id,node:hash(node['label'])) for g in chunk]

    def vectorize_multiproc(self, graphs):
        with multiprocessing.Pool(self.multiproc) as p:
            res = (p.map(self.vectorize_chunk, self.grouper(1000,graphs)))
        return itertools.chain.from_iterable(res)

    def vectorize(self,graphs):
        if self.multiproc>1:
            return self.vectorize_multiproc(graphs)
        else:
            return self.vectorize_chunk(graphs)
'''
