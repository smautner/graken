from ego.setup import *
from ego.vectorize import vectorize as ego_vectorize
import time
import random
import numpy as np
import scipy as sp
import networkx as nx
from lmz import *


class egoestimator: 
    def __init__(self, graph, neighs): 
        self.model = oracle_setup(neighs)
    def get_k_best(self,graphs, num):
        return select_k_best(graphs, self.model, num=num)

def select_k_best(graphs, oracle_func, num=100):
    scores = Map(oragle_func, graphs)
    noisescores, scores = Transpose(scores) 
    sortid  = np.argsort(noisescores)[::-1]
    return [graphs[x] for x in sortid[:num]] , max(scores) > .98

def ego_oracle_setup(target_graphs, df=None, preproc=None):
    target_graph_vecs = ego_vectorize(target_graphs, decomposition_funcs=df, preprocessors=preproc)
    target_norm = np.mean(  [tgv.dot(tgv.T).A[0,0] for tgv in target_graph_vecs])

    def oracle_func(g):
        g_vec = ego_vectorize([g], decomposition_funcs=df, preprocessors=preproc)
        g_norm =  g_vec.dot(g_vec.T).A[0,0]
        scale_factor = np.sqrt(g_norm * target_norm)
        score = np.mean(target_graph_vecs.dot(g_vec.T))/scale_factor
        return score
    return oracle_func


def oracle_setup(target_graphs, random_noise=0.05, include_structural_similarity=True):
    df = do_decompose(decompose_cycles_and_non_cycles, decompose_neighborhood(radius=2), do_decompose(decompose_cycles, compose_function=decompose_edge_join))
    preproc = preprocess_abstract_label(node_label='C', edge_label='1')
    structural_oracle_func = ego_oracle_setup(target_graphs, df, preproc)

    df = do_decompose(decompose_nodes_and_edges)
    compositional_oracle_func = ego_oracle_setup(target_graphs, df, None)

    df = do_decompose(decompose_path(length=2), decompose_neighborhood, do_decompose(decompose_cycles, compose_function=decompose_edge_join))
    comp_and_struct_oracle_func = ego_oracle_setup(target_graphs, df, None)
    
    target_size = np.mean(map(len,target_graphs))

    def oracle_func(g, explain=False):
        g_size = len(g)
        size_similarity = max(0, 1 - abs(g_size - target_size)/float(target_size))
        structural_similarity = structural_oracle_func(g)
        composition_similarity = compositional_oracle_func(g)
        comp_and_struct_similarity = comp_and_struct_oracle_func(g)
        score = sp.stats.gmean([size_similarity,structural_similarity,composition_similarity,comp_and_struct_similarity])
        noise = random.random()*random_noise
        tot_score = score + noise 
        if explain:
            explanation = {'1) size':size_similarity, '2) struct':structural_similarity, '3) comp':composition_similarity, '4) comp&struct':comp_and_struct_similarity}
            return score, explanation
        else:
            return tot_score, score

    return oracle_func

