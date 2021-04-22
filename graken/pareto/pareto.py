import logging
import time
import random
import numpy as np
import structout as so
from sklearn.metrics.pairwise import euclidean_distances
from graken.pareto import pareto_funcs

def calc_average(l):
    """
    Small function to mitigate possibility
    of an empty list of average productions.
    """
    if len(l) == 0:
        return 0
    return sum(l)/len(l)


class LocalLandmarksDistanceOptimizer(object):
    """LocalLandmarksDistanceOptimizer."""

    def __init__( self, n_iter=5,
            keepgraphs=30,
            filter = 'default',
            estimator = None,
            vectorizer = None,
            remove_duplicates = True,
            grammar = None):
            
        self.grammar = grammar
        self.estimator = estimator
        self.filter=filter
        self.keepgraphs = keepgraphs
        self.n_iter = n_iter
        self.vectorizer = vectorizer
        self.rmdup = remove_duplicates
        
        self.avarages=[]
        self.seen_graphs = []

        #######################
        #  OPTIMIZE
        #####################

    def optimize(self, start_graph_list=False, target_graph_vector=None):
        starttime = time.time()
        for i in range(self.n_iter):
            logging.debug("++++++++  START OPTIMIZATION STEP %d +++++++" % i)
            '''
                filter:calculate costs and check if we are done
                expand
                duplicate rm
            '''
            graphs, done = self.filter(graphs)
            if done:
                return True, i, time.time() - starttime, np.mean(self.averages)

            num_graphs = len(graphs)
            graphs = self.grammar.expand_neighbors(graphs)
            avg_productions = len(graphs) / num_graphs
            self.averages.append(avg_productions)
            logging.log(10, f"Average productions per graph: {avg_productions}")
            if self.rmdup:
                graphs = self.duplicate_rm(graphs)


        
        return False, -1 , time.time() - starttime, np.mean(self.averages)
        

    ##############
    # FILTERING 
    #################
    def filter(self,graphs):
        '''
        calc costs
        are we done?
        do filtering
        '''
        timenow = time.time()
        in_count = len(graphs)


        if self.filter in ['greed', 'random'] or len(graphs) < self.keepgraphs:
            distances = euclidean_distances(self.target, self.vectorizer.transform(graphs))
            done = min(distances) < 0.0001
            if self.filter == 'greed':
                ranked_distances = np.argsort(distances)[:self.keepgraphs]
                graphs = [graphs[i] for i in ranked_distances]
            if self.filter == 'random':
                graphs = random.sample(graphs, self.keepgraphs)
        else:
            costs = self.estimator.decision_function(graphs)
            graphs, costs = pareto_funcs._pareto_set(graphs, costs, return_costs=True)
            costs = self.add_rank(costs) # costs = distances rank size
            done =  min(costs[:,0]) < 0.00001

            if self.filter in ['all','pareto_only']:
                random.shuffle(graphs)
                if self.filter == "pareto_only":
                    graphs = graphs[:self.keepgraphs]
                    
            if self.filter == 'paretogreed':
                graphs = [graphs[x] for x in np.argsort(costs[:,3])[:self.keepgraphs]]
                
            if self.filter == 'defaut':
                graphs = self._default_selektor(costs, graphs)

        logging.log(10, f"cost_filter: got {in_count} graphs, reduced to {len(graphs)} (%.2fs)"%(time.time()-timenow))

        # print
        #g = [graphs[e] for e in np.argmin(costs, axis=0)]
        #logging.debug(so.graph.make_picture(g, edgelabel='label', size=10))
        #logging.log(10, [x.number_of_nodes() for x in g])

        return graphs, done

    def _default_selektor(self, costs, graphs):
        costs_ranked = np.argsort(costs, axis=0)[:int(self.keepgraphs / 6), [0, 1, 3]]  # 2 is size and that is bad, right?
        want, counts = np.unique(costs_ranked, return_counts=True)
        res = [graphs[idd] for idd, count in zip(want, counts) if count > 0]
        dontwant = [i for i in range(len(graphs)) if i not in want]
        restgraphs = [graphs[i] for i in dontwant]
        restcosts = costs[dontwant][:, [0, 1, 2]]
        paretographs = pareto_funcs._pareto_set(restgraphs, restcosts)
        graphs = res + random.sample(paretographs, int(self.keepgraphs / 2))
        return graphs

        

    def _add_rank(self,costs):
        '''so we want another column indicating the average rank
            the problem is that there are many graphs with the same size ...
        '''
        sort = np.argsort(costs, axis=0)
        ranks = np.argsort(costs, axis=0)

        # we have the ranks, but the ranks for size are shit
        translate = {}
        for i,ind in enumerate(sort[:,2]):
            translate.setdefault(costs[ind,2],i)
        ranks[:,2] = [ translate[v] for v in costs[:,2] ]

        costs = np.hstack((costs, np.sum(ranks, axis=1).reshape(-1, 1)))
        return costs



    ################
    #  DUPLICATE RM 
    #################

    def duplicate_rm(self, graphs):
        timenow = time.time()
        count = len(graphs)
        graphs = list(self._duplicate_rm(graphs))
        logging.debug("duplicate_rm: %d -> %d graphs (%.2fs)" % (count, len(graphs), time.time() - timenow))
        return graphs

    def _duplicate_rm(self, graphs):
        hashes = self.hash_vectorizer.vectorize(graphs)
        for i, (ha, gr) in enumerate(zip(hashes, graphs)):
            if ha not in self.seen_graphs:
                self.seen_graphs[ha] = i
                yield gr


