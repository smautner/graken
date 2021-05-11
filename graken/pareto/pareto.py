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
            greedyvec = None,
            remove_duplicates = True,
            grammar = None):
            
        self.grammar = grammar
        self.estimator = estimator
        self.paretofilter=filter
        self.keepgraphs = keepgraphs
        self.n_iter = n_iter
        self.vectorizer = vectorizer
        self.greedyvectorizer = greedyvec
        self.rmdup = remove_duplicates
        
        self.seen_graphs = {}

        #######################
        #  OPTIMIZE
        #####################

    def optimize(self, graphs=False, target_graph_vector=None):
        self.target = target_graph_vector
        starttime = time.time()
        done = False
        for i in range(self.n_iter):
            logging.debug("++++++++  START OPTIMIZATION STEP %d +++++++" % i)
            '''
                filter:calculate costs and check if we are done
                expand
                duplicate rm
            '''
            graphs, done = self.filter(graphs)
            if done:
                break

            graphs = self.grammar.expand_neighbors(graphs)
            if self.rmdup:
                graphs = self.duplicate_rm(graphs)


        logging.debug('\n'+so.graph.make_picture(random.sample(graphs,3), edgelabel='label', size=10))
        logging.debug(f"success: {done}")
        return done, i , time.time() - starttime
        

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
        frontsize=''
        if self.paretofilter in ['greedy', 'random'] or len(graphs) < self.keepgraphs:
            distances = euclidean_distances(self.target, self.greedyvectorizer.transform(graphs))
            done = distances.min() < 0.0001
            if self.paretofilter == 'greedy':
                ranked_distances = np.argsort(distances)[0]
                graphs = [graphs[i] for i in ranked_distances[:self.keepgraphs]]
            if self.paretofilter == 'random':
                numsample = min(self.keepgraphs, len(graphs))
                graphs = random.sample(graphs, numsample)
        elif self.paretofilter == 'default':
                graphs,done, frontsize = self._default_selector(graphs)
        else:
            costs = self.estimator.decision_function(graphs)
            graphs, costs = pareto_funcs._pareto_set(graphs, costs, return_costs=True)
            costs = self._add_rank(costs) # costs = distances rank size
            frontsize=len(graphs)
            done =  costs[:,0].min() < 0.00001

            if len(graphs) < self.keepgraphs or self.paretofilter  == 'all':
               pass  # noo need for further selection

            elif self.paretofilter == 'pareto_only':
                random.shuffle(graphs)
                graphs = graphs[:self.keepgraphs]
                    
            elif self.paretofilter == 'paretogreed':
                graphs = [graphs[x] for x in np.argsort(costs[:,3])[:self.keepgraphs]]
                

        logging.log(10, f"cost_filter: got {in_count} graphs (pareto:{frontsize}), reduced to {len(graphs)} (%.2fs)"%(time.time()-timenow))

        # print
        return graphs, done

    def _default_selector(self, graphs):
        costs = self.estimator.decision_function(graphs)
        done =  costs[:,0].min() < 0.00001
        costs = self._add_rank(costs) 
        costs_ranked = np.argsort(costs, axis=0)[:int(self.keepgraphs / 6), [0, 1, 3]]  
        want, counts = np.unique(costs_ranked, return_counts=True)
        res = [graphs[idd] for idd, count in zip(want, counts) if count > 0]
        dontwant = [i for i in range(len(graphs)) if i not in want]
        restgraphs = [graphs[i] for i in dontwant]
        restcosts = costs[dontwant][:, [0, 1, 2]]
        paretographs = pareto_funcs._pareto_set(restgraphs, restcosts)
        samplenum = min(len(paretographs), self.keepgraphs - len(res))
        add =  random.sample(paretographs,samplenum)
        graphs = res + add
        return graphs, done, len(paretographs)

        

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
        return self.vectorizer.duplicate_rm(graphs, self.seen_graphs) 

