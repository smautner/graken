import logging
logger=logging.getLogger(__name__)
from lmz import *
import basics as ba
import time
import random
import numpy as np
import structout as so
from sklearn.metrics.pairwise import euclidean_distances
from graken.pareto import pareto_funcs
from graken.pareto import editdistance
from scipy import sparse



import dirtyopts as opts
doc = '--drawerror bool False'
moduleargs = opts.parse(doc)
if moduleargs.drawerror:
    import matplotlib
    matplotlib.use('module://matplotlib-sixel')
    import matplotlib.pyplot as plt


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
            targetgraph = None,
            grammar = None):
        self.targetgraph = targetgraph
        self.grammar = grammar
        self.estimator = estimator
        self.paretofilter=filter
        self.keepgraphs = keepgraphs
        self.n_iter = n_iter
        self.vectorizer = vectorizer
        self.rmdup = remove_duplicates

        self.seen_graphs = {}
        self.best_graph = (9999,None)
        self.expand_queue = [[],[]]

        #######################
        #  OPTIMIZE
        #####################

    def optimize(self, graphs=False, target_graph_vector=None):
        self.target = target_graph_vector
        starttime = time.time()
        done = False
        for i in range(self.n_iter):
            logger.debug("++++++++  START OPTIMIZATION STEP %d +++++++" % i)
            '''
                filter:calculate costs and check if we are done
                expand
                duplicate rm
            '''
            graphs, done = self.filter(graphs)
            if done:
                break

            graphs = self.grammar.expand_neighbors(graphs)
            if moduleargs.drawerror:
                self.plotpredictedscores(graphs) # TODO REMOVE :)
            if self.rmdup:
                graphs = self.duplicate_rm(graphs)

        logger.debug('\n'+so.graph.make_picture(random.sample(graphs,3), edgelabel='label', size=10))
        logger.debug(f"success: {done}")
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
        if self.paretofilter == 'random' or len(graphs) <= self.keepgraphs:
            vectors = self.vectorizer.transform(graphs)
            distances = euclidean_distances(self.target, vectors)
            done = distances.min() < 0.0001
            if self.paretofilter == 'random':
                numsample = min(self.keepgraphs, len(graphs))
                graphs = random.sample(graphs, numsample)
        if self.paretofilter =='greedy':
            '''
            so we wan to keep the best X*100 graph with scores
            and choose the best
            '''
            vectors = self.vectorizer.transform(graphs)
            distances = euclidean_distances(self.target, vectors)
            done = distances.min() < 0.0001
            # keep the best graph:
            if np.min(distances) < self.best_graph[0]:
                self.best_graph = np.min(distances), graphs[np.argmin(distances)]


            # keep expand-list...
            if np.any(self.expand_queue[0]):
                #print("NEW AND OLD:")
                #so.lprint(np.sort(distances.ravel())[:100])
                #so.lprint(np.sort(self.expand_queue[0].ravel())[:100])
                distances = np.hstack((distances, self.expand_queue[0]))
            allgraphs = graphs+self.expand_queue[1]

            ranked_distances = np.argsort(distances)[0]
            graphs = [allgraphs[i] for i in ranked_distances[:self.keepgraphs]]
            self.expand_queue[1] = [allgraphs[i] for i in ranked_distances[self.keepgraphs:200]]
            self.expand_queue[0] = distances[:,ranked_distances[self.keepgraphs:200]]

        elif self.paretofilter == 'default':
                graphs,done, frontsize = self._default_selector(graphs)
        elif self.paretofilter in ['pareto_only','paretogreed','all']:
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

        elif self.paretofilter == 'geometric':
            z =  self.estimator.get_k_best(graphs, self.keepgraphs)
            graphs, done = z

        logger.log(10, f"cost_filter: got {in_count} graphs (pareto:{frontsize}), reduced to {len(graphs)} (%.2fs)"%(time.time()-timenow))

        # self.check_true_distance(graphs)

        # print
        return graphs, done




    def check_true_distance(self, graphs):
        if self.targetgraph:
            now = time.time()
            dists = ba.mpmap(editdistance.dist, [(g,self.targetgraph) for g in graphs],poolsize = 30)

            z = np.argsort(dists)
            dists.sort()
            logger.debug( f'TRUE DISTS: { dists[:5] }  ({time.time()- now}s)' )
            so.gprint([graphs[i] for i in z[:5]])

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

    def plotpredictedscores(self, graphs):
        if 'predscores' in self.grammar.__dict__:

            # real scores
            vectors = self.vectorizer.transform(graphs)
            realscores = vectors.dot(self.target.T).todense().A1
            grammarscores= np.array(self.grammar.predscores)


            sig = np.std(realscores)
            print(f"{len(grammarscores)}  {len(realscores)}")
            zerror  = [ abs(a-b)/sig for a,b in zip(grammarscores,realscores)]

            print("average z-error:", np.mean(zerror))
            #print(list(realscores))
            #print(list(grammarscores))
            # asd = Zip(realscores, grammarscores)
            # asd.sort()
            # r,g = Transpose(asd)
            # plt.plot(r)
            # plt.plot(g)
            plt.scatter(realscores, grammarscores)
            plt.plot([0,1],[0,1], ':', c='gray', alpha=.4)
            plt.axis('square')
            plt.grid()
            c = np.corrcoef(grammarscores , realscores)[0,1]
            plt.title(f"pearson corr {c:.3}")
            plt.show()
            plt.close()
        else:
            print("there is no predscore in the grammar")



