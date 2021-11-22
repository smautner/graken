from lmz import Map,Zip,Filter,Grouper,Range,Transpose
import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
import basics as ba
import random
import eden.graph as eg
import bayesconstruct
import dimensionmap
from graken.ml import vector

import matplotlib
matplotlib.use('module://matplotlib-sixel')
from matplotlib import pyplot as plt
'''
the idea is that we want to optimize a graph while using just few tries
- load some graphs and pca
- define an oracle
- ask a few graphs
- use scores and pca projection to do bayesian opt -> pca point -> inquire -> repeat
'''


def print_optimizer(optimizer):
    print([list(row) for row in optimizer._space._params])
    print(list(optimizer._space._target))

graphs = ba.loadfile('../graken/chemtasks/119')
random.shuffle(graphs)
graphs = graphs[:1000]
def oracle(graph):
    return abs( sum([label=='C' for n,label in graph.nodes(data='label')])  - 10)

dimension_converter = dimensionmap.BestTransformer()
vectorizer = vector.Vectorizer(radius=2,distance=1, normalize=True, eden=True)
graphsV = vectorizer.transform(graphs).todense()
graphsS = np.array(Map(oracle,graphs))

#mea = np.mean(graphsS)
#graphsS = [ 1 if s > mea else 0 for s in graphsS  ]

lowdim = dimension_converter.fit_transform(graphsV,graphsS )



graphs = graphs[:100]
graphsV = graphsV[:100]
graphsS = graphsS[:100]
lowdim = lowdim[:100]

#plt.scatter(lowdim[:,0], lowdim[:,1], c=graphsS)
#plt.show()
#plt.close()


from sklearn.neighbors import NearestNeighbors as NN
nn = NN(n_neighbors = 25 , metric = 'cosine').fit(graphsV)



def score_pcacoordinate(x=None,y=None,z=None):
    #pcacoordinate = np.array([x,y,z]).reshape(1,-1)
    pcacoordinate = x
    targetV = dimension_converter.inverse_transform(pcacoordinate)
    graph = bayesconstruct.construct(targetV,graphs,graphsV, vectorizer, nn, n_iter =15)
    return graph

import ubergauss.optimization.blackboxBORE as bre
import ubergauss.tools as ts
opti = bre.BAY(f = score_pcacoordinate,space = [ (min(lowdim[:,i]),max(lowdim[:,i])) for i in range(lowdim.shape[1])], n_init = 0)
opti.params = [lowdim[i] for i in Range(lowdim)]
opti.values = list(graphsS)


def plotcoodiff(graphs, pt):
    pt = np.array(pt)
    plt.scatter(pt[:,0],pt[:,1])
    graphsV = vectorizer.transform(graphs).todense()
    ld = dimension_converter.transform(graphsV)
    plt.scatter(ld[:,0],ld[:,1])
    plt.show()
    plt.close()


for i in range(3):
    pt = opti.fit(draw=True, sample = 30)
    newgraphs = ts.xmap(score_pcacoordinate,pt)
    score = Map(oracle, newgraphs)
    plotcoodiff(newgraphs, pt)
    opti.register(pt,score)



'''


# Bounded region of parameter space
pbounds = {'x': (lowdim.min(), lowdim.max()),
        'y': (lowdim.min(), lowdim.max()),
        'z': (lowdim.min(), lowdim.max())}

optimizer = BayesianOptimization(
    f=score_pcacoordinate,
    pbounds=pbounds, ## musst be defined !!!!!!!!!
    random_state=1,
)


# TODO loop over this and refit the PCA projection
# this is the default optimizer, but we want to be awesome and initialize with many guesses
# prime the optimizer with parallel executed guesses
# INITIALIZING WHITH PARALLEL PROBING OF RANDOM POINTS >>> THEN MAXIMIZING
if False:
    # init by probing
    pts = [optimizer._space.random_sample() for i in range(10)]

    def f(args):
        return score_pcacoordinate(args[0],args[1],args[2])

    targets = ba.mpmap(f,pts, poolsize =10)
    for poi, tgt in zip(pts, targets):
        optimizer.register(params=poi, target=tgt)

else:
    for coo,target in zip(lowdim,graphsS):
        optimizer.register(params = coo, target = target)

# this tests te suggest function
if False:
    print_optimizer(optimizer)
    util = UtilityFunction(kind='ucb',
                                   kappa=2.576,
                                   xi=0.0,
                                   kappa_decay=1,
                                   kappa_decay_delay=0)
    print(optimizer.suggest(util))
    print(optimizer.suggest(util))
    print(optimizer.suggest(util))
    print(optimizer.suggest(util))
    print(optimizer.suggest(util))
    print(optimizer.suggest(util))
    print(optimizer.suggest(util))
    print(optimizer.suggest(util))
    print(optimizer.suggest(util))
    exit()


optimizer.maximize( init_points=0, n_iter=4)
print_optimizer(optimizer)
'''

