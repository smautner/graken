from lmz import *
import numpy as np
from bayes_opt import BayesianOptimization
import basics as ba
import random
from sklearn.decomposition import KernelPCA, PCA, TruncatedSVD
import eden.graph as eg
import bayesconstruct

'''
the idea is that we want to optimize a graph while using just few tries
- load some graphs and pca
- define an oracle
- ask a few graphs
- use scores and pca projection to do bayesian opt -> pca point -> inquire -> repeat
'''


def poptimizer(optimizer):
    print([list(row) for row in optimizer._space._params])
    print(list(optimizer._space._target))

graphs = ba.loadfile('chemtasks/119')
random.shuffle(graphs)
graphs = graphs[:100]
#pca = KernelPCA(2, fit_inverse_transform = True)# might want to do normal PCA to 100 and then kernel..
#pca = PCA(2)# might want to do normal PCA to 100 and then kernel..
pca = TruncatedSVD(3)# might want to do normal PCA to 100 and then kernel..
vectorizer= eg.Vectorizer(r=2, d=1)
graphsV = vectorizer.transform(graphs)
zz = pca.fit_transform(graphsV)
from sklearn.neighbors import NearestNeighbors as NN
nn = NN(n_neighbors = 50 , metric = 'cosine').fit(graphsV)



def oracle(graph):
    return -abs( sum([label=='C' for n,label in graph.nodes(data='label')])  - 10)


def score_pcacoordinate(x=None,y=None,z=None):
    pcacoordinate = np.array([x,y,z]).reshape(1,-1)
    targetV = pca.inverse_transform(pcacoordinate)
    graph = bayeshelper.construct(targetV,graphs,graphsV, vectorizer, nn)
    return oracle(graph)


# Bounded region of parameter space
pbounds = {'x': (zz.min(), zz.max()),
        'y': (zz.min(), zz.max()),
        'z': (zz.min(), zz.max())}

optimizer = BayesianOptimization(
    f=score_pcacoordinate,
    pbounds=pbounds, ## musst be defined !!!!!!!!!
    random_state=1,
)


# TODO loop over this and refit the PCA projection
# this is the default optimizer, but we want to be awesome and initialize with many guesses
# prime the optimizer with parallel executed guesses
'''
INITIALIZING WHITH PARALLEL PROBING OF RANDOM POINTS >>> THEN MAXIMIZING
'''
pts = [optimizer._space.random_sample() for i in range(10)]
def f(args):
    return score_pcacoordinate(args[0],args[1],args[2])
targets = ba.mpmap(f,pts, poolsize =10)
for poi, tgt in zip(pts, targets):
    optimizer.register(params=poi, target=tgt)
poptimizer(optimizer)
optimizer.maximize( init_points=0, n_iter=50)

poptimizer(optimizer)
