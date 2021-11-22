from lmz import *
import numpy as np
import basics as ba
import random
from sklearn.decomposition import KernelPCA, PCA, TruncatedSVD
import eden.graph as eg
import bayesconstruct

'''
bayesian optimization is problematic because only 1 guess at a time ..
https://stackabuse.com/kernel-density-estimation-in-python-using-scikit-learn/ here i guess...
also i should write an optimizer and add it to the ubergauss

BAYES DOCS ARE:
the idea is that we want to optimize a graph while using just few tries
- load some graphs and pca
- define an oracle
- ask a few graphs
- use scores and pca projection to do bayesian opt -> pca point -> inquire -> repeat
'''


def poptimizer(optimizer):
    print([list(row) for row in optimizer._space._params])
    print(list(optimizer._space._target))

graphs = ba.loadfile('../graken/chemtasks/119')
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
    graph = bayesconstruct.construct(targetV,graphs,graphsV, vectorizer, nn)
    return oracle(graph)


from hyperopt import hp
space = hp.choice('a',
    [
        ('case 1', 1 + hp.lognormal('c1', 0, 1)),
        ('case 2', hp.uniform('c2', -10, 10))
    ])


