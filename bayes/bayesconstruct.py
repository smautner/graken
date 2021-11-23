from graken.ml import vector
from graken.pareto import grammar, pareto

def construct(target,graphs,graphsV, vectorizer,nn, n_iter = 10):

    n = nn.kneighbors(target)[1][0]
    #print(f"{n=} {len(graphs)=}")
    lando = [graphs[i] for i in n[:10]]
    ranke = [graphs[i] for i in n]

    mygrammar = grammar.edengrammar(radii = [0,1,2],
                                    vectorizer = vectorizer,
                                    thickness=1,
                                    graphsizelimiter= lambda x: x.mean()*1.25,
                                    selector = 1,
                                    selektor_k =10,
                                    nodelevel_radius_and_thickness=True)
    mygrammar.eden_r  = vectorizer.edenvec.r
    mygrammar.eden_d  = vectorizer.edenvec.d
    mygrammar.fit(ranke, lando,target)

    optimizer = pareto.LocalLandmarksDistanceOptimizer(
                n_iter=n_iter,
                targetgraph = target,
                keepgraphs=15,
                filter = 'greedy',
                estimator = None,
                vectorizer = vectorizer,
                remove_duplicates = True,
                grammar = mygrammar )
    optimizer.optimize(lando,target)

    return optimizer.best_graph[1]

