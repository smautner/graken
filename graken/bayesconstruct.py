


from graken.ml import vector
from graken.pareto import grammar, pareto

def construct(target,graphs,graphsV, vectorizer,nn):

    n = nn.kneighbors(target)[1][0]
    lando = [graphs[i] for i in n[:30]]
    ranke = [graphs[i] for i in n]

    grammarvec = vector.Vectorizer(radius = vectorizer.r, distance =vectorizer.d)
    mygrammar = grammar.edengrammar(radii = [0,1,2],
                                    vectorizer = grammarvec,
                                    thickness=1,
                                    graphsizelimiter= lambda x: x.mean()*1.25,
                                    selector = 1,
                                    selektor_k =10,
                                    nodelevel_radius_and_thickness=True)
    mygrammar.eden_r  = vectorizer.r
    mygrammar.eden_d  = vectorizer.d
    mygrammar.fit(ranke, lando,target)

    optimizer = pareto.LocalLandmarksDistanceOptimizer(
                n_iter=5,
                targetgraph = target,
                keepgraphs=30,
                filter = 'greedy',
                estimator = None,
                vectorizer = grammarvec,
                remove_duplicates = True,
                grammar = mygrammar )
    optimizer.optimize(lando,target)

    return optimizer.best_graph[1]

