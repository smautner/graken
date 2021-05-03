import json
import dill

dumpfile = lambda thing, filename: dill.dump(thing, open(filename, "wb"))
jdumpfile = lambda thing, filename:  open(filename,'w').write(json.dumps(thing))
loadfile = lambda filename: dill.load(open(filename, "rb"))
jloadfile = lambda filename:  json.loads(open(filename,'r').read())


import logging, sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

from graken.ml import cost_estimator
from graken.ml import vector
from graken.ml import neighbors
from graken.pareto import grammar
import dirtyopts as opts
from graken.pareto import pareto


doc='''
# LOAD 
--i str default:.taskfile
--n_train int default:-1
--taskid int default:0

# INIT
--v_radius int default:2  
--v_distance int default:1  
--v_nonormalize
--n_landmarks int default:10
--n_neighbors int default:100

# GRAMMAR
--maxcoresize int default:2 
--contextsize int default:1
--filter_min_cip int default:2
--cipselector int default:2           0 -> k is on populationlevel , 1 -> k is io graphlevel ,2 -> k is on ciplevel
--cipselector_k int default:1
--size_limiter eval default:lambda x: x.max()+(int(x.std()))

# OPTIMIZER 
--removedups 
--n_iter int default:10
--pareto str default:default          ['default', 'random', 'greedy', 'pareto_only', 'all']
--keepgraphs int default:30

--out str default:res/out.txt
'''

if __name__ == "__main__":
    ###################
    # 1. LOAD
    ###################
    args = opts.parse(doc)
    logging.debug(args.__dict__)
    graphs = loadfile(args.i)[0]

    domain = graphs[:args.n_train]
    target = graphs[-(args.taskid+1)]
    assert (args.n_train+args.taskid) < len(graphs)


    #################
    # INIT
    ##################
    # build a vectorizer for everything
    vectorizer = vector.Vectorizer(args.v_radius, args.v_distance, not args.v_nonormalize)


    # find neighbors/landmarks
    landmark_graphs, desired_distances, ranked_graphs = neighbors.initialize(
                                                            n_landmarks=args.n_landmarks, 
                                                            n_neighbors=args.n_neighbors, 
                                                            vectorizer=vectorizer,
                                                            graphs=domain,
                                                            target=target)





    mygrammar = grammar.gradigrammar(radii=list(range(args.maxcoresize+1)),
                               thickness=args.contextsize,
                               graphsizelimiter= args.size_limiter,
                               vectorizer=vectorizer,
                               selector=args.cipselector,
                                selektor_k= args.cipselector_k,
                               filter_min_cip= args.filter_min_cip,
                               nodelevel_radius_and_thickness=True)
    mygrammar.fit(ranked_graphs, vectorizer.transform([target]))


    # build estimator thing
    if args.pareto != 'greedy':
        multiopesti = cost_estimator.DistRankSizeCostEstimator(vectorizer=vectorizer).fit(desired_distances, landmark_graphs, ranked_graphs)
    else:
        multiopesti = None



    #############
    # Main loop
    ##############
    # produce children
    # kill the unfit (while checking if we are done)

    optimizer = pareto.LocalLandmarksDistanceOptimizer(
                n_iter=args.n_iter,
                keepgraphs=args.keepgraphs,
                filter = args.pareto,
                estimator = multiopesti,
                vectorizer = vectorizer,
                remove_duplicates = args.removedups,
                grammar = mygrammar )


    result = optimizer.optimize(landmark_graphs, vectorizer.transform([target]))
    dumpfile(result, args.out)
