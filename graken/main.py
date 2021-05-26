import json
import dill

dumpfile = lambda thing, filename: dill.dump(thing, open(filename, "wb"))
jdumpfile = lambda thing, filename:  open(filename,'w').write(json.dumps(thing))
loadfile = lambda filename: dill.load(open(filename, "rb"))
jloadfile = lambda filename:  json.loads(open(filename,'r').read())


import logging, sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

import random
import time
from graken.ml import cost_estimator
import random 
from graken.ml import vector
from graken.ml import neighbors
from graken.pareto import grammar
import dirtyopts as opts
from graken.pareto import pareto
import structout as so
import eden.graph as eg

doc='''
# LOAD 
--i str tasks/task_0
--n_train int -1
--taskid int 0
--shuffle int 0     -> random seed for shuffling

# INIT
--v_radius int 2  
--v_distance int 1  
--v_nonormalize bool False
--n_landmarks int 10
--n_neighbors int 100

# GRAMMAR
--maxcoresize int 2 
--contextsize int 1
--filter_min_cip int 2
--cipselector int 2           0 -> k is on populationlevel , 1 -> k is io graphlevel ,2 -> k is on ciplevel
--cipselector_k int 1
--size_limiter eval lambda x:x.max()+(int(x.std()))

# OPTIMIZER 
--removedups bool False
--n_iter int 10
--pareto str default          
        ['default', 'random', 'greedy', 'paretogreed', 'pareto_only', 'all']
--keepgraphs int 30

--out str res/out.txt
'''

if __name__ == "__main__":
    ###################
    # 1. LOAD
    ###################
    args = opts.parse(doc)
    logging.debug(args.__dict__)

    graphs = loadfile(args.i)
    random.seed(args.shuffle)
    random.shuffle(graphs)

    domain = graphs[:args.n_train]
    target = graphs[-(args.taskid+1)]

    assert (args.n_train+args.taskid) < len(graphs) , f"{args.n_train} {args.taskid} {len(graphs)}"
    logging.debug(f"loading done")


    #################
    # INIT
    ##################
    # build a vectorizer for everything
    vectorizer = vector.Vectorizer(args.v_radius, args.v_distance, not args.v_nonormalize)
    estiandinitvec = eg.Vectorizer(r=3, d=3)

    # find neighbors/landmarks
    landmark_graphs, desired_distances, ranked_graphs = neighbors.initialize(
                                                            n_landmarks=args.n_landmarks, 
                                                            n_neighbors=args.n_neighbors, 
                                                            vectorizer=estiandinitvec,
                                                            graphs=domain,
                                                            target=target)




    
    # fit grammar
    t = time.time()
    mygrammar = grammar.gradigrammar(radii=list(range(args.maxcoresize+1)),
                               thickness=args.contextsize,
                               graphsizelimiter= args.size_limiter,
                               vectorizer=vectorizer,
                               selector=args.cipselector,
                                selektor_k= args.cipselector_k,
                               filter_min_cip= args.filter_min_cip,
                               nodelevel_radius_and_thickness=True)
    mygrammar.fit(ranked_graphs, vectorizer.transform([target]))
    logging.debug(f"fit grammar done {time.time()-t:.2}s")



    # build estimator
    if args.pareto != 'greedy':
        #time.sleep(random.randint(0,4)*15)
        t = time.time()
        #multiopesti = cost_estimator.DistRankSizeCostEstimator(vectorizer=vectorizer)
        multiopesti = cost_estimator.DistRankSizeCostEstimator(vectorizer=estiandinitvec)
        multiopesti.fit(desired_distances, landmark_graphs, ranked_graphs)
        logging.debug(f"fit multiopesti done {time.time()-t:.2}s")
    else:
        multiopesti = None


    #############
    # Main loop
    ##############
    optimizer = pareto.LocalLandmarksDistanceOptimizer(
                n_iter=args.n_iter,
                keepgraphs=args.keepgraphs,
                filter = args.pareto,
                estimator = multiopesti,
                vectorizer = vectorizer,
                greedyvec = estiandinitvec,
                remove_duplicates = args.removedups,
                grammar = mygrammar )

    ######
    #  DONE
    ###### 
    result = optimizer.optimize(landmark_graphs, estiandinitvec.transform([target]))
    so.gprint(target)
    dumpfile(result, args.out)
