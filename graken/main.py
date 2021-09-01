import json
import sys # for exit :)
import dill

dumpfile = lambda thing, filename: dill.dump(thing, open(filename, "wb"))
jdumpfile = lambda thing, filename:  open(filename,'w').write(json.dumps(thing))
loadfile = lambda filename: dill.load(open(filename, "rb"))
jloadfile = lambda filename:  json.loads(open(filename,'r').read())


import logging, sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
import random
import time
from graken.ml import cost_estimator, cost_estimator_ego_real as CEER
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
--shuffle int -1     -> random seed for shuffling
--specialset bool False

# INIT
--v_radius int 2
--v_distance int 1
--v_normalize bool True
--n_landmarks int 30    # setting it equal to keepgraps to the first round is about as fast as the others
--n_neighbors int 100

# GRAMMAR
--maxcoresize int 2
--contextsize int 1
--filter_min_cip int 2
--cipselector str cip assert pop graph cip
--cipselector_k int 1
--size_limiter eval lambda x: int(sum(x)/len(x))+(int(x.std()))
--eden_d int 1  # eden r+d are only used when i activate the eden grammar...
--eden_r int 1
--grammarname str eden

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
    starttime = time.time()
    args = opts.parse(doc)
    logging.debug(args.__dict__)

    graphs = loadfile(args.i)
    print(f"########## THERE ARE {len(graphs)} graphs in the file ##########")
    if args.specialset:
        graphs, origs = graphs[:-30], graphs[-30:]
        if args.shuffle != -1:
            random.seed(args.shuffle)
            random.shuffle(graphs)
            random.shuffle(origs)

        assert args.n_train <= len(graphs)
        domain = graphs[:args.n_train]
        target = origs[args.taskid]
    else:
        if args.shuffle != -1:
            random.seed(args.shuffle)
            random.shuffle(graphs)

        if args.n_train < 0:
            args.n_train  = len(graphs) + args.n_train

        domain = graphs[:args.n_train]
        target = graphs[-(args.taskid+1)]




        if args.n_train + args.taskid < len(graphs):  # TODO remove this :)
            args.n_train = len(graphs) - 30

        assert (args.n_train+args.taskid) < len(graphs) , f"{args.n_train} {args.taskid} {len(graphs)}"


    logging.debug(f"loading done")


    #################
    # INIT
    ##################
    # build a vectorizer for everything
    vectorizer = vector.Vectorizer(args.v_radius, args.v_distance, args.v_normalize)
    target_vector = vectorizer.transform([target])


    # find neighbors/landmarks
    landmark_graphs, desired_distances, ranked_graphs = neighbors.initialize(
                                                            n_landmarks=args.n_landmarks,
                                                            n_neighbors=args.n_neighbors,
                                                            vectorizer=vectorizer,
                                                            graphs=domain,
                                                            target=target,
                                                            target_vec = target_vector)





    # fit grammar
    t = time.time()
    if args.grammarname == 'gradi':
        mygrammar = grammar.gradigrammar(radii=list(range(args.maxcoresize+1)),
                                   thickness=args.contextsize,
                                   graphsizelimiter= args.size_limiter,
                                   vectorizer=vectorizer,
                                   selector=args.cipselector,
                                    selektor_k= args.cipselector_k,
                                   filter_min_cip= args.filter_min_cip,
                                   nodelevel_radius_and_thickness=True)
        mygrammar.fit(ranked_graphs,landmark_graphs,target_vector)

    elif args.grammarname == 'eden':
        mygrammar = grammar.edengrammar(radii=list(range(args.maxcoresize+1)),
                                   thickness=args.contextsize,
                                   graphsizelimiter= args.size_limiter,
                                   vectorizer=vectorizer,
                                   selector=args.cipselector,
                                    selektor_k= args.cipselector_k,
                                   filter_min_cip= args.filter_min_cip,
                                   nodelevel_radius_and_thickness=True)
        mygrammar.eden_d = args.eden_d
        mygrammar.eden_r = args.eden_r
        mygrammar.fit(ranked_graphs,landmark_graphs, target_vector)
    else:
        mygrammar = grammar.sizecutgrammar(args.size_limiter,
                                    radii=list(range(args.maxcoresize+1)),
                                   thickness=args.contextsize,
                                   filter_min_cip= args.filter_min_cip,
                                   nodelevel_radius_and_thickness=True)
        mygrammar.fit(ranked_graphs,landmark_graphs)
    logging.debug(f"fit grammar done {time.time()-t:.2}s")


    # build estimator
    t = time.time()
    if args.pareto == "greedy":
        multiopesti  = None
    elif args.pareto == 'geometric':
        multiopesti = CEER.egoestimator(target, landmark_graphs,debug=True) # TODO rm debug :)
    else:
        #time.sleep(random.randint(0,4)*15)
        #multiopesti = cost_estimator.DistRankSizeCostEstimator(vectorizer=vectorizer)
        multiopesti = cost_estimator.DistRankSizeCostEstimator(vectorizer=estiandinitvec)
        multiopesti.fit(desired_distances, landmark_graphs, ranked_graphs)
    logging.debug(f"fit multiopesti done {time.time()-t:.2}s")


    #############
    # Main loop
    ##############
    optimizer = pareto.LocalLandmarksDistanceOptimizer(
                n_iter=args.n_iter,
                targetgraph = target if args.specialset else False,
                keepgraphs=args.keepgraphs,
                filter = args.pareto,
                estimator = multiopesti,
                vectorizer = vectorizer,
                remove_duplicates = args.removedups,
                grammar = mygrammar )

    ######
    #  DONE
    ######
    result = optimizer.optimize(landmark_graphs, target_vector )
    print(f" Target:")
    so.gprint(target)
    dumpfile(list(result)+[time.time() - starttime,mygrammar.size()[-1]], args.out)
    sys.exit(int(result[0]))
