import random
import graphlearn.util.util as u
import graphlearn.local_substitution_graph_grammar as lsgg
from graken.ml import vector

def rule_rand_graphs(input_set, numgr =100, iter= 1, bottleneck = 500, graphsize= 12):
    # make grammar, fit on input
    grammar = lsgg.LocalSubstitutionGraphGrammar(radii=[1,2], thickness=1,
                 filter_min_cip=1, filter_min_interface=2, nodelevel_radius_and_thickness=True)
    grammar.fit(input_set)
    grammar.structout()
    #grammar.structout()
    cleaner = vector.Vectorizer()
    startgraphs = input_set
    ####
    # graph filtering
    #####
    sss = {} # permanent banlist for graphs
    def filtergraphs(graphs, low, up ):
        graphs = cleaner.duplicate_rm(graphs,sss)
        graphs = [g for g in graphs if low <= len(g) <= up]
        return graphs

    for i in range(iter):
        input_set = filtergraphs(input_set, low = graphsize  -4, up = graphsize +4)
        random.shuffle(input_set)
        input_set= input_set[:bottleneck]
        input_set = [g for start in input_set for g  in grammar.neighbors(start)]
        print(f"graphs generated after iter{i}:{len(input_set)}")
    # also needs duplicate removal

    input_set = filtergraphs(input_set, low = graphsize, up = graphsize)

    print(f"graphs after lastsizefilter:{len(input_set)}")
    random.shuffle(input_set)
    input_set+=startgraphs
    return input_set[:numgr], grammar




def test_rulerand():
    import util.random_graphs as rg
    import structout as so
    import graphlearn3.util.setoperations as setop
    grs  = rg.make_graphs_static()[:30]#[:10] # default returns 100 graphs..
    res1, grammar1 =rule_rand_graphs(grs, numgr = 500,iter=2)
    #res, grammar2=rule_rand_graphs(res1, numgr = 50, iter=1)
    #so.gprint(res) #!!!!!!!!

    '''
    print("initial grammar:")
    print (grammar1)
    print("grammar after iteration")
    print (grammar2)
    inter = setop.intersect(grammar1, grammar2)
    print("intersection between grammars:")
    print (inter)
    '''


    #print("grammar1 - grammar2")
    #diff = setop.difference(grammar1,grammar2)
    #u.draw_grammar_term(diff)
    #u.draw_grammar_term(unique2)  # !!!!!!!!!!!!!!!!!!!!!!!!!
    #print ("generated %d graphs" % len(res1))
