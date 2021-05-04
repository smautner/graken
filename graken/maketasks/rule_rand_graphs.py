import random
import graphlearn.util.util as u
import graphlearn.local_substitution_graph_grammar as lsgg
from graken.ml import vector

def rule_rand_graphs(input_set, numgr =100, iter= 1, bottleneck = 500):
    # make grammar, fit on input
    grammar = lsgg.LocalSubstitutionGraphGrammar(radii=[1,2], thickness=1,
                 filter_min_cip=1, filter_min_interface=2, nodelevel_radius_and_thickness=True)#,
                 #cip_root_all = False,
                 #half_step_distance= False )
    grammar.fit(input_set)
    #grammar.structout()
    cleaner = vector.Vectorizer()
    
    sss = {} # permanent banlist for graphs
    for i in range(iter):
        random.shuffle(input_set)
        input_set = cleaner.duplicate_rm(input_set,sss)
        input_set= input_set[:bottleneck]
        input_set = [g for start in input_set for g  in grammar.neighbors(start)]
    # also needs duplicate removal
    
    input_set = cleaner.duplicate_rm(input_set,sss)
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
