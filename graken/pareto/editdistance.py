from networkx.algorithms.similarity import graph_edit_distance

def dist(arg):
    a,target = arg
    labelmatch = lambda a,b: a['label'] == b['label']
    return int(graph_edit_distance(a,target, labelmatch, labelmatch ))

