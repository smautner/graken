

import structout as so
from sklearn.neighbors import NearestNeighbors

import logging


def initialize(n_landmarks=10, n_neighbors=100, vectorizer=None, graphs=None, target=None):
    
        assert n_neighbors < len(graphs)

        vecs = vectorizer.transform(graphs)
        nn = NearestNeighbors(n_neighbors=n_neighbors).fit(vecs)
        


        target_vec = vectorizer.transform([target])
        distances, neighbors = nn.kneighbors(target_vec, return_distance=True)
        
        distances = distances[0]
        neighbors = neighbors[0]

        ranked_graphs = [graphs[i] for i in neighbors]
        landmark_graphs = ranked_graphs[:n_landmarks]
        desired_distances = distances[:n_landmarks]

        logging.debug ("target(%d,%d) and nn(%d,%d)" % (target.number_of_nodes(),
                                                       target.number_of_edges(),
                                                       ranked_graphs[0].number_of_nodes(),
                                                       ranked_graphs[0].number_of_edges()))

        so.gprint([target, ranked_graphs[0]], edgelabel='label')
        
        return landmark_graphs, desired_distances, ranked_graphs
