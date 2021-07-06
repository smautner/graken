import structout as so
from sklearn.neighbors import NearestNeighbors as neigh
import logging
import time


def initialize(n_landmarks=10, n_neighbors=100, vectorizer=None, graphs=None, target=None):
        t= time.time() 
        assert n_neighbors < len(graphs)

        vecs = vectorizer.transform(graphs)
        nn = neigh(n_neighbors=n_neighbors,metric='cosine').fit(vecs)

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
        
        logging.debug(f"finding landmarks done {time.time()-t:.3}s")
        return landmark_graphs, desired_distances, ranked_graphs
