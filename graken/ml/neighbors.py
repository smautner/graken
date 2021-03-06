import structout as so
from sklearn.neighbors import NearestNeighbors as neigh
import logging
logger=logging.getLogger(__name__)
import time


def initialize(n_landmarks=10, n_neighbors=100, vectorizer=None, graphs=None, target=None, target_vec = None):
        t= time.time()
        assert n_neighbors < len(graphs), f"neigh{n_neighbors}, gr: {len(graphs)}"

        vecs = vectorizer.transform(graphs)
        nn = neigh(n_neighbors=n_neighbors,metric='cosine').fit(vecs)

        distances, neighbors = nn.kneighbors(target_vec, return_distance=True)

        distances = distances[0]
        neighbors = neighbors[0]

        ranked_graphs = [graphs[i] for i in neighbors]
        landmark_graphs = ranked_graphs[:n_landmarks]
        desired_distances = distances[:n_landmarks]

        logger.debug ("target(%d,%d) and nn(%d,%d)" % (target.number_of_nodes(),
                                                       target.number_of_edges(),
                                                       ranked_graphs[0].number_of_nodes(),
                                                       ranked_graphs[0].number_of_edges()))
        print('neighbors: target and NN:')
        so.gprint([target, ranked_graphs[0]], edgelabel='label')

        logger.debug(f"finding landmarks done {time.time()-t:.3}s")
        return landmark_graphs, desired_distances, ranked_graphs
