import hdbscan
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import wasserstein_distance
from sklearn.metrics import pairwise_distances

import CaBE.helper as hlp


class HAC:
    def __init__(self, distance_threshold, similarity, linkage):
        self.threshold = distance_threshold
        self.similarity = similarity
        self.linkage = linkage
        self.clustering = AgglomerativeClustering(
            distance_threshold=self.threshold,
            n_clusters=None,
            affinity=self.affinity,
            linkage=self.linkage)

    def run(self, elements):
        assigned_clusters = self.clustering.fit_predict(elements)
        return hlp.transform_clusters(assigned_clusters)

    def file_name(self, name):
        threshold = f'{self.distance_threshold:.6f}'
        names = [name, self.linkage, self.similarity, threshold]
        return "_".join(names)
      
    @property
    def affinity(self):
        if self.similarity != 'wasserstein':
            return self.similarity

        return lambda X: pairwise_distances(X, metric=wasserstein_distance, n_jobs=-1)


class HDBSCAN:
    def __init__(self, similarity, cluster_size):
        self.similarity = similarity
        self.clustering = hdbscan.HDBSCAN(min_cluster_size=cluster_size,
                                          gen_min_span_tree=True,
                                          metric=self.similarity)

    def run(self, elements):
        assigned_clusters = self.clustering.fit_predict(elements)
        return hlp.transform_clusters(assigned_clusters)

    def file_name(self, name):
        names = [name, self.similarity]
        return "_".join(names)
      
