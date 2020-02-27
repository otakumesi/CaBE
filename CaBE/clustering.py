import hdbscan
from numpy import cov
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import haversine_distances
from sklearn.neighbors import DistanceMetric

import CaBE.helper as hlp


class HAC:
    def __init__(self, threshold, similarity, linkage, n_layer):
        self.threshold = threshold
        self.similarity = similarity
        self.linkage = linkage
        self.n_layer = n_layer
        self.clustering = AgglomerativeClustering(
            distance_threshold=self.threshold,
            n_clusters=None,
            affinity=self.affinity,
            linkage=self.linkage)

    def run(self, elements):
        if self.similarity == 'mahalanobis':
            dist = DistanceMetric.get_metric('mahalanobis', V=cov(elements))
            elements = dist.pairwise(elements)
        assigned_clusters = self.clustering.fit_predict(elements)
        return hlp.transform_clusters(assigned_clusters)

    @property
    def name(self):
        threshold = f'{self.threshold:.4f}'
        props = [f'{self.n_layer}', self.similarity, self.linkage, threshold]
        return '_'.join(props)

    @property
    def affinity(self):
        if self.similarity == 'mahalanobis':
            return 'precomputed'

        if self.similarity == 'haversine':
            return haversine_distances

        return self.similarity


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
      
