import hdbscan
from numpy import cov
from sklearn.cluster import AgglomerativeClustering
import CaBE.helper as hlp


class HAC:
    def __init__(self, threshold, similarity, linkage, n_layer):
        self.threshold = threshold
        self.similarity = similarity
        self.linkage = linkage
        self.n_layer = n_layer

    def run(self, elements):
        if self.similarity == 'mahalanobis':
            V = empirical_covariance(elements)
            affinity = lambda X: pairwise_distances(X, metric='mahalanobis', V=V)
        else:
            affinity = self.similarity
        assigned_clusters = self.clustering(affinity).fit_predict(elements)
        return hlp.transform_clusters(assigned_clusters)

    def clustering(self, affinity):
        return AgglomerativeClustering(
            distance_threshold=self.threshold,
            n_clusters=None,
            affinity=affinity,
            linkage=self.linkage)

    @property
    def name(self):
        threshold = f'{self.threshold:.4f}'
        props = [f'{self.n_layer}', self.similarity, self.linkage, threshold]
        return '_'.join(props)


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
