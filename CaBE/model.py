import pickle
import os
from datetime import datetime

import CaBE.helper as hlp
from CaBE.dataset import Triples

DATA_PATH = './data'
CLUSTER_PATH = './pkls/clusters'


class CaBE:
    def __init__(self, model, file_name):
        self.model = model
        self.file_name = file_name
        file_path = hlp.get_abspath(f'{DATA_PATH}/{file_name}')
        self.data = Triples.from_file(file_path)
        self.__init_elements()

    def __init_elements(self):
        print("--- Start: encode phrases ---")
        self.ent_embs, self.rel_embs = self.model.encode(self.data,
                                                         self.file_name)
        print("--- End: encode phrases ---")

    def get_encoded_elems(self, np_n_layer, rp_n_layer):
        entities = self.ent_embs[:, np_n_layer, :]
        relations = self.rel_embs[:, rp_n_layer, :]
        return entities, relations

    def run(self, np_clustering, rp_clustering):
        print("----- Start: run CaBE -----")
        ent2cluster = self.np_clusters(np_clustering)
        rel2cluster = self.rp_clusters(rp_clustering)
        self.dump_clusters(((np_clustering, ent2cluster),
                            (rp_clustering, rel2cluster)))
        print("----- End: run CaBE -----")

        return ent2cluster, rel2cluster

    def np_clusters(self, clust):
        print("--- Start: np cluster phrases ---")
        entities = self.ent_embs[:, clust.n_layer, :]
        np_clusters = clust.run(entities)
        ent2cluster = self.__format_cluster(np_clusters, self.data.id2ent)
        print("--- End: np cluster phrases ---")
        return ent2cluster

    def rp_clusters(self, clust):
        print("--- Start: rp cluster phrases ---")
        relations = self.rel_embs[:, clust.n_layer, :]
        rp_clusters = clust.run(relations)
        rel2cluster = self.__format_cluster(rp_clusters, self.data.id2rel)
        print("--- End: rp cluster phrases ---")
        return rel2cluster

    def __format_cluster(self, clusters, id2elem):
        elem_outputs = hlp.canonical_phrases(clusters, id2elem)
        elem2cluster = {}
        for ele, cluster in elem_outputs.items():
            for phrase in cluster:
                elem2cluster[phrase] = ele
        return elem2cluster

    def dump_clusters(self, clusters):
        os.makedirs(hlp.get_abspath(self.cluster_dumped_dir), exist_ok=True)
        with open(hlp.get_abspath(self.cluster_dumped_path), 'wb') as f:
            pickle.dump(clusters, f, protocol=pickle.HIGHEST_PROTOCOL)

    def read_clusters(self, cluster_dumped_path):
        with open(hlp.get_abspath(cluster_dumped_path), 'rb') as f:
            return pickle.load(f)

    @property
    def cluster_dumped_path(self):
        now = datetime.now()
        cluster_file_name = now.strftime('%Y%m%d_%H%M%S')
        return f'{self.cluster_dumped_dir}/{cluster_file_name}.pkl'

    @property
    def gold_ent2cluster(self):
        return self.data.gold_ent2cluster

    @property
    def gold_rel2cluster(self):
        return self.data.gold_rel2cluster

    @property
    def cluster_dumped_dir(self):
        return f'{CLUSTER_PATH}/{self.file_name}'
