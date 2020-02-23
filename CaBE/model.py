import pickle
import os

import CaBE.helper as hlp
from CaBE.dataset import Triples

DATA_PATH = './data'
CLUSTER_PATH = './pkls/clusters'


class CaBE:
    def __init__(self, name, model, file_name, clustering):
        self.name = name
        self.model = model
        self.clustering = clustering
        self.file_name = file_name
        file_path = hlp.get_abspath(f'{DATA_PATH}/{file_name}')
        self.data = Triples.from_file(file_path)

    def get_encoded_elems(self, num_layer=12):
        return self.model.encode(self.data,
                                 num_layer=num_layer,
                                 file_prefix=self.file_name)

    def run(self, num_layer=12):
        print("----- Start: run CaBE -----")

        print("--- Start: encode phrases ---")
        entities, relations = self.get_encoded_elems(num_layer)
        print("--- End: encode phrases ---")

        print("--- Start: cluster phrases ---")
        ent2cluster, rel2cluster = self.__cluster(entities, relations)
        self.dump_clusters((ent2cluster, rel2cluster))
        print("--- End: cluster phrases ---")

        print("----- End: run CaBE -----")

        return ent2cluster, rel2cluster

    def __cluster(self, entities, relations):
        ent2cluster = self.__gen_cluster(entities, self.data.id2ent)
        rel2cluster = self.__gen_cluster(relations, self.data.id2rel)
        return ent2cluster, rel2cluster

    def __gen_cluster(self, elements, id2elem):
        raw_clusters = self.clustering.run(elements)
        elem_outputs = hlp.canonical_phrases(raw_clusters, id2elem)
        raw_elem2cluster = {}
        for ele, cluster in elem_outputs.items():
            for phrase in cluster:
                raw_elem2cluster[phrase] = ele

        return raw_elem2cluster

    def dump_clusters(self, clusters):
        os.makedirs(hlp.get_abspath(self.cluster_dumped_dir), exist_ok=True)
        with open(hlp.get_abspath(self.cluster_dumped_path), 'wb') as f:
            pickle.dump(clusters, f, protocol=pickle.HIGHEST_PROTOCOL)

    def read_clusters(self):
        with open(hlp.get_abspath(self.cluster_dumped_path), 'rb') as f:
            return pickle.load(f)

    @property
    def gold_ent2cluster(self):
        return self.data.gold_ent2cluster

    @property
    def gold_rel2cluster(self):
        return self.data.gold_rel2cluster

    @property
    def cluster_dumped_dir(self):
        return f'{CLUSTER_PATH}/{self.file_name}'

    @property
    def cluster_file_name(self):
        return self.clustering.file_name(self.name)

    @property
    def cluster_dumped_path(self):
        return f'{self.cluster_dumped_dir}/{self.cluster_file_name}.pkl'
