import pickle
import os
from collections import Counter, defaultdict
import hydra

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from scipy.stats import wasserstein_distance

import CaBE.helper as hlp

DATA_PATH = './data'
CLUSTER_PATH = './pkls/clusters'


class CaBE:
    def __init__(self, name, model, file_name, distance_threshold, similarity, linkage):
        self.name = name
        self.model = model
        self.distance_threshold = distance_threshold
        self.linkage = linkage
        self.similarity = similarity
        self.file_name = file_name
        file_path = hlp.get_abspath(f'{DATA_PATH}/{file_name}')
        self.__init_triples(file_path)

    def __init_triples(self, file_path):
        raw_triples = hlp.read_triples(file_path)
        (raw_ents,
         raw_rels,
         gold_ent2cluster,
         gold_rel2cluster) = hlp.extract_phrases(raw_triples)

        self.__gold_ent2cluster = gold_ent2cluster
        self.__gold_rel2cluster = gold_rel2cluster
        self.triples = raw_triples

        self.ent2freq = Counter(raw_ents)
        self.rel2freq = Counter(raw_rels)

        self.entities = list(raw_ents)
        self.relations = list(raw_rels)

        self.id2ent = {k: v for k, v in enumerate(self.entities)}
        self.ent2id = defaultdict(set)
        for k, v in self.id2ent.items():
            self.ent2id[v].add(k)
        self.id2rel = {k: v for k, v in enumerate(self.relations)}

        self.rel2id = defaultdict(set)
        for k, v in self.id2rel.items():
            self.rel2id[v].add(k)

    def get_encoded_elems(self, num_layer=12):
        ent_prefix = f'{self.file_name}_ent'
        entities = self.model.encode(self.entities,
                                     num_layer=num_layer,
                                     file_prefix=ent_prefix)
        rel_prefix = f'{self.file_name}_rel'
        relations = self.model.encode(self.relations,
                                      num_layer=num_layer,
                                      file_prefix=rel_prefix)
        return entities, relations

    def run(self, num_layer=12):
        print("----- Start: run CaBE -----")

        print("--- Start: encode entities ---")
        ent_prefix = f'{self.file_name}_ent'
        entities = self.model.encode(self.entities,
                                     num_layer=num_layer,
                                     file_prefix=ent_prefix)
        print("--- End: encode entities ---")

        print("--- Start: encode relations ---")
        rel_prefix = f'{self.file_name}_rel'
        relations = self.model.encode(self.relations,
                                      num_layer=num_layer,
                                      file_prefix=rel_prefix)
        print("--- End: encode relations ---")

        print("--- Start: cluster phrases ---")
        ent2cluster, rel2cluster = self.__cluster(entities, relations)
        self.dump_clusters((ent2cluster, rel2cluster))
        print("--- End: cluster phrases ---")

        print("----- End: run CaBE -----")

        return ent2cluster, rel2cluster

    def __cluster(self, entities, relations):
        # threshold, affinity, linkageをentとrelでそれぞれ指定できるようにする前フリ
        raw_ent2cluster = self.__output_cluster(entities,
                                                self.id2ent,
                                                self.ent2freq,
                                                self.distance_threshold,
                                                self.__affinity(),
                                                self.linkage)

        raw_rel2cluster = self.__output_cluster(relations,
                                                self.id2rel,
                                                self.rel2freq,
                                                self.distance_threshold,
                                                self.__affinity(),
                                                self.linkage)

        output_ent2cluster = {}
        output_rel2cluster = {}
        for triple in self.triples:
            sbj, rel, obj = triple['triple_norm'][0], triple['triple_norm'][1], triple['triple_norm'][2]
            triple_id = triple['_id']
            sbj_u, rel_u, obj_u = f'{sbj}|{triple_id}', f'{rel}|{triple_id}', f'{obj}|{triple_id}'
            output_ent2cluster[sbj_u] = raw_ent2cluster[sbj]
            output_ent2cluster[obj_u] = raw_ent2cluster[obj]
            output_rel2cluster[rel_u] = raw_rel2cluster[rel]
        return output_ent2cluster, output_rel2cluster

    def __output_cluster(self, elements, id2elem, elem2freq, threshold, affinity, linkage):
        raw_clusters = self.__cluster_elems(elements,
                                            threshold,
                                            affinity,
                                            linkage)
        elem_outputs = hlp.canonical_phrases(raw_clusters, id2elem, elem2freq)

        raw_elem2cluster = {}
        for ele, cluster in elem_outputs.items():
            for phrase in cluster:
                raw_elem2cluster[phrase] = ele

        return raw_elem2cluster

    def __cluster_elems(self, elements, threshold, affinity, linkage):
        elem_cluster = AgglomerativeClustering(
            distance_threshold=threshold,
            n_clusters=None,
            affinity=affinity,
            linkage=linkage)
        assigned_clusters = elem_cluster.fit_predict(elements)
        return hlp.transform_clusters(assigned_clusters)

    def __affinity(self):
        if self.similarity != 'wasserstein':
            return self.similarity

        return lambda X: pairwise_distances(X, metric=wasserstein_distance, n_jobs=-1)

    def dump_clusters(self, clusters):
        os.makedirs(hlp.get_abspath(self.cluster_dumped_dir), exist_ok=True)
        with open(hlp.get_abspath(self.cluster_dumped_path), 'wb') as f:
            pickle.dump(clusters, f, protocol=pickle.HIGHEST_PROTOCOL)

    def read_clusters(self):
        with open(hlp.get_abspath(self.cluster_dumped_path), 'rb') as f:
            return pickle.load(f)

    @property
    def gold_ent2cluster(self):
        return self.__gold_ent2cluster

    @property
    def gold_rel2cluster(self):
        return self.__gold_rel2cluster

    @property
    def cluster_dumped_dir(self):
        return f'{CLUSTER_PATH}/{self.file_name}'

    @property
    def cluster_file_name(self):
        threshold = f'{self.distance_threshold:.6f}'
        names = [self.name, self.linkage, self.similarity, threshold]
        return "_".join(names)

    @property
    def cluster_dumped_path(self):
        return f'{self.cluster_dumped_dir}/{self.cluster_file_name}.pkl'
