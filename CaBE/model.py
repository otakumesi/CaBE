import pickle
from collections import Counter, defaultdict
import hydra

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from scipy.stats import wasserstein_distance

from CaBE.helper import read_triples, extract_phrases, canonical_phrases, transform_clusters

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
        file_path = hydra.utils.to_absolute_path(f'{DATA_PATH}/{file_name}')
        self.__init_triples(file_path)

    def __init_triples(self, file_path):
        raw_triples = read_triples(file_path)
        (raw_ents,
         raw_rels,
         gold_ent2cluster,
         gold_rel2cluster) = extract_phrases(raw_triples)

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
        output_ent2cluster, output_rel2cluster = self.__cluster(entities, relations)
        self.dump_clusters(output_ent2cluster, 'ent')
        self.dump_clusters(output_rel2cluster, 'rel')
        print("--- End: cluster phrases ---")

        print("----- End: run CaBE -----")

        return output_ent2cluster, output_rel2cluster

    def __cluster(self, entities, relations):
        ent_raw_clusters = self.__cluster_entities(entities)
        ent_outputs = canonical_phrases(
            ent_raw_clusters, self.id2ent, self.ent2freq)

        rel_raw_clusters = self.__cluster_relations(relations)
        rel_outputs = canonical_phrases(
            rel_raw_clusters, self.id2rel, self.rel2freq)

        raw_ent2cluster = {}
        for ent, cluster in ent_outputs.items():
            for phrase in cluster:
                raw_ent2cluster[phrase] = ent

        raw_rel2cluster = {}
        for rel, cluster in rel_outputs.items():
            for phrase in cluster:
                raw_rel2cluster[phrase] = rel

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

    def __cluster_entities(self, entities):
        entity_cluster = AgglomerativeClustering(
            distance_threshold=self.distance_threshold,
            n_clusters=None,
            affinity=self.__affinity(),
            linkage=self.linkage)
        assigned_clusters = entity_cluster.fit_predict(entities)
        return transform_clusters(assigned_clusters)

    def __cluster_relations(self, relations):
        relation_cluster = AgglomerativeClustering(
            distance_threshold=self.distance_threshold,
            n_clusters=None,
            affinity=self.__affinity(),
            linkage=self.linkage)
        assigned_clusters = relation_cluster.fit_predict(relations)
        return transform_clusters(assigned_clusters)

    def __affinity(self):
        if self.similarity != 'wasserstein':
            return self.similarity

        return lambda X: pairwise_distances(X, metric=wasserstein_distance, n_jobs=-1)


    @property
    def gold_ent2cluster(self):
        return self.__gold_ent2cluster

    @property
    def gold_rel2cluster(self):
        return self.__gold_rel2cluster

    def dump_clusters(self, clusters, prefix):
        threshold = f'{self.distance_threshold:.6f}'
        names = [prefix, self.name, self.linkage, self.similarity, threshold]
        file_name = f'{CLUSTER_PATH}/{"_".join(names)}.pkl'
        file_path = hydra.utils.to_absolute_path(file_name)
        with open(file_path, 'wb') as f:
            pickle.dump(clusters, f, protocol=pickle.HIGHEST_PROTOCOL)
