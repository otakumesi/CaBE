import pickle
from collections import Counter

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

from CaBE.helper import read_triples, extract_phrases, canonical_phrases, transform_clusters


class CaBE:
    def __init__(self, name, model, file_name, distance_threshold, linkage):
        self.name = name
        self.model = model
        self.distance_threshold = distance_threshold
        self.linkage = linkage
        self.__init_triples(file_name)

    def __init_triples(self, file_path):
        raw_triples = read_triples(file_path)
        raw_ents, raw_rels, gold_ent2cluster = extract_phrases(raw_triples)

        self.file_path = file_path

        self.triples = raw_triples

        self.ent2freq = Counter(raw_ents)
        self.rel2freq = Counter(raw_rels)

        self.entities = list(set(raw_ents))
        self.relations = list(set(raw_rels))

        self.ent2id = {v: k for k, v in enumerate(self.entities)}
        self.id2ent = {v: k for k, v in self.ent2id.items()}
        self.rel2id = {v: k for k, v in enumerate(self.relations)}
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        self.__gold_ent2cluster = gold_ent2cluster

    def run(self, num_layer=12):
        print("----- Start: run CaBE -----")

        print("--- Start: encode entities ---")
        ent_pkl_path = f'{self.file_path}_ent_{self.name}'
        entities = self.model.encode(self.entities,
                                     num_layer=num_layer,
                                     file_prefix=ent_pkl_path)
        print("--- End: encode entities ---")

        print("--- Start: encode relations ---")
        rel_pkl_path = f'{self.file_path}_rel_{self.name}.pkl'
        relations = self.model.encode(self.relations,
                                      num_layer=num_layer,
                                      file_prefix=rel_pkl_path)
        print("--- End: encode relations ---")

        print("--- Start: cluster phrases ---")
        output_ent2cluster, rel_outputs = self.__cluster(entities, relations)
        self.dump_clusters(output_ent2cluster, 'ent')
        self.dump_clusters(rel_outputs, 'rel')
        print("--- End: cluster phrases ---")

        print("----- End: run CaBE -----")

        return output_ent2cluster, rel_outputs

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

        output_ent2cluster = {}
        for triple in self.triples:

            sbj, obj = triple['triple_norm'][0], triple['triple_norm'][2]
            triple_id = triple['_id']
            sbj_u, obj_u = f'{sbj}|{triple_id}', f'{obj}|{triple_id}'

            output_ent2cluster[sbj_u] = raw_ent2cluster[sbj]
            output_ent2cluster[obj_u] = raw_ent2cluster[obj]

        return output_ent2cluster, rel_outputs

    def __cluster_entities(self, entities):
        entity_cluster = AgglomerativeClustering(
            distance_threshold=self.distance_threshold,
            n_clusters=None,
            affinity='cosine',
            linkage=self.linkage)
        assigned_clusters = entity_cluster.fit_predict(entities)
        return transform_clusters(assigned_clusters)

    def __cluster_relations(self, relations):
        relation_cluster = AgglomerativeClustering(
            distance_threshold=self.distance_threshold,
            n_clusters=None,
            affinity='cosine',
            linkage=self.linkage)
        assigned_clusters = relation_cluster.fit_predict(relations)
        return transform_clusters(assigned_clusters)

    @property
    def gold_ent2cluster(self):
        return self.__gold_ent2cluster

    def dump_clusters(self, clusters, prefix):
        file_path = f'./data/{prefix}-{self.name}-{self.linkage}-threshold_{self.distance_threshold}.pkl'
        pickle.dump(clusters, open(file_path, 'wb'))
