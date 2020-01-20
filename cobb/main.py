import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from bert_serving.client import BertClient

DEFAULT_TRIPLE_PATH = '../data/reverb45k_test'


class COBB_Main():
    def __init__(self, distance_threshold=.4239, triple_path=DEFAULT_TRIPLE_PATH):
        self.client = BertClient()
        self.distance_threshold = distance_threshold
        self.__init_triples(triple_path)

    def __init_triples(self, triple_path):
        raw_triples = self.__read_triples(triple_path)
        self.entities, self.relations = self.__extract_phrases(raw_triples)

        self.ent2id = {v:k for k, v in enumerate(self.entities)}
        self.id2ent = {v:k for k, v in self.ent2id.items()}
        self.rel2id = {v:k for k, v in enumerate(self.relations)}
        self.id2rel = {v:k for k, v in self.rel2id.items()}

    def run(self):
        print("----- Start: run COBB -----")

        print("--- Start: encode entities ---")
        entities = self.client.encode(self.entities)
        print("--- End: encode entities ---")

        print("--- Start: encode relations ---")
        relations = self.client.encode(self.relations)
        print("--- End: encode relations ---")

        print("--- Start: cluster phrases ---")
        self.__cluster(entities, relations)
        print("--- End: cluster phrases ---")

        print("----- End: run COBB -----")

    def __read_triples(self, path):
        triples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                triples.append(json.loads(line))
        return triples

    def __extract_phrases(self, raw_triples):
        entities, relations = [], []
        for raw_triple in raw_triples:
            subject_phrase = raw_triple['triple'][0]
            relation_phrase = raw_triple['triple'][1]
            object_phrase = raw_triple['triple'][2]

            entities.extend([subject_phrase, object_phrase])
            relations.append(relation_phrase)

        entities, relations = list(set(entities)), list(set(relations))

        return entities, relations

    def __cluster(self, entities, relations):
        ent_assined_cluster_list = self.__cluster_entities(entities)
        rel_assined_cluster_list = self.__cluster_relations(relations)

        self.ent2cluster = {self.id2ent[k]:v  for k, v in enumerate(ent_assined_cluster_list)}
        self.rel2cluster = {self.id2rel[k]:v  for k, v in enumerate(rel_assined_cluster_list)}
        print(self.ent2cluster, self.rel2cluster)

    def __cluster_entities(self, entities):
        entity_cluster = AgglomerativeClustering(
            distance_threshold=self.distance_threshold,
            n_clusters=None,
            affinity="cosine",
            linkage="single")
        print(entity_cluster.n_clusters, entity_cluster.distance_threshold)
        return entity_cluster.fit_predict(entities)

    def __cluster_relations(self, relations):
        relation_cluster = AgglomerativeClustering(
            distance_threshold=self.distance_threshold,
            n_clusters=None,
            affinity="cosine",
            linkage="single")
        return relation_cluster.fit_predict(relations)


COBB_Main().run()
