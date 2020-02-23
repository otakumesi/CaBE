from CaBE.helper import read_triples
from collections import defaultdict


class Triples:
    @classmethod
    def from_file(cls, file_path):
        raw_triples = read_triples(file_path)
        return cls(raw_triples)
     
    def __init__(self, raw_triples):
        self.raw_triples = raw_triples
        self.triples = []
        self.entities = []
        self.relations = []
        self.entities_ = []
        self.relations_ = []

        self.gold_ent2cluster = {}
        self.gold_rel2cluster = {}
        self.sbj2sbj_u = {}
        self.obj2obj_u = {}
        self.rel2rel_u = {}

        self.__initialize(raw_triples)
        self.__after_init()

    def __initialize(self, raw_triples):
        for raw_triple in raw_triples:
            triple_id = raw_triple['_id']
            sbj = raw_triple['triple'][0]
            rel = raw_triple['triple'][1]
            obj = raw_triple['triple'][2]

            sbj_u = f'{sbj}|{triple_id}'
            rel_u = f'{rel}|{triple_id}'
            obj_u = f'{obj}|{triple_id}'

            gold_link_sbj = raw_triple['true_link']['subject']
            gold_link_obj = raw_triple['true_link']['object']

            self.gold_ent2cluster[sbj_u] = gold_link_sbj
            self.gold_rel2cluster[rel_u] = rel
            self.gold_ent2cluster[obj_u] = gold_link_obj

            self.triples.append((sbj, rel, obj))
            self.entities.extend([sbj, obj])
            self.relations.append(rel)

            self.entities_.extend([sbj_u, obj_u])
            self.relations_.append(rel_u)

            self.sbj2sbj_u[sbj] = sbj_u
            self.obj2obj_u[obj] = obj_u
            self.rel2rel_u[rel] = rel_u

    def __after_init(self):
        self.id2ent = {k: v for k, v in enumerate(self.entities_)}
        self.ent2id = defaultdict(set)
        for k, v in self.id2ent.items():
            self.ent2id[v].add(k)
        self.id2rel = {k: v for k, v in enumerate(self.relations_)}

        self.rel2id = defaultdict(set)
        for k, v in self.id2rel.items():
            self.rel2id[v].add(k)
