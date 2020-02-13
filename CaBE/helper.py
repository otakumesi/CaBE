import pickle
import os
import json
from collections import defaultdict


def read_triples(file_path):
    pickle_path = f'{file_path}.pkl'
    if os.path.isfile(pickle_path):
        with open(pickle_path, 'rb') as f:
            triples = pickle.load(f)
    else:
        triples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                triples.append(json.loads(line))
        with open(pickle_path, 'wb') as f:
            pickle.dump(triples, f)
    return triples


def extract_phrases(raw_triples):
    entities, relations = [], []
    gold_ent2cluster = {}
    for raw_triple in raw_triples:
        triple_id = raw_triple['_id']
        subject_phrase = raw_triple['triple_norm'][0]
        relation_phrase = raw_triple['triple_norm'][1]
        object_phrase = raw_triple['triple_norm'][2]

        gold_link_sub = raw_triple['true_link']['subject']
        gold_link_obj = raw_triple['true_link']['object']

        gold_ent2cluster[f'{subject_phrase}|{triple_id}'] = gold_link_sub
        gold_ent2cluster[f'{object_phrase}|{triple_id}'] = gold_link_obj

        entities.extend([subject_phrase, object_phrase])
        relations.append(relation_phrase)

    return entities, relations, gold_ent2cluster


def canonical_phrases(clusters, id2phrase, id2freq):
    canonicalized_phrases = defaultdict(list)
    for phrase_ids in clusters.values():
        representative = id2phrase[phrase_ids[0]]
        cluster_phrases = set()
        for phrase_id in phrase_ids:
            current_phrase = id2phrase[phrase_id]
            cluster_phrases.add(current_phrase)
            if id2freq[representative] < id2freq[current_phrase]:
                representative = current_phrase
        canonicalized_phrases[representative] = list(cluster_phrases)
    return canonicalized_phrases


def transform_clusters(cluster_values):
    clusters = defaultdict(list)
    for ent_id, cluster in enumerate(cluster_values):
        clusters[cluster].append(ent_id)
    return clusters
