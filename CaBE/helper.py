import pickle
import hydra

import os
import json
from collections import defaultdict
from sklearn.manifold import TSNE


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
    gold_rel2cluster = {}
    for raw_triple in raw_triples:
        triple_id = raw_triple['_id']
        sub_phrase = raw_triple['triple_norm'][0]
        rel_phrase = raw_triple['triple_norm'][1]
        obj_phrase = raw_triple['triple_norm'][2]

        gold_link_sub = raw_triple['true_link']['subject']
        gold_link_obj = raw_triple['true_link']['object']

        gold_ent2cluster[f'{sub_phrase}|{triple_id}'] = gold_link_sub
        gold_ent2cluster[f'{obj_phrase}|{triple_id}'] = gold_link_obj

        gold_rel2cluster[f'{rel_phrase}|{triple_id}'] = rel_phrase

        entities.extend([sub_phrase, obj_phrase])
        relations.append(rel_phrase)

    return entities, relations, gold_ent2cluster, gold_rel2cluster


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


def get_abspath(file_path):
    return hydra.utils.to_absolute_path(file_path)


def scatter_tsne(elems, ele2clusters, ax, n_max_elems, n_min_elems):
    u_elems = ele2clusters.keys()
    clusters = ele2clusters.values()

    if not n_min_elems:
        n_min_elems = 0

    if n_max_elems:
        elems = elems[n_min_elems:n_max_elems]
        u_elems = list(u_elems)[n_min_elems:n_max_elems]
        clusters = list(ele2clusters.values())[n_min_elems:n_max_elems]

    elems_reduced = TSNE(n_components=2, random_state=0).fit_transform(elems)
    ele2id = {name: id for id, name in enumerate(clusters)}
    ids = [ele2id[name] for name in clusters]
    ax.scatter(elems_reduced[:, 0], elems_reduced[:, 1], c=ids)
    for i, phrase in enumerate(u_elems):
        ax.annotate(phrase, (elems_reduced[i, 0], elems_reduced[i, 1]))

