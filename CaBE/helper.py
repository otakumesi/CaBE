import pickle
import os
import json
from collections import defaultdict

import hydra
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


def canonical_phrases(clusters, id2phrase):
    canonicalized_phrases = {}
    for phrase_ids in clusters.values():
        u_phrases = [id2phrase[p_id] for p_id in phrase_ids]
        representative = u_phrases[0]
        canonicalized_phrases[representative] = u_phrases
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

    elems_reduced = TSNE(n_components=2, random_state=None).fit_transform(elems)
    ele2id = {name: id for id, name in enumerate(clusters)}
    ids = [ele2id[name] for name in clusters]
    ax.scatter(elems_reduced[:, 0], elems_reduced[:, 1], c=ids)
    for i, phrase in enumerate(u_elems):
        ax.annotate(phrase, (elems_reduced[i, 0], elems_reduced[i, 1]))
