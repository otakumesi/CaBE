import os

from itertools import product
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt

import mlflow
from mlflow import log_param, log_metric, log_artifact

from CaBE.model import CaBE
from CaBE.evaluator import Evaluator
from CaBE.clustering import HAC, HDBSCAN
import CaBE.language_model_encoder as lme
from CaBE.helper import get_abspath, scatter_tsne

CLUSTER_PATH = './pkls/clusters'
LOG_PATH = './log'
LMS = {'BERT': lme.BertEncoder,
       'BERT-attn': lme.BertAttentionEncoder,
       'Elmo': lme.ElmoEncoder}


def predict(cfg):
    enc_name = cfg.ex.enc
    linkage = cfg.model.linkage
    threshold = cfg.model.threshold
    similarity = cfg.ex.similarity

    enc_model = LMS[enc_name]()
    num_layer = cfg.model.num_layer or enc_model.default_max_layer

    clustering = HAC(distance_threshold=threshold,
                     similarity=similarity,
                     linkage=linkage)
    model = build_model(name=f'{enc_name}_{num_layer}',
                        enc_model=enc_model,
                        file_name=cfg.ex.file_name,
                        clustering=clustering)

    params = {"enc_name": enc_name,
              "num_layer": num_layer,
              "threshold": threshold,
              "similarity": similarity,
              "linkage": linkage}

    experiment(model, params)


def grid_search(cfg):
    thresholds = np.arange(cfg.grid_search.min_threshold,
                           cfg.grid_search.max_threshold,
                           cfg.grid_search.threshold_step)

    enc_name = cfg.ex.enc
    enc_model = LMS[enc_name]()

    max_layer = cfg.grid_search.max_layer or enc_model.default_max_layer
    layers = range(cfg.grid_search.min_layer, max_layer+1)

    configs_for_clustering = product([enc_name],
                                     [cfg.ex.file_name],
                                     [enc_model],
                                     thresholds,
                                     [cfg.ex.similarity],
                                     cfg.grid_search.linkages,
                                     layers)

    with Pool(processes=cfg.grid_search.num_process) as p:
        results = p.starmap(_grid_search, configs_for_clustering)

    sorted_confs = sorted(results, key=lambda kv: np.mean(kv[1]))
    for name, f1s in sorted_confs:
        print(f'{name}: {f1s[0]:.4f}, {f1s[1]:.4f}, {f1s[2]:.4f}')


def _grid_search(enc_name, file_name, enc, thd, sim, link, layer):
    clustering = HAC(distance_threshold=thd,
                     similarity=sim,
                     linkage=link)

    model = build_model(name=f'{enc_name}_{layer}',
                        enc_model=enc,
                        file_name=file_name,
                        clustering=clustering)

    params = {"enc_name": enc_name,
              "num_layer": layer,
              "threshold": thd,
              "similarity": sim,
              "linkage": link}

    (macro_f1, micro_f1, pairwise_f1), _ = experiment(model, params)
    log_name = '_'.join([f'{k}_{v}' for k, v in params.items()])

    return log_name, (macro_f1, micro_f1, pairwise_f1)


def build_model(name, enc_model, file_name, clustering):
    return CaBE(name=name,
                model=enc_model,
                file_name=file_name,
                clustering=clustering)


def experiment(model, params):
    ent_outputs, rel_outputs = model.run(params['num_layer'])
    enc_name, num_layer = params["enc_name"], params["num_layer"]
    threshold, linkage, similarity = params['threshold'], params['linkage'], params["similarity"]

    with mlflow.start_run():
        log_param('Language Model', enc_name)
        log_param('Model Layer', num_layer)
        log_param('Clustering Threshold', threshold)
        log_param('Similarity', similarity)

        log_param('Linkage', linkage)
        param_log = f'Language Model: {enc_name}, Layer: {num_layer}, '\
            f'Threshold: {threshold}, Similarity: {similarity}, Linkage: {linkage}'
        print(param_log)

        print("--- Start: evlaluate noun phrases ---")
        ent_evl = Evaluator(ent_outputs, model.gold_ent2cluster)
        eval_and_log(ent_evl)
        ent_f1s = (ent_evl.macro_f1_score,
                   ent_evl.micro_f1_score,
                   ent_evl.pairwise_f1_score)
        print("--- End: evaluate noun phrases ---")

        print("--- Start: evlaluate rel phrases ---")
        rel_evl = Evaluator(rel_outputs, model.gold_rel2cluster)
        eval_and_log(rel_evl)
        rel_f1s = (rel_evl.macro_f1_score,
                   rel_evl.micro_f1_score,
                   rel_evl.pairwise_f1_score)
        print("--- End: evaluate rel phrases ---")

    return ent_f1s, rel_f1s


def eval_and_log(evl):
    print('Macro Precision: {}'.format(evl.macro_precision))
    log_metric('Macro Precision', evl.macro_precision)

    print('Macro Recall: {}'.format(evl.macro_recall))
    log_metric('Macro Recall', evl.macro_recall)

    print('Macro F1: {}'.format(evl.macro_f1_score))
    log_metric('Macro F1', evl.macro_f1_score)

    print('Micro Precision: {}'.format(evl.micro_precision))
    log_metric('Micro Precision', evl.micro_precision)

    print('Micro Recall: {}'.format(evl.micro_recall))
    log_metric('Micro Recall', evl.micro_recall)

    print('Micro F1: {}'.format(evl.micro_f1_score))
    log_metric('Micro F1', evl.micro_f1_score)

    print('Pairwise Precision: {}'.format(evl.pairwise_precision))
    log_metric('Pairwise Precision', evl.pairwise_precision)

    print('Pairwise Recall: {}'.format(evl.pairwise_recall))

    print('Pairwise F1: {}'.format(evl.pairwise_f1_score))
    log_metric('Pairwise F1', evl.pairwise_f1_score)

    log_artifact(get_abspath(LOG_PATH))


def visualize_cluster(cfg):
    file_name = cfg.ex.file_name
    enc_name = cfg.ex.enc
    linkage = cfg.model.linkage
    similarity = cfg.ex.similarity
    threshold = cfg.model.threshold

    enc_model = LMS[enc_name]()
    num_layer = cfg.model.num_layer or enc_model.default_max_layer

    print('--- Start: build model ---')
    clustering = HAC(distance_threshold=threshold,
                     similarity=similarity,
                     linkage=linkage)
    model = build_model(name=f'{enc_name}_{num_layer}',
                        enc_model=enc_model,
                        file_name=file_name,
                        clustering=clustering)
    print('--- End: build model ---')

    print('--- Start: read phrases ---')
    entities, relations = model.get_encoded_elems(num_layer=num_layer)
    print('--- End: read phrases ---')

    print('--- Start: read clusters ---')
    ent2clusters, rel2clusters = model.read_clusters()
    print('--- End: read clusters ---')

    plt_dir = f'plt_img/{model.cluster_dumped_dir}'
    os.makedirs(get_abspath(plt_dir), exist_ok=True)

    plt_path = f'{plt_dir}/{model.cluster_file_name}'

    n_min_elems = cfg.vis.n_min_elems
    n_max_elems = cfg.vis.n_max_elems
    if n_min_elems:
        plt_path += f'_min{n_min_elems}'
    if n_max_elems:
        plt_path += f'_max{n_max_elems}'
    plt_path += f'.png'

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle('t-SNE of entities and relation clusters')

    print('--- Start: plot noun phrases ---')
    scatter_tsne(entities, ent2clusters, axes[0], n_max_elems, n_min_elems)
    print('--- End: plot noun phrases ---')

    print('--- Start: plot rel phrases ---')
    scatter_tsne(relations, rel2clusters, axes[1], n_max_elems, n_min_elems)
    print('--- End: plot rel phrases ---')

    plt.savefig(get_abspath(plt_path))
