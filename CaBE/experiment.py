import os

from itertools import product
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt

import mlflow
from mlflow import log_param, log_metric, log_artifact

from CaBE.model import CaBE
from CaBE.evaluator import Evaluator
from CaBE.clustering import HAC
import CaBE.language_model_encoder as lme
from CaBE.helper import get_abspath, scatter_tsne

CLUSTER_PATH = './pkls/clusters'
LOG_PATH = './log'
LMS = {'BERT': lme.BertEncoder,
       'RoBERTa': lme.RobertaEncoder,
       'Elmo': lme.ElmoEncoder}


def predict(cfg):
    enc_name = cfg.ex.enc

    np_sim = cfg.ex.np_sim
    np_thd = cfg.model.np_thd
    np_linkage = cfg.model.np_linkage

    rp_sim = cfg.ex.rp_sim
    rp_thd = cfg.model.rp_thd
    rp_linkage = cfg.model.rp_linkage

    enc_model = LMS[enc_name]()
    if cfg.model.np_n_layer is None:
        np_n_layer = enc_model.default_max_layer
    else:
        np_n_layer = cfg.model.np_n_layer

    if cfg.model.rp_n_layer is None:
        rp_n_layer = enc_model.default_max_layer
    else:
        rp_n_layer = cfg.model.rp_n_layer

    np_clustering = HAC(threshold=np_thd,
                        similarity=np_sim,
                        linkage=np_linkage,
                        n_layer=np_n_layer)
    rp_clustering = HAC(threshold=rp_thd,
                        similarity=rp_sim,
                        linkage=rp_linkage,
                        n_layer=rp_n_layer)

    model = build_model(enc_model=enc_model, file_name=cfg.ex.file_name)

    params = {"enc_name": enc_name,
              "np_clustering": np_clustering,
              "rp_clustering": rp_clustering}

    experiment(model, params)


def grid_search(cfg):
    np_thresholds = np.arange(cfg.grid_search.np_min_thd,
                              cfg.grid_search.np_max_thd,
                              cfg.grid_search.np_thd_step)

    rp_thresholds = np.arange(cfg.grid_search.rp_min_thd,
                              cfg.grid_search.rp_max_thd,
                              cfg.grid_search.rp_thd_step)

    enc_name = cfg.ex.enc
    enc_model = LMS[enc_name]()

    if cfg.grid_search.np_max_layer is None:
        np_max_layer = enc_model.default_max_layer
    else:
        np_max_layer = cfg.grid_search.np_max_layer

    if cfg.grid_search.rp_max_layer is None:
        rp_max_layer = enc_model.default_max_layer
    else:
        rp_max_layer = cfg.grid_search.rp_max_layer

    np_layers = range(cfg.grid_search.np_min_layer, np_max_layer+1)
    rp_layers = range(cfg.grid_search.rp_min_layer, rp_max_layer+1)

    model = build_model(enc_model=enc_model, file_name=cfg.ex.file_name)

    config_np_clust = product([model],
                              np_thresholds,
                              [cfg.ex.np_sim],
                              cfg.grid_search.linkages,
                              np_layers)

    config_rp_clust = product([model],
                              rp_thresholds,
                              [cfg.ex.rp_sim],
                              cfg.grid_search.linkages,
                              rp_layers)

    with Pool(processes=cfg.grid_search.n_process) as p:
        np_results = p.starmap(_np_grid_search, config_np_clust)
        rp_results = p.starmap(_rp_grid_search, config_rp_clust)

    np_results = filter(None, np_results)
    rp_results = filter(None, rp_results)

    np_sorted = sorted(np_results, key=lambda kv: np.mean(kv[1]))
    for name, f1s in np_sorted:
        print(f'{name}-np: {f1s[0]:.4f}, {f1s[1]:.4f}, {f1s[2]:.4f}')

    rp_sorted = sorted(rp_results, key=lambda kv: np.mean(kv[1]))
    for name, f1s in rp_sorted:
        print(f'{name}-rp: {f1s[0]:.4f}, {f1s[1]:.4f}, {f1s[2]:.4f}')


def _np_grid_search(model, thd, sim, link, layer):
    clust = HAC(threshold=thd,
                similarity=sim,
                linkage=link,
                n_layer=layer)

    log_name = f'{model.model.__class__.__name__}_{clust.name}'

    try:
        print("--- Start: evlaluate noun phrases ---")
        ent2cluster = model.np_clusters(clust)
        evl = Evaluator(ent2cluster, model.gold_ent2cluster)
        eval_and_log(evl)
        macro_f1 = evl.macro_f1_score
        micro_f1 = evl.micro_f1_score
        pairwise_f1 = evl.pairwise_f1_score
        print("--- End: evaluate noun phrases ---")
    except ValueError:
        # TODO: 根本的な対応はいつか。
        print(f'{log_name} is invalid.')
        return None
    return log_name, (macro_f1, micro_f1, pairwise_f1)


def _rp_grid_search(model, thd, sim, link, layer):
    clust = HAC(threshold=thd,
                similarity=sim,
                linkage=link,
                n_layer=layer)

    log_name = f'{model.model.__class__.__name__}_{clust.name}'

    try:
        print("--- Start: evlaluate rel phrases ---")
        rel2cluster = model.rp_clusters(clust)
        evl = Evaluator(rel2cluster, model.gold_rel2cluster)
        eval_and_log(evl)
        macro_f1 = evl.macro_f1_score
        micro_f1 = evl.micro_f1_score
        pairwise_f1 = evl.pairwise_f1_score
        print("--- End: evaluate rel phrases ---")
    except ValueError:
        # TODO: 根本的な対応はいつか。
        print(f'{log_name} is invalid.')
        return None
    return log_name, (macro_f1, micro_f1, pairwise_f1)


def build_model(enc_model, file_name):
    return CaBE(model=enc_model, file_name=file_name)


def experiment(model, params):
    np_clust = params["np_clustering"]
    rp_clust = params["rp_clustering"]

    ent2cluster, rel2cluster = model.run(np_clust, rp_clust)

    with mlflow.start_run():
        enc_name = model.model.__class__.__name__
        log_param('Encoder', enc_name)

        log_param('Model Layer of NP', np_clust.n_layer)
        log_param('Clustering Threshold of NP', np_clust.threshold)
        log_param('Similarity of NP', np_clust.similarity)
        log_param('Linkage of NP', np_clust.linkage)

        log_param('Model Layer of RP', rp_clust.n_layer)
        log_param('Clustering Threshold of RP', rp_clust.threshold)
        log_param('Similarity of RP', rp_clust.similarity)
        log_param('Linkage of RP', rp_clust.linkage)

        param_log = f'Encoder: {enc_name}, '\
            f'NP Layer: {np_clust.n_layer}, '\
            f'NP Threshold: {np_clust.threshold}, '\
            f'NP Similarity: {np_clust.similarity}, '\
            f'NP Linkage: {np_clust.linkage}, ' \
            f'RP Layer: {rp_clust.n_layer}, '\
            f'RP Threshold: {rp_clust.threshold}, '\
            f'RP Similarity: {rp_clust.similarity}, '\
            f'RP Linkage: {rp_clust.linkage}'
        print(param_log)

        print("--- Start: evlaluate noun phrases ---")
        ent_evl = Evaluator(ent2cluster, model.gold_ent2cluster)
        eval_and_log(ent_evl)
        ent_f1s = (ent_evl.macro_f1_score,
                   ent_evl.micro_f1_score,
                   ent_evl.pairwise_f1_score)
        print("--- End: evaluate noun phrases ---")

        print("--- Start: evlaluate rel phrases ---")
        rel_evl = Evaluator(rel2cluster, model.gold_rel2cluster)
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
    enc_model = LMS[enc_name]()

    np_n_layer = cfg.model.np_n_layer or enc_model.default_max_layer
    rp_n_layer = cfg.model.rp_n_layer or enc_model.default_max_layer

    print('--- Start: build model ---')
    model = build_model(enc_model=enc_model, file_name=file_name)
    print('--- End: build model ---')

    print('--- Start: read phrases ---')
    entities, relations = model.get_encoded_elems(np_n_layer, rp_n_layer)
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
