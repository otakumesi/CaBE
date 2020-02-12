from itertools import product
import numpy as np
import hydra
import mlflow
from mlflow import log_param, log_metric, log_artifact

from CaBE.model import CaBE
from CaBE.evaluator import Evaluator
from CaBE.language_model_encoder import BertEncoder, ElmoEncoder


DEFAULT_LOG_PATH = './log'
LMS = {'BERT': BertEncoder, 'Elmo': ElmoEncoder}


def predict(cfg):
    lm_name = cfg.ex.lm_name
    linkage = cfg.model.linkage
    threshold = cfg.model.threshold

    lang_model = LMS[lm_name]()
    num_layer = cfg.model.num_layer or lang_model.default_max_layer()

    file_name = hydra.utils.to_absolute_path(cfg.ex.filename)
    model = build_model(name=f'{lm_name}_{num_layer}',
                        lang_model=lang_model,
                        file_name=file_name,
                        threshold=threshold,
                        linkage=linkage)

    params = {"lm_name": lm_name,
              "num_layer": num_layer,
              "threshold": threshold,
              "linkage": linkage}

    experiment(model, params)


def grid_search(cfg):
    thresholds = np.arange(cfg.grid_search.min_threshold,
                           cfg.grid_search.max_threshold,
                           cfg.grid_search.threshold_step)

    lm_name = cfg.ex.lm_name
    lang_model = LMS[lm_name]()

    max_layer = cfg.grid_search.max_layer or lang_model.default_max_layer()
    layers = range(cfg.grid_search.min_layer, max_layer+1)

    clusteringcfgs = product(thresholds, cfg.grid_search.linkages, layers)
    file_name = hydra.utils.to_absolute_path(cfg.ex.filename)

    results = {}
    for thd, link, layer in clusteringcfgs:
        model = build_model(name=f'{lm_name}_{layer}',
                            lang_model=lang_model,
                            file_name=file_name,
                            threshold=thd,
                            linkage=link)

        params = {"lm_name": lm_name,
                  "num_layer": layer,
                  "threshold": thd,
                  "linkage": link}

        macro_f1, micro_f1, pairwise_f1 = experiment(model, params)
        log_name = '_'.join([f'{k}_{v}' for k, v in params.items()])
        results[log_name] = (macro_f1, micro_f1, pairwise_f1)

    sorted_confs = sorted(results.items(), key=lambda kv: np.mean(kv[1]))
    for name, f1s in sorted_confs:
        print(f'{name}: {f1s[0]:.4f}, {f1s[1]:.4f}, {f1s[2]:.4f}')


def build_model(name, lang_model, file_name, threshold, linkage):
    return CaBE(name=name,
                model=lang_model,
                file_name=file_name,
                distance_threshold=threshold,
                linkage=linkage)


def experiment(model, params):
    ent_outputs, rel_outputs = model.run(params['num_layer'])

    with mlflow.start_run():
        log_param('Language Model', params['lm_name'])
        log_param('Model Layer', params['num_layer'])
        log_param('Clustering Threshold', params['threshold'])
        log_param('Linkage', params['linkage'])

        print("--- Start: evlaluate noun phrases ---")
        evl = Evaluator(ent_outputs, model.gold_ent2cluster)

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
        log_metric('Pairwise Recall', evl.pairwise_recall)

        print('Pairwise F1: {}'.format(evl.pairwise_f1_score))
        log_metric('Pairwise F1', evl.pairwise_f1_score)

        log_artifact(hydra.utils.to_absolute_path(DEFAULT_LOG_PATH))

        print("--- End: evaluate noun phrases ---")
    return evl.macro_f1_score, evl.micro_f1_score, evl.pairwise_f1_score
