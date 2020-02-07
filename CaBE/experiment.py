from itertools import product
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver
from CaBE.model import CaBE
from CaBE.evaluator import Evaluator
from CaBE.language_model_encoder import BertEncoder


DEFAULT_REVERB_PATH = './data/reverb45k_test'
DEFAULT_LOG_PATH = './log'
ex = Experiment('CaBE Expriment')
ex.observers.append(FileStorageObserver(DEFAULT_LOG_PATH))

LMS = {'BERT': BertEncoder, 'Elmo': None}


@ex.config
def experiment_config():
    file_name = DEFAULT_REVERB_PATH
    lm_name = 'BERT'

    # Config for main
    num_layers = 12
    threshold = .0000
    linkage = 'complete'

    # Config for grid_search
    min_threshold = 0.00000
    max_threshold = 0.00001
    threshold_step = 0.000001
    min_layer = 1
    max_layer = 12
    linkages = ['single', 'complete', 'average']


@ex.automain
def main(_run, _config):
    lm_name = _config['lm_name']
    num_layers = _config["num_layers"]
    linkage = _config["linkage"]
    threshold = _config["threshold"]

    lang_model = LMS[lm_name]()

    name = f'{lm_name}_{num_layers}'
    model = build_model(name=name,
                        lang_model=lang_model,
                        file_name=_config['file_name'],
                        threshold=threshold,
                        linkage=linkage)
    log_name = f'{lm_name}_{num_layers}_{linkage}_{threshold}'

    experiment(_run, model, log_name, num_layers)


@ex.command
def grid_search(_run, _config):
    thresholds = np.arange(_config['min_threshold'],
                           _config['max_threshold'],
                           _config['threshold_step'])
    layers = range(_config['min_layer'], _config['max_layer']+1)

    clustering_configs = product(thresholds, _config['linkages'], layers)
    lm_name = _config["lm_name"]
    lang_model = LMS[lm_name]()

    results = {}
    for thd, link, layer in clustering_configs:
        name = f'{lm_name}_{layer}'
        model = build_model(name=name,
                            lang_model=lang_model,
                            file_name=_config['file_name'],
                            threshold=thd,
                            linkage=link)

        log_name = f'{lm_name}_{layer}_{link}_{thd:.5f}'
        macro_f1, micro_f1, pairwise_f1 = experiment(_run, model, log_name, layer)
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


def experiment(_run, model, log_name, num_layers):
    ent_outputs, rel_outputs = model.run(num_layers)

    print("--- Start: evlaluate noun phrases ---")
    evl = Evaluator(ent_outputs, model.gold_ent2cluster)

    print('Macro Precision: {}'.format(evl.macro_precision))
    _run.log_scalar('Macro Precision', evl.macro_precision, log_name)

    print('Macro Recall: {}'.format(evl.macro_recall))
    _run.log_scalar('Macro Recall', evl.macro_recall, log_name)

    print('Macro F1: {}'.format(evl.macro_f1_score))
    _run.log_scalar('Macro F1', evl.macro_f1_score, log_name)

    print('Micro Precision: {}'.format(evl.micro_precision))
    _run.log_scalar('Micro Precision', evl.micro_precision, log_name)

    print('Micro Recall: {}'.format(evl.micro_recall))
    _run.log_scalar('Micro Recall', evl.micro_recall, log_name)

    print('Micro F1: {}'.format(evl.micro_f1_score))
    _run.log_scalar('Micro F1', evl.micro_f1_score, log_name)

    print('Pairwise Precision: {}'.format(evl.pairwise_precision))
    _run.log_scalar('Pairwise Precision', evl.pairwise_precision, log_name)

    print('Pairwise Recall: {}'.format(evl.pairwise_recall))
    _run.log_scalar('Pairwise Recall', evl.pairwise_recall, log_name)

    print('Pairwise F1: {}'.format(evl.pairwise_f1_score))
    _run.log_scalar('Pairwise F1', evl.pairwise_f1_score, log_name)

    print("--- End: evaluate noun phrases ---")

    return evl.macro_f1_score, evl.micro_f1_score, evl.pairwise_f1_score
