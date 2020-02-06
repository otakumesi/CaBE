from itertools import product
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver
from CaBE.model import CaBE
from CaBE.evaluator import Evaluator
from CaBE.language_model_encoder import BERTEncoder


DEFAULT_REVERB_PATH = './data/reverb45k_test'
DEFAULT_LOG_PATH = './log'
ex = Experiment('CaBE Expriment')
ex.observers.append(FileStorageObserver(DEFAULT_LOG_PATH))

THRESHOLDS = np.arange(0.0000, 0.00003, 0.00001)
LINKAGES = ['single', 'complete', 'average']
LMS = {'BERT': BERTEncoder, 'Elmo': None}
LAYERS = range(1, 12)


@ex.config
def experiment_config():
    name = 'CaBE - reverb45K'
    lm_name = 'BERT'
    file_name = DEFAULT_REVERB_PATH
    threshold = .0000
    linkage = 'complete'
    num_layers = 12
    tune = False


@ex.main
def experiment_main(_run, name, lm_name, file_name, threshold, linkage, num_layers, tune):
    lang_model = LMS[lm_name]()
    if not tune:
        experiment_proc(_run, name, lang_model, file_name, threshold, linkage, num_layers)
    else:
        clustering_configs = product(THRESHOLDS, LINKAGES, LAYERS)
        results = {}
        for thd, link, layer in clustering_configs:
            macro_f1, micro_f1, pairwise_f1 = experiment_proc(_run, name, lang_model, file_name, thd, link, layer)
            results[(link, thd)] = (macro_f1, micro_f1, pairwise_f1)

        sorted_configs = sorted(results.items(), key=lambda kv: np.mean(kv[1]))
        for conf, f1s in sorted_configs:
            print(f'{conf[0]}, {conf[1]:.5f}: {f1s[0]:.4f}, {f1s[1]:.4f}, {f1s[2]:.4f}')


def experiment_proc(_run, name, lang_model, file_name, threshold, linkage, num_layers):
    model = CaBE(name=name,
                 model=lang_model,
                 file_name=file_name,
                 distance_threshold=threshold,
                 linkage=linkage)
    ent_outputs, rel_outputs = model.run(num_layers)

    print("--- Start: evlaluate noun phrases ---")
    evl = Evaluator(ent_outputs, model.gold_ent2cluster)

    step_name = f'{linkage}_{threshold}'

    print('Macro Precision: {}'.format(evl.macro_precision))
    _run.log_scalar('Macro Precision', evl.macro_precision, step_name)

    print('Macro Recall: {}'.format(evl.macro_recall))
    _run.log_scalar('Macro Recall', evl.macro_recall, step_name)

    print('Macro F1: {}'.format(evl.macro_f1_score))
    _run.log_scalar('Macro F1', evl.macro_f1_score, step_name)

    print('Micro Precision: {}'.format(evl.micro_precision))
    _run.log_scalar('Micro Precision', evl.micro_precision, step_name)

    print('Micro Recall: {}'.format(evl.micro_recall))
    _run.log_scalar('Micro Recall', evl.micro_recall, step_name)

    print('Micro F1: {}'.format(evl.micro_f1_score))
    _run.log_scalar('Micro F1', evl.micro_f1_score, step_name)

    print('Pairwise Precision: {}'.format(evl.pairwise_precision))
    _run.log_scalar('Pairwise Precision', evl.pairwise_precision, step_name)

    print('Pairwise Recall: {}'.format(evl.pairwise_recall))
    _run.log_scalar('Pairwise Recall', evl.pairwise_recall, step_name)

    print('Pairwise F1: {}'.format(evl.pairwise_f1_score))
    _run.log_scalar('Pairwise F1', evl.pairwise_f1_score, step_name)

    print("--- End: evaluate noun phrases ---")

    return evl.macro_f1_score, evl.micro_f1_score, evl.pairwise_f1_score
