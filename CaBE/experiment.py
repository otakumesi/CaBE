from itertools import product
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver
from CaBE.model import CaBE
from CaBE.evaluator import Evaluator
from CaBE.BERT import BERT


DEFAULT_REVERB_PATH = './data/reverb45k_test'
DEFAULT_LOG_PATH = './log'
ex = Experiment('CaBE Expriment')
ex.observers.append(FileStorageObserver(DEFAULT_LOG_PATH))

THRESHOLDS = np.arange(0.0001, 0.00015, 0.00001)
LINKAGES = ['single', 'complete', 'average']


@ex.config
def experiment_config():
    name = 'CaBE - reverb45K'
    lm_model= BERT()
    file_name = DEFAULT_REVERB_PATH
    threshold = .0003
    linkage = 'complete'
    tune: False


@ex.main
def experiment_main(_run, name, lm_model, file_name, threshold, linkage, tune):
    if not tune:
        experiment_proc(_run, name, lm_model, file_name, threshold, linkage)
    else:
        clustering_configs = product(THRESHOLDS, LINKAGES)
        for thd, link in clustering_configs:
            experiment_proc(_run, name, lm_model, file_name, thd, link)


def experiment_proc(_run, name, lm_model, file_name, threshold, linkage):
    model = CaBE(name=name,
                 model=lm_model,
                 file_name=file_name,
                 distance_threshold=threshold,
                 linkage=linkage)
    ent_outputs, rel_outputs = model.run()

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
