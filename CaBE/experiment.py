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

THRESHOLDS = np.arange(0.0, 1.0, 0.1)
LINKAGES = ['single', 'complete', 'average']


@ex.config
def experiment_config():
    model= BERT()
    model_name = 'CaBE - reverb45K'
    file_name = DEFAULT_REVERB_PATH
    threshold = .25
    linkage = 'single'
    tune: False


@ex.main
def experiment_main(_run, name, model_name, file_name, threshold, linkage, tune):
    if not tune:
        experiment_proc(_run, name, model_name, file_name, threshold, linkage)
    else:
        clustering_configs = product(THRESHOLDS, LINKAGES)
        print(clustering_configs)
        for thd, link in clustering_configs:
            experiment_proc(_run, name, model_name, file_name, thd, link)


def experiment_proc(_run, name, model_name, file_name, threshold, linkage):
    model = CaBE(name=name,
                 model=model_name,
                 file_name=file_name,
                 distance_threshold=threshold,
                 linkage=linkage)
    ent_outputs, rel_outputs = model.run()

    print("--- Start: evlaluate noun phrases ---")
    evl = Evaluator(ent_outputs, model.gold_ent2cluster)

    print('Macro Precision: {}'.format(evl.macro_precision))
    _run.log_scalar('Macro Precision', evl.macro_precision, threshold)

    print('Macro Recall: {}'.format(evl.macro_recall))
    _run.log_scalar('Macro Recall', evl.macro_recall, threshold)

    print('Macro F1: {}'.format(evl.macro_f1_score))
    _run.log_scalar('Macro F1', evl.macro_f1_score, threshold)

    print('Micro Precision: {}'.format(evl.micro_precision))
    _run.log_scalar('Micro Precision', evl.micro_precision, threshold)

    print('Micro Recall: {}'.format(evl.micro_recall))
    _run.log_scalar('Micro Recall', evl.micro_recall, threshold)

    print('Micro F1: {}'.format(evl.micro_f1_score))
    _run.log_scalar('Micro F1', evl.micro_f1_score, threshold)

    print('Pairwise Precision: {}'.format(evl.pairwise_precision))
    _run.log_scalar('Pairwise Precision', evl.pairwise_precision, threshold)

    print('Pairwise Recall: {}'.format(evl.pairwise_recall))
    _run.log_scalar('Pairwise Recall', evl.pairwise_recall, threshold)

    print('Pairwise F1: {}'.format(evl.pairwise_f1_score))
    _run.log_scalar('Pairwise F1', evl.pairwise_f1_score, threshold)

    print("--- End: evaluate noun phrases ---")
