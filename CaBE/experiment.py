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
    name = 'CaBE - reverb45K'
    file_name = DEFAULT_REVERB_PATH
    threshold = .25
    linkage = 'single'


@ex.main
def experiment_main(_run, name, model, file_name, threshold, linkage):
    cabe = CaBE(name=name,
                model=model,
                file_name=file_name,
                distance_threshold=threshold)
    ent_outputs, rel_outputs = cabe.run()

    print("--- Start: evlaluate noun phrases ---")
    evl = Evaluator(ent_outputs, cabe.gold_ent2cluster)

    macro_precision = evl.macro_precision()
    print('Macro Precision: {}'.format(macro_precision))
    _run.log_scalar('Macro Precision', macro_precision, threshold)

    macro_recall = evl.macro_recall()
    print('Macro Recall: {}'.format(macro_recall))
    _run.log_scalar('Macro Recall', macro_recall, threshold)

    macro_f1 = evl.macro_f1_score()
    print('Macro F1: {}'.format(macro_f1))
    _run.log_scalar('Macro F1', macro_f1, threshold)

    micro_precision = evl.micro_precision()
    print('Micro Precision: {}'.format(micro_precision))
    _run.log_scalar('Micro Precision', micro_precision, threshold)

    micro_recall = evl.micro_recall()
    print('Micro Recall: {}'.format(micro_recall))
    _run.log_scalar('Micro Recall', micro_recall, threshold)

    micro_f1 = evl.micro_f1_score()
    print('Micro F1: {}'.format(micro_f1))
    _run.log_scalar('Micro F1', micro_f1, threshold)

    pairwise_precision = evl.pairwise_precision()
    print('Pairwise Precision: {}'.format(pairwise_precision))
    _run.log_scalar('Pairwise Precision', pairwise_precision, threshold)

    pairwise_recall = evl.pairwise_recall()
    print('Pairwise Recall: {}'.format(pairwise_recall))
    _run.log_scalar('Pairwise Recall', pairwise_recall, threshold)

    pairwise_f1 = evl.pairwise_f1_score()
    print('Pairwise F1: {}'.format(pairwise_f1))
    _run.log_scalar('Pairwise F1', pairwise_f1, threshold)

    print("--- End: evaluate noun phrases ---")
