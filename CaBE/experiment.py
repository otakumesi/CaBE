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

THRESHOLDS = list(range(0, 1, 0.1))
LINKAGES = ['single', 'complete', 'average']


@ex.config
def experiment_config():
    model= BERT()
    name = 'CaBE - reverb45K'
    file_name = DEFAULT_REVERB_PATH
    threshold = .25
    linkage = 'single'


@ex.main
def experiment_main(_run, name, model, file_name, threshold, linkage, thresholds):
    cabe = CaBE(name=name,
                model=model,
                file_name=file_name,
                distance_threshold=threshold)
    ent_outputs, rel_outputs = cabe.run()

    print("--- Start: evlaluate noun phrases ---")
    evl = np_evaluate(ent_outputs, cabe.gold_ent2cluster)
    _run.log_scalar('Macro Precision', evl.macro_precision(), threshold)
    _run.log_scalar('Macro Recall', evl.macro_recall(), threshold)
    _run.log_scalar('Micro Precision', evl.micro_precision(), threshold)
    _run.log_scalar('Micro Recall', evl.micro_recall(), threshold)
    _run.log_scalar('Pairwise Precision', evl.pairwise_precision(), threshold)
    _run.log_scalar('Pairwise Recall', evl.pairwise_recall(), threshold)
    print("--- End: evaluate noun phrases ---")
