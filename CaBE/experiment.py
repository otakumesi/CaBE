from sacred import Experiment
from sacred.observers import FileStorageObserver
from CaBE import CaBE
from evaluator import np_evaluate
from BERT import BERT


DEFAULT_REVERB_PATH = './data/reverb45k_test'
DEFAULT_LOG_PATH = './log'
ex = Experiment('CaBE Expriment')
ex.observers.append(FileStorageObserver(DEFAULT_LOG_PATH))


@ex.config
def experiment_config():
    model= BERT()
    name = 'CaBE - reverb45K'
    file_name = DEFAULT_REVERB_PATH
    threshold = .25


@ex.main
def experiment_main(_run, name, model, file_name, threshold):
    cabe = CaBE(name=name,
                model=model,
                file_name=file_name,
                distance_threshold=threshold)
    ent_outputs, rel_outputs = cabe.run()

    print("--- Start: evlaluate noun phrases ---")
    evl = np_evaluate(ent_outputs, cabe.gold_ent2cluster)
    _run.log_scalar('Macro Precision', evl.macro_precision())
    _run.log_scalar('Macro Recall', evl.macro_recall())
    _run.log_scalar('Micro Precision', evl.micro_precision())
    _run.log_scalar('Micro Recall', evl.micro_recall())
    _run.log_scalar('Pairwise Precision', evl.pairwise_precision())
    _run.log_scalar('Pairwise Recall', evl.pairwise_recall())
    print("--- End: evaluate noun phrases ---")
