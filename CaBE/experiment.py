from sacred import Experiment
from sacred.observers import FileStorageObserver

from bert_serving.client import BertClient

from CaBE import CaBE
from evaluator import np_evaluate


DEFAULT_REVERB_PATH = './data/reverb45k_test'
DEFAULT_LOG_PATH = './log'
ex = Experiment('CaBE Expriment')
ex.observers.append(FileStorageObserver(DEFAULT_LOG_PATH))


@ex.config
def experiment_config():
    model= BertClient()
    name = 'CaBE - reverb45K'
    file_name = DEFAULT_REVERB_PATH
    threshold = .25


@ex.capture
def experiment_proc(name, model, file_name, threshold):

    cabe = CaBE(name=name,
                model=model,
                file_name=file_name,
                distance_threshold=threshold)
    ent_outputs, rel_outputs = cabe.run()

    print("--- Start: evlaluate noun phrases ---")
    np_evaluate(ent_outputs, cabe.gold_ent2cluster)
    print("--- End: evaluate noun phrases ---")


@ex.main
def experiment_main():
    experiment_proc()
