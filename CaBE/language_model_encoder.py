import os
from collections import defaultdict

import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from allennlp.commands.elmo import ElmoEmbedder

SAVED_MODEL_PATH = '/tmp/MODEL'
PRETRAINED_BERT_NAME = 'bert-large-uncased'
PRETRAINED_ELMO_OPTION_URL = "https://s3-us-west-2.amazonaws.com/"\
    "allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/"\
    "elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
PRETRAINED_ELMO_WEIGHT_URL = "https://s3-us-west-2.amazonaws.com/"\
    "allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/"\
    "elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"


class BertEncoder:
    def __init__(self, pretrained_name=PRETRAINED_BERT_NAME):
        path = f'{SAVED_MODEL_PATH}-{pretrained_name}'
        if os.path.exists(path) and os.listdir(path):
            self.model = BertModel.from_pretrained(path,
                                                   output_hidden_states=True)
            self.tokenizer = BertTokenizer.from_pretrained(path)
        else:
            self.model = BertModel.from_pretrained(pretrained_name,
                                                   output_hidden_states=True)
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_name)

            os.makedirs(path, exist_ok=True)
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)

    def encode(self, phrases, num_layer=12):
        encoded_phrases = []

        for phrase in phrases:
            token_ids = torch.tensor(
                self.tokenizer.encode(phrase,
                                      max_length=7,
                                      pad_to_max_length=True),
                dtype=torch.long)

            with torch.no_grad():
                _, _, hidden_states = self.model(token_ids.unsqueeze(0))
                # TODO: 隠れ層をそもそもdumpするようにしてしまう？
                encoded_phrase = hidden_states[num_layer].squeeze()
                encoded_phrases.append(encoded_phrase)
        return torch.stack(encoded_phrases, axis=0)

    @classmethod
    def default_max_layer(cls):
        return 12


class ElmoEncoder:
    def __init__(self):
        self.model = ElmoEmbedder(PRETRAINED_ELMO_OPTION_URL,
                                  PRETRAINED_ELMO_WEIGHT_URL)

    def encode(self, phrases, num_layer=2):
        encoded_phrases = self.model.embed_sentence(phrases)
        return encoded_phrases[num_layer]

    @classmethod
    def default_max_layer(cls):
        return 2
