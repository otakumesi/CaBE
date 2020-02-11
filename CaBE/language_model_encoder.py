import pickle
import os

import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from allennlp.commands.elmo import ElmoEmbedder

DEFAULT_FILE_PREFIX = './data/language_model'
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

    def encode(self, phrases, num_layer, file_prefix=DEFAULT_FILE_PREFIX):
        encoded_phrases = []

        for phrase in phrases:
            token_ids = torch.tensor(
                self.tokenizer.encode(phrase,
                                      max_length=7,
                                      pad_to_max_length=True),
                dtype=torch.long)

            with torch.no_grad():
                emb_pkl_path = f'{file_prefix}_emb.pkl'
                if os.path.isfile(emb_pkl_path):
                    hidden_states = pickle.load(open(emb_pkl_path, 'rb'))
                else:
                    _, _, hidden_states = self.model(token_ids.unsqueeze(0))
                    pickle.dump(hidden_states, open(emb_pkl_path, 'wb'))

                encoded_tokens = hidden_states[num_layer].squeeze()
                encoded_phrase = torch.mean(encoded_tokens, axis=0)
                encoded_phrases.append(encoded_phrase)
        return torch.stack(encoded_phrases, axis=0)

    @classmethod
    def default_max_layer(cls):
        return 24


class ElmoEncoder:
    def __init__(self):
        self.model = ElmoEmbedder(PRETRAINED_ELMO_OPTION_URL,
                                  PRETRAINED_ELMO_WEIGHT_URL)

    def encode(self, phrases, num_layer, file_prefix=DEFAULT_FILE_PREFIX):
        emb_pkl_path = f'{file_prefix}_emb.pkl'
        if os.path.isfile(emb_pkl_path):
            encoded_phrases = pickle.load(open(emb_pkl_path, 'rb'))
        else:
            encoded_phrases = self.model.embed_sentence(phrases)

            pickle.dump(encoded_phrases, open(emb_pkl_path, 'wb'))

        return encoded_phrases[num_layer]

    @classmethod
    def default_max_layer(cls):
        return 2
