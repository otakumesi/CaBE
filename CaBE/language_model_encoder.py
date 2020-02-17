import pickle
import os

import numpy as np
import torch
import hydra
from transformers import BertTokenizer, BertModel
from allennlp.commands.elmo import ElmoEmbedder

ELEM_FILE_PATH = './pkls/elems'
DEFAULT_FILE_PREFIX = 'elem'

SAVED_MODEL_PATH = '/tmp/MODEL'
PRETRAINED_BERT_NAME = 'bert-large-uncased'
PRETRAINED_ELMO_OPTION_URL = "https://s3-us-west-2.amazonaws.com/"\
    "allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/"\
    "elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
PRETRAINED_ELMO_WEIGHT_URL = "https://s3-us-west-2.amazonaws.com/"\
    "allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/"\
    "elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"


class BertEncoder:
    default_max_layer = 24

    def __init__(self, pretrained_name=PRETRAINED_BERT_NAME):
        self.pretrained_name = pretrained_name

        path = f'{SAVED_MODEL_PATH}_{pretrained_name}'
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
        emb_pkl_path = f'{ELEM_FILE_PATH}/{file_prefix}_{self.pretrained_name}.pkl'
        emb_pkl_path = hydra.utils.to_absolute_path(emb_pkl_path)

        if os.path.isfile(emb_pkl_path):
            with open(emb_pkl_path, 'rb') as f:
                return pickle.load(f)[:, num_layer, :]

        token_ids_list = []
        for phrase in phrases:
            token_ids = self.tokenizer.encode(phrase,
                                              max_length=7,
                                              pad_to_max_length=True)
            token_ids_list.append(torch.tensor(token_ids, dtype=torch.long))
        token_ids_list = torch.stack(token_ids_list, axis=0)

        with torch.no_grad():
            _, _, hid_states = self.model(token_ids_list)
            hid_states = torch.stack(hid_states, axis=0).transpose(0, 1)
            enc_phrases_of_layers = torch.mean(hid_states, axis=2)

            with open(emb_pkl_path, 'wb') as f:
                pickle.dump(enc_phrases_of_layers, f,
                            protocol=pickle.HIGHEST_PROTOCOL)
        return enc_phrases_of_layers[:, num_layer, :]


class BertAttentionEncoder:
    default_max_layer = 24

    def __init__(self, pretrained_name=PRETRAINED_BERT_NAME):
        self.pretrained_name = pretrained_name

        path = f'{SAVED_MODEL_PATH}_{pretrained_name}'
        if os.path.exists(path) and os.listdir(path):
            self.model = BertModel.from_pretrained(path,
                                                   output_hidden_states=True)
            self.tokenizer = BertTokenizer.from_pretrained(path)
        else:
            self.model = BertModel.from_pretrained(pretrained_name,
                                                   output_attentions=True)
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_name)

            os.makedirs(path, exist_ok=True)
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)

    def encode(self, phrases, num_layer, file_prefix=DEFAULT_FILE_PREFIX):
        emb_pkl_path = f'{ELEM_FILE_PATH}/attn_{file_prefix}_{self.pretrained_name}.pkl'
        emb_pkl_path = hydra.utils.to_absolute_path(emb_pkl_path)

        if os.path.isfile(emb_pkl_path):
            with open(emb_pkl_path, 'rb') as f:
                return pickle.load(f)[:, num_layer, :]

        token_ids_list = []
        for phrase in phrases:
            token_ids = self.tokenizer.encode(phrase,
                                              max_length=7,
                                              pad_to_max_length=True)
            token_ids_list.append(torch.tensor(token_ids, dtype=torch.long))
        token_ids_list = torch.stack(token_ids_list, axis=0)

        with torch.no_grad():
            _, _, attentions = self.model(token_ids_list)
            attentions = torch.stack(attentions, axis=0).transpose(0, 1)
            mean_attns_of_layers = torch.mean(attentions, axis=2)

            with open(emb_pkl_path, 'wb') as f:
                pickle.dump(mean_attns_of_layers, f,
                            protocol=pickle.HIGHEST_PROTOCOL)
        return mean_attns_of_layers[:, num_layer, :]


class ElmoEncoder:
    default_max_layer = 2

    def __init__(self):
        self.model = ElmoEmbedder(PRETRAINED_ELMO_OPTION_URL,
                                  PRETRAINED_ELMO_WEIGHT_URL)

    def encode(self, phrases, num_layer, file_prefix=DEFAULT_FILE_PREFIX):
        emb_pkl_path = f'{ELEM_FILE_PATH}/{file_prefix}_{self.pretrained_name}.pkl'
        emb_pkl_path = hydra.utils.to_absolute_path(emb_pkl_path)
        if os.path.isfile(emb_pkl_path):
            with open(emb_pkl_path, 'rb') as f:
                encoded_phrases = pickle.load(f)
        else:
            encoded_phrases = self.model.embed_sentence(phrases)
            with open(emb_pkl_path, 'wb') as f:
                pickle.dump(encoded_phrases, f,
                            protocol=pickle.HIGHEST_PROTOCOL)

        return encoded_phrases[num_layer]
