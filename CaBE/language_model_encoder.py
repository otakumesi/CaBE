import pickle
import os

import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from allennlp.commands.elmo import ElmoEmbedder

from CaBE.helper import get_abspath

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

        self.model, self.tokenizer = _init_pretrained_model(BertModel,
                                                            BertTokenizer,
                                                            pretrained_name)

    def encode(self, data, num_layer, file_prefix=DEFAULT_FILE_PREFIX):
        emb_pkl_path = embed_elements_path(file_prefix, self.pretrained_name)

        if os.path.isfile(emb_pkl_path):
            with open(emb_pkl_path, 'rb') as f:
                ents, rels = pickle.load(f)
                return ents[:, num_layer, :], rels[:, num_layer, :]

        token_ids_list = []
        token_lens_list = []
        for phrases in data.triples:
            tokens_triple = [self.tokenizer.tokenize(phrase) for phrase in phrases]
            token_lens_list.append([len(tokens) for tokens in tokens_triple])
            token_ids = self.tokenizer.convert_tokens_to_ids(sum(tokens_triple, []))
            token_ids_list.append(torch.tensor(token_ids, dtype=torch.long))

        ent_of_layers = []
        rel_of_layers = []
        model_prepared_list = zip(token_ids_list, token_lens_list)
        with torch.no_grad():
            for token_ids, (sbj_l, rel_l, obj_l) in model_prepared_list:
                _, _, hid_states = self.model(token_ids.unsqueeze(0))
                hid_states = torch.stack(hid_states, axis=0).squeeze(1)

                sbj_states = hid_states[:, :sbj_l, :]
                ent_of_layers.append(torch.mean(sbj_states, axis=1))

                rel_l_padded = sbj_l + rel_l
                rel_states = hid_states[:, sbj_l:rel_l_padded, :]
                rel_of_layers.append(torch.mean(rel_states, axis=1))

                obj_l_paddded = rel_l_padded + obj_l
                obj_states = hid_states[:, rel_l:obj_l_paddded, :]
                ent_of_layers.append(torch.mean(obj_states, axis=1))

        ent_of_layers = torch.stack(ent_of_layers, axis=0)
        rel_of_layers = torch.stack(rel_of_layers, axis=0)

        with open(emb_pkl_path, 'wb') as f:
            pickle.dump((ent_of_layers, rel_of_layers), f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        return ent_of_layers[:, num_layer, :], rel_of_layers[:, num_layer, :]


class ElmoEncoder:
    default_max_layer = 2

    def __init__(self):
        self.model = ElmoEmbedder(PRETRAINED_ELMO_OPTION_URL,
                                  PRETRAINED_ELMO_WEIGHT_URL)

    def encode(self, data, num_layer, file_prefix=DEFAULT_FILE_PREFIX):
        emb_pkl_path = embed_elements_path(file_prefix, self.pretrained_name)

        if os.path.isfile(emb_pkl_path):
            with open(emb_pkl_path, 'rb') as f:
                entities, relations = pickle.load(f)
        else:

            entities = self.model.embed_sentence(data.entities)
            relations = self.model.embed_sentence(data.relations)
            with open(emb_pkl_path, 'wb') as f:
                pickle.dump((entities, relations), f,
                            protocol=pickle.HIGHEST_PROTOCOL)

        return entities[num_layer], relations[num_layer]


def embed_elements_path(file_prefix, pretrained_name):
    return get_abspath(f'{ELEM_FILE_PATH}/{file_prefix}_{pretrained_name}.pkl')


def _init_pretrained_model(model, tokenizer, pretrained_name):
    save_path = f'{SAVED_MODEL_PATH}_{pretrained_name}'
    if os.path.exists(save_path) and os.listdir(save_path):
        model = BertModel.from_pretrained(save_path,
                                          output_hidden_states=True)
        tokenizer = BertTokenizer.from_pretrained(save_path)
    else:
        model = BertModel.from_pretrained(pretrained_name,
                                          output_hidden_states=True)
        tokenizer = BertTokenizer.from_pretrained(pretrained_name)

        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
    return model, tokenizer
