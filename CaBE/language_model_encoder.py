import pickle
import os

import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel

from allennlp.commands.elmo import ElmoEmbedder
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer

from CaBE.helper import get_abspath

ELEM_FILE_PATH = './pkls/elems'
DEFAULT_FILE_PREFIX = 'elem'

SAVED_MODEL_PATH = '/tmp/MODEL'
PRETRAINED_BERT_NAME = 'bert-large-uncased'
PRETRAINED_ROBERTA_NAME = 'roberta-large'

PRETRAINED_ELMO_OPTION_URL = "https://s3-us-west-2.amazonaws.com/"\
    "allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/"\
    "elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
PRETRAINED_ELMO_WEIGHT_URL = "https://s3-us-west-2.amazonaws.com/"\
    "allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/"\
    "elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"


class CLMEncoder:
    def __init__(self, pretrained_name, model_cls, tokenizer_cls):
        self.pretrained_name = pretrained_name

        self.model, self.tokenizer = _init_pretrained_model(pretrained_name,
                                                            model_cls,
                                                            tokenizer_cls)

    def encode(self, data, file_prefix=DEFAULT_FILE_PREFIX):
        emb_pkl_path = embed_elements_path(file_prefix, self.pretrained_name)

        if os.path.isfile(emb_pkl_path):
            with open(emb_pkl_path, 'rb') as f:
                ent_embs, rel_embs = pickle.load(f)
                return ent_embs, rel_embs

        token_ids_list = []
        token_pos_list = []
        for phrases in data.triples:
            tokens_triple = [self.tokenizer.tokenize(phrase) for phrase in phrases]
            token_pos_list.append([len(tokens) for tokens in tokens_triple])
            token_ids = self.tokenizer.convert_tokens_to_ids(sum(tokens_triple, []))
            token_ids = self.tokenizer.prepare_for_model(token_ids, max_length=20, pad_to_max_length=True)['input_ids']
            token_ids_list.append(torch.tensor(token_ids, dtype=torch.long))
        token_ids_list = torch.stack(token_ids_list, axis=0)
        ent_embs = []
        rel_embs = []
        with torch.no_grad():
            _, _, hid_states = self.model(token_ids_list)
            hid_states = torch.stack(hid_states, axis=0).transpose(0, 1)

        hid_states_with_pos = zip(hid_states, token_pos_list)

        for hid_states, (sbj_p, rel_p, obj_p) in hid_states_with_pos:
            sbj_states = hid_states[:, :sbj_p, :]
            ent_embs.append(torch.mean(sbj_states, axis=1))

            rel_p_padded = sbj_p + rel_p
            rel_states = hid_states[:, sbj_p:rel_p_padded, :]
            rel_embs.append(torch.mean(rel_states, axis=1))

            obj_p_padded = rel_p_padded + obj_p
            obj_states = hid_states[:, rel_p_padded:obj_p_padded, :]
            ent_embs.append(torch.mean(obj_states, axis=1))

        ent_embs = torch.stack(ent_embs, axis=0)
        rel_embs = torch.stack(rel_embs, axis=0)

        with open(emb_pkl_path, 'wb') as f:
            pickle.dump((ent_embs, rel_embs), f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        return ent_embs, rel_embs


class BertEncoder(CLMEncoder):
    default_max_layer = 24

    def __init__(self, pretrained_name=PRETRAINED_BERT_NAME):
        super().__init__(pretrained_name, BertModel, BertTokenizer)


class RobertaEncoder(CLMEncoder):
    default_max_layer = 24

    def __init__(self, pretrained_name=PRETRAINED_ROBERTA_NAME):
        super().__init__(pretrained_name, RobertaModel, RobertaTokenizer)


class ElmoEncoder:
    default_max_layer = 2

    def __init__(self):
        self.model = ElmoEmbedder(PRETRAINED_ELMO_OPTION_URL,
                                  PRETRAINED_ELMO_WEIGHT_URL)
        self.pretrained_name = 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights'
        self.tokenizer = WordTokenizer()

    def encode(self, data, file_prefix=DEFAULT_FILE_PREFIX):
        emb_pkl_path = embed_elements_path(file_prefix, self.pretrained_name)

        if os.path.isfile(emb_pkl_path):
            with open(emb_pkl_path, 'rb') as f:
                entities, relations = pickle.load(f)
                return entities, relations

        ent_embs = []
        rel_embs = []

        for triple in data.triples:
            sent_tokens = self.tokenizer.batch_tokenize(triple)
            sbj_p, rel_p, obj_p = [len(tokens) for tokens in sent_tokens]
            tokens = [token.text for token in sum(sent_tokens, [])]
            hid_states = self.model.embed_sentence(tokens)
            hid_states = torch.from_numpy(hid_states)

            sbj_states = hid_states[:, :sbj_p, :]
            ent_embs.append(torch.mean(sbj_states, axis=1))

            rel_p_padded = sbj_p + rel_p
            rel_states = hid_states[:, sbj_p:rel_p_padded, :]
            rel_embs.append(torch.mean(rel_states, axis=1))

            obj_p_padded = rel_p_padded + obj_p
            obj_states = hid_states[:, rel_p_padded:obj_p_padded, :]
            ent_embs.append(torch.mean(obj_states, axis=1))

        ent_embs = torch.stack(ent_embs, axis=0)
        rel_embs = torch.stack(rel_embs, axis=0)

        with open(emb_pkl_path, 'wb') as f:
            pickle.dump((ent_embs, rel_embs), f,
                        protocol=pickle.HIGHEST_PROTOCOL)

        return ent_embs, rel_embs


def embed_elements_path(file_prefix, pretrained_name):
    return get_abspath(f'{ELEM_FILE_PATH}/{file_prefix}_{pretrained_name}.pkl')


def _init_pretrained_model(pretrained_name, model_cls, tokenizer_cls):
    print(pretrained_name, model_cls, tokenizer_cls)
    save_path = f'{SAVED_MODEL_PATH}_{pretrained_name}'
    if os.path.exists(save_path) and os.listdir(save_path):
        model = model_cls.from_pretrained(save_path,
                                          output_hidden_states=True)
        tokenizer = tokenizer_cls.from_pretrained(save_path)
    else:
        model = model_cls.from_pretrained(pretrained_name,
                                          output_hidden_states=True)
        tokenizer = tokenizer_cls.from_pretrained(pretrained_name)

        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
    return model, tokenizer
