import os
from collections import defaultdict

import numpy as np
import torch
from transformers import BertTokenizer, BertModel

SAVED_MODEL_PATH = '/tmp/MODEL'
PRETRAINED_BERT_NAME = 'bert-base-uncased'


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

    def encode(self, phrases, num_layers=12):
        encoded_phrases = []

        token_ids_per_phrase = []
        for phrase in phrases:
            token_ids_per_phrase.append(
                torch.tensor(self.tokenizer.encode(phrase,
                                                   max_length=7,
                                                   pad_to_max_length=True),
                             dtype=torch.long))

        token_idfs = calc_token_idfs(token_ids_per_phrase)

        for token_ids in token_ids_per_phrase:
            with torch.no_grad():
                _, _, hidden_states = self.model(token_ids.unsqueeze(0))
                encoded_tokens = hidden_states[num_layers].squeeze()
                weights_by_idf = [token_idfs[int(tid)]+1 for tid in token_ids]
                weighted_tensor = torch.tensor(weights_by_idf, dtype=torch.long).view(-1, 1)

                sum_of_weighted_tokens = torch.sum(encoded_tokens * weighted_tensor, axis=1)
                encoded_phrase = sum_of_weighted_tokens / torch.sum(weighted_tensor)
            encoded_phrases.append(encoded_phrase)

        return torch.stack(encoded_phrases, axis=0)


def calc_token_idfs(token_ids_per_phrase):
    num_phrases = len(token_ids_per_phrase)
    token_dfs = defaultdict(int)

    for token_ids in token_ids_per_phrase:
        for token_id in token_ids.unique():
            token_dfs[int(token_id)] += 1

    token_idfs = {k: np.log(num_phrases / v) for k, v in token_dfs.items()}
    return token_idfs


class ElmoEncoder:
    def __init__(self):
        pass

    def encode(self, phrases):
        pass
