import os
import torch
from transformers import BertTokenizer, BertModel

SAVED_MODEL_PATH = '/tmp/BERT_MODEL'
PRETRAINED_NAME = 'bert-large-uncased'


class BertEncoder:
    def __init__(self):
        path = f'{SAVED_MODEL_PATH}-{PRETRAINED_NAME}'
        if os.path.exists(path) and os.listdir(path):
            self.model = BertModel.from_pretrained(SAVED_MODEL_PATH, output_hidden_states=True)
            self.tokenizer = BertTokenizer.from_pretrained(SAVED_MODEL_PATH)
        else:
            self.model = BertModel.from_pretrained(PRETRAINED_NAME, output_hidden_states=True)
            self.tokenizer = BertTokenizer.from_pretrained(PRETRAINED_NAME)

            os.makedirs(SAVED_MODEL_PATH, exist_ok=True)
            self.model.save_pretrained(SAVED_MODEL_PATH)
            self.tokenizer.save_pretrained(SAVED_MODEL_PATH)

    def encode(self, phrases, num_layers=12):
        encoded_phrases = []
        for phrase in phrases:
            input_ids = torch.tensor([self.tokenizer.encode(phrase, max_length=7, pad_to_max_length=True)])
            with torch.no_grad():
                _, _, hidden_states = self.model(input_ids)
                encoded_tokens = hidden_states[num_layers].squeeze()
                encoded_phrase = torch.mean(encoded_tokens, 1)
            encoded_phrases.append(encoded_phrase)

        return torch.stack(encoded_phrases, axis=0)


class ElmoEncoder:
    def __init__(self):
        pass

    def encode(self, phrases):
        pass
