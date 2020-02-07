import os
import torch
from transformers import BertTokenizer, BertModel

SAVED_MODEL_PATH = '/tmp/MODEL'
PRETRAINED_BERT_NAME = 'bert-large-uncased'


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
