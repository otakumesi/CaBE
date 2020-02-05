import os
import torch
from transformers import BertTokenizer, BertModel

SAVED_MODEL_PATH = '/tmp/BERT_MODEL'


class BERT:
    def __init__(self):
        if os.path.exists(SAVED_MODEL_PATH) and os.listdir(SAVED_MODEL_PATH):
            self.model = BertModel.from_pretrained(SAVED_MODEL_PATH)
            self.tokenizer = BertTokenizer.from_pretrained(SAVED_MODEL_PATH)
        else:
            self.model = BertModel.from_pretrained('bert-base-uncased')
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

            os.makedirs(SAVED_MODEL_PATH, exist_ok=True)
            self.model.save_pretrained(SAVED_MODEL_PATH)
            self.tokenizer.save_pretrained(SAVED_MODEL_PATH)

    def encode(self, phrases):
        input_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(phrases)]).reshape(-1, 1)
        with torch.no_grad():
            return self.model(input_ids)[0].squeeze()
