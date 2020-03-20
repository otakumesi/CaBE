from pathlib import Path
from multiprocessing import Pool

import json
from fastavro import reader
from fastavro.schema import load_schema

def join_tokens(tokens):
    return ' '.join([token['word'] for token in tokens])

def join_norm_phrases(tokens):
    return ' '.join([(token['lemma'] or token['word']).lower() for token in tokens])

def read_avro(file_name):
    dataset = []
    with open(file_name, 'rb') as f:
        for article in reader(f):
            sbj = article['subject']
            sbj_phrase = join_tokens(sbj)
            sbj_norm = join_norm_phrases(sbj)

            rel = article['relation']
            rel_phrase = join_tokens(rel)
            rel_norm = join_norm_phrases(rel)

            obj = article['object']
            obj_phrase = join_tokens(obj)
            obj_norm = join_norm_phrases(obj)

            sent_token_data = article['sentence_linked']['tokens']
            src_sentence = ' '.join([token['word'] for token in sent_token_data])

            triple_record = {
                'id': article['triple_id'],
                'triple': [sbj_phrase, rel_phrase, obj_phrase],
                'triple_norm': [sbj_norm, rel_norm, obj_norm],
                'true_link': {'subject': sbj[0]['w_link']['wiki_link'], 'object': obj[0]['w_link']['wiki_link']},
                'src_sentences': [src_sentence]
            }
        dataset.append(triple_record)
    return dataset

def build_datasets(args):
    i, file_name = args
    dataset = read_avro(file_name)
    with open(f'{OUTPUT_FOLDER}/triples-{i}.json', 'w') as f:
        json.dump(dataset, f)


AVRO_SCHEMA_FILE = './avroschema/WikiArticleLinkedNLP.avsc'
AVRO_FOLDER = './data/OPIEC-Linked-triples/'
OUTPUT_FOLDER = './data/opiec-for-canonical/'

avro_folder = Path(AVRO_FOLDER)
AVRO_FILES = avro_folder.glob('*.avro')

schema = load_schema(AVRO_SCHEMA_FILE)

with Pool(10) as p:
    p.map(build_datasets, enumerate(AVRO_FILES))
