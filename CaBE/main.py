from argparse import ArgumentParser
from bert_serving.client import BertClient

import CaBE

DEFAULT_REVERB_PATH = './data/reverb45k_test'


if __name__ == '__main__':
    parser = ArgumentParser(
        description="CaBE: Canonicalizion Open Knowledge Bases with BERT")
    parser.add_argument('-name', dest='name', default=None,
                        help='Give an output file name')
    parser.add_argument('-file', dest='file', default=DEFAULT_REVERB_PATH,
                        help='Give an input file name')
    args = parser.parse_args()

    CaBE(name=args.name, model=BertClient(), file_name=args.file).run()
