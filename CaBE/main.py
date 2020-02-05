from argparse import ArgumentParser
from experiment import ex


if __name__ == '__main__':
    parser = ArgumentParser(
        description="CaBE: Canonicalizion Open Knowledge Bases with BERT")
    parser.add_argument('-name', dest='name', default=None,
                        help='Give an output file name')
    parser.add_argument('-file', dest='file', default=None,
                        help='Give an input file name')
    parser.add_argument('-threshold', dest='threshold', default=.25,
                        help='Give an threshold of clustering')

    args = parser.parse_args()

    config = {"name": args.name,
              "file_name": args.file,
              "threshold": args.threshold}
    config = {k: v for k, v in config.items() if v is not None}
    ex.run(config_updates=config)
