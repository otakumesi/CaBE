from argparse import ArgumentParser
from CaBE.experiment import ex


if __name__ == '__main__':
    parser = ArgumentParser(
        description="CaBE: Canonicalizion Open Knowledge Bases with BERT")
    parser.add_argument('-name', dest='name', default=None,
                        help='Give an output file name')
    parser.add_argument('-file', dest='file', default=None,
                        help='Give an input file name')
    parser.add_argument('-threshold', dest='threshold',
                        default=None, type=float,
                        help='Give an threshold of clustering')
    parser.add_argument('-linkage', dest='linkage', default=None,
                        help='Give an threshold of clustering')
    parser.add_argument('-tune', dest='tune',
                        default=False, type=bool,
                        help='Whether this is hyper parameter tuning')

    args = parser.parse_args()

    config = {"name": args.name,
              "file_name": args.file,
              "threshold": args.threshold,
              "linkage": args.linkage,
              "tune": args.tune}
    config = {k: v for k, v in config.items() if v is not None}
    ex.run(config_updates=config)
