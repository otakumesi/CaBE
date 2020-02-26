import hydra
from CaBE.experiment import predict


@hydra.main(config_path='../conf.yml', strict=False)
def perform(cfg):
    predict(cfg)


if __name__ == '__main__':
    perform()
