import hydra
from CaBE.experiment import grid_search


@hydra.main(config_path='../conf.yml', strict=False)
def perform(cfg):
    grid_search(cfg)


if __name__ == '__main__':
    perform()
