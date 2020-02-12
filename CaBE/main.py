import hydra
from CaBE.experiment import predict, grid_search


@hydra.main(config_path='../conf.yml', strict=False)
def perform(cfg):
    exec(f'{cfg.task}(cfg)')


if __name__ == '__main__':
    perform()
