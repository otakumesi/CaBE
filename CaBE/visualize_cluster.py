import hydra
from CaBE.experiment import visualize_cluster


@hydra.main(config_path='../conf.yml', strict=False)
def perform(cfg):
    visualize_cluster(cfg)


if __name__ == '__main__':
    perform()
