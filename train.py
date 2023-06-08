import hydra
from omegaconf import DictConfig, OmegaConf
from src.datamodules.components import datasets

OmegaConf.register_new_resolver("eval", eval)


@hydra.main(config_path="configs/", config_name="train.yaml")
def main(config: DictConfig):
    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src import utils
    from src.training_pipeline import train

    # Applies optional utilities
    utils.extras(config)

    # Train model
    return train(config)


if __name__ == "__main__":
    main()
