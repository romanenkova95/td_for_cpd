import hydra
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("eval", eval)


@hydra.main(config_path="configs/", config_name="test.yaml")
def main(config: DictConfig):
    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src import utils
    from src.testing_pipeline import test

    # Applies optional utilities
    utils.extras(config)

    # Train model
    return test(config)


if __name__ == "__main__":
    main()
