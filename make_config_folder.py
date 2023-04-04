import yaml
from pathlib import Path
import argparse
import pandas as pd

config_fields = [
    'rnn_type', 'block_type', 'input_block', 'output_block', 'bias_rank',
    'emb_dim', 'hid_dim', 'rnn_ranks', 'output_ranks'
]  # 'input_ranks'


def get_args():
    parser = argparse.ArgumentParser(description='Create configs')
    parser.add_argument("experiments_name", type=str, help='Experiment name')
    parser.add_argument("config_template",
                        type=str,
                        help='yaml file with config template')
    parser.add_argument("config_folder",
                        type=str,
                        help='folder with config files')
    parser.add_argument("selected_configs",
                        type=str,
                        help='csv with selected configurations')
    return parser.parse_args()


def main(args):

    with Path(args.config_template).open("r") as f:
        config = yaml.safe_load(f)

    df = pd.read_csv(args.selected_configs, index_col=0, header=0)

    output_folder = Path(args.config_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    for i, (r, row) in enumerate(df.iterrows()):
        for label in ["emb_dim", "hid_dim"]:
            rl = row[label]
            if "," not in rl:
                row[label] = int(rl)

        config["train"].update(
            {k: {"value": v}
             for k, v in row[config_fields].items()})

        config["train"]["experiments_name"]["value"] = args.experiments_name

        for label_block, label_ranks in zip(
            ["block_type", "output_block"],
            ["rnn_ranks", "output_ranks"]):
            # "input_block"

            if config["train"][label_block]['value'] != "trl3dhalf":
                config["train"].pop(label_ranks, None)
            elif pd.isna(config["train"][label_ranks]['value']):
                config["train"].pop(label_ranks, None)
                                
        with (output_folder / f'config_bce_best_{r}_{i}.yaml').open("w") as f:
            yaml.dump(config, f)


if __name__ == "__main__":

    args = get_args()
    main(args)
