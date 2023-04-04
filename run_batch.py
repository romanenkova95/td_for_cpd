from pathlib import Path
import argparse
from termcolor import cprint
import subprocess


def get_args():
    parser = argparse.ArgumentParser(description='Launch batch of experiments with configs from given folder')
    parser.add_argument("config_folder",
                        type=str,
                        help='folder with configs')
    return parser.parse_args()


def main(args):

    config_folder = Path(args.config_folder)
    for file in config_folder.iterdir():
        if file.name.startswith("passed"):
            continue

        cprint(f'Start training and evaluating config: {file.name}', "yellow")
        subprocess.run(["python3", "run.py", str(file)])
        file.rename(config_folder / f'passed_{file.name}')


if __name__ == "__main__":

    args = get_args()
    main(args)