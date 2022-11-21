from collections import defaultdict
from itertools import product
from typing import Dict
import yaml
from pathlib import Path
from subprocess import PIPE, Popen
import pickle
import json
import argparse
from termcolor import cprint

def get_args():
    parser = argparse.ArgumentParser(description='Test your model')
    parser.add_argument("config_file", type=str, help='timestamp to be processed')
    return parser.parse_args()

def load_config(file_name: str):

    with Path(file_name).open("r") as f:
        config = yaml.safe_load(f)


    key_lists = defaultdict(dict)
    for typ in ["train", "test"]:
        for key, value in config[typ].items():
            if "value" in value:
                config[typ][key] = value["value"]
            else:
                key_lists[typ][key] = value["values"]
                config[typ][key] = value["values"][0]

    return config, key_lists


def main(config: Dict):

    train_cmd = ["python3", "script_bce.py"]
    for key, value in config["train"].items():
        train_cmd += [f'--{key.replace("_", "-")}', str(value)]

    cprint("Train command:\n" + " ".join(train_cmd), "red")
    with Popen(train_cmd, stdout=PIPE) as proc:
        output, _ = proc.communicate()
        # print(output.decode("utf-8").splitlines())
        timestamp = output.decode("utf-8").splitlines()[-1]

    
    test_cmd = ["python3", "script_test_bce.py", timestamp,
                "--experiments-name", config["train"]["experiments_name"],
                "--threshold-number", str(config["test"]["threshold_number"])]

    cprint("Test command:\n" + " ".join(test_cmd), "red")
    with Popen(test_cmd, stdout=PIPE) as proc:
        output, _ = proc.communicate()
        # print(output.decode("utf-8"))


    temp_file = Path(f'saves/temp_{timestamp}.pickle')
    with temp_file.open("rb") as f:
        results = pickle.load(f)

    logging_info = {"timestamp": timestamp}
    logging_info.update(config["train"])
    logging_info.update(config["test"])

    with Path("saves/results/log_bce.txt").open("a") as f:
        for result in results:
            logging_info.update(result)
            # FIXME fix bug with `Object of type int32 is not JSON serializable` in logging_info["best_conf_matrix"]
            logging_info["best_conf_matrix"] = [int(i) for i in logging_info["best_conf_matrix"]]
            json.dump(logging_info, f)
            f.write("\n")

    temp_file.unlink()



if __name__ == "__main__":

    args = get_args()
    config, key_lists = load_config(args.config_file)
    # print(config)

    keys_train = list(key_lists["train"].keys())
    keys_test = list(key_lists["test"].keys())
    values_train = list(product(*key_lists["train"].values()))
    values_test = list(product(*key_lists["test"].values()))

    # print(keys_train, values_train)
    # print(keys_test, values_test)

    update_train, update_test = len(keys_train) > 0, len(keys_test) > 0
    if not update_train:
        values_train = [None]
    if not update_test:
        values_test = [None]
    if (not update_train) and (not update_test):
        main(config)
    else:
        for value_train in values_train:
            if update_train:
                config["train"].update(dict(zip(keys_train, value_train)))
            for value_test in values_test:
                if update_test:
                    config["test"].update(dict(zip(keys_test, value_test)))

                print(config)
                main(config)