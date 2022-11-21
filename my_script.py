from datetime import datetime
import time
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Test your model')
    parser.add_argument("--ext-name", type=str, default="x3d_m", help='name of extractor model')
    parser.add_argument("--block-type", type=str, default="tcl3d", help='type of block', 
                        choices=["tcl3d", "tcl", "trl", "linear", "trl-half", "masked"])
    parser.add_argument("--epochs", type=int, default=200, help='Max number of epochs to train')
    parser.add_argument("--bias-rank", type=int, default=4, help='bias rank in TCL')
    parser.add_argument("--emb-dim", type=str, default="32,8,8", help='GRU embedding dim')
    parser.add_argument("--hid-dim", type=str, default="16,4,4", help='GRU hidden dim')
    parser.add_argument("--dryrun", action="store_true", help="Make test run")
    parser.add_argument("--experiments-name", type=str, default="explosion", help='name of dataset', choices=["explosion", "road_accidents"])
    return parser

if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()
    # print(args)
    time.sleep(2)
    timestamp = datetime.now().strftime("%y%m%dT%H%M%S")
    print(timestamp)