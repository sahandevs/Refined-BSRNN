#!/usr/bin/env python3

import argparse
from .train import init_arg_parser as train_init_parser, do_train
from .infer import init_arg_parser as infer_init_parser, do_infer
import utils


def main():
    parser = argparse.ArgumentParser(prog="bsrnn")
    parser.add_argument("--device", type=str, default="auto", help="torch.device")
    parser.add_argument(
        "--seed", type=int, default=42, help="seed random generators. set -1 for random"
    )

    subparsers = parser.add_subparsers(help="commands")
    TRAIN_CMD, train_parser = train_init_parser(subparsers)
    INFER_CMD, infer_parser = infer_init_parser(subparsers)

    args = parser.parse_args()
    utils.GLOBAL_CONFIG = utils.GLOBAL_CONFIG(args)

    command = args["subparser_name"]
    if command == TRAIN_CMD:
        do_train(train_parser)
    if command == INFER_CMD:
        do_infer(infer_parser)


if __name__ == "__main__":
    main()
