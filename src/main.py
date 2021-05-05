#!/usr/bin/env python

"""Main entry point for this project."""
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace


def __main(args: Namespace):
    print(f"This is the data directory: {args.data_dir}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Main entry point for this project",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/",
        help="path to the directory containing the unprocessed data",
    )

    __main(parser.parse_args())
