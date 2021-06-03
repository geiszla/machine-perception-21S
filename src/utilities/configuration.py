"""
Define common constants that are used globally.

Also define a configuration object whose parameters can be set via the command line or loaded from
an existing json file. Here you can add more configuration parameters that should be exposed via
the command line. In the code, you can access them via `config.your_parameter`. All parameters are
automatically saved to disk in JSON format.

Copyright ETH Zurich, Manuel Kaufmann
"""
import argparse
import json
import os
import pprint

import torch


class Constants(object):
    """Singleton."""

    class __Constants:
        def __init__(self):
            # Environment setup.
            self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.DTYPE = torch.float32
            self.DATA_DIR = os.environ["MP_DATA"]
            self.EXPERIMENT_DIR = os.environ["MP_EXPERIMENTS"]
            self.METRIC_TARGET_LENGTHS = [5, 10, 19, 24]  # @ 60 fps, in ms: 83.3, 166.7, 316.7, 400

    instance = None

    def __new__(cls, *args, **kwargs):
        """Create a new instance of constants."""
        if not Constants.instance:
            Constants.instance = Constants.__Constants()
        return Constants.instance

    def __getattr__(self, item):
        """Get a constant."""
        return getattr(self.instance, item)

    def __setattr__(self, key, value):
        """Set a constant."""
        return setattr(self.instance, key, value)


CONSTANTS = Constants()


class Configuration(object):
    """Configuration parameters exposed via the commandline."""

    def __init__(self, adict):
        """Create a new configuration object."""
        self.__dict__.update(adict)

    def __str__(self):
        """Print the configuration."""
        return pprint.pformat(vars(self), indent=4)

    @staticmethod
    def parse_cmd():
        """Parse command line arguments."""
        parser = argparse.ArgumentParser()

        # General.
        parser.add_argument(
            "--data_workers",
            type=int,
            default=4,
            help="Number of parallel threads for data loading.",
        )
        parser.add_argument(
            "--print_every",
            type=int,
            default=200,
            help="Print stats to console every so many iters.",
        )
        parser.add_argument(
            "--eval_every",
            type=int,
            default=400,
            help="Evaluate validation set every so many iters.",
        )
        parser.add_argument("--tag", default="", help="A custom tag for this experiment.")
        parser.add_argument("--seed", type=int, default=None, help="Random number generator seed.")

        # Data.
        parser.add_argument(
            "--seed_seq_len", type=int, default=120, help="Number of frames for the seed length."
        )
        parser.add_argument(
            "--target_seq_len", type=int, default=24, help="How many frames to predict."
        )

        # Learning configurations.
        parser.add_argument("--model", default="DummyModel", help="Model file name.")
        parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
        parser.add_argument("--n_epochs", type=int, default=50, help="Number of epochs.")
        parser.add_argument(
            "--bs_train", type=int, default=16, help="Batch size for the training set."
        )
        parser.add_argument(
            "--bs_eval", type=int, default=16, help="Batch size for valid/test set."
        )
        parser.add_argument("--train_on_val", type=bool, default=False, help="Train on both train and val set.")
        parser.add_argument("--eval_on_val", type=bool, default=False, help="Evaluate on validation set")

        # Model parameters.

        # GCN.
        parser.add_argument("--gcn_dct_n", type=int, default=70, help="GCN number of DCT coef.")
        parser.add_argument("--gcn_h", type=int, default=256, help="GCN hidden size.")
        parser.add_argument("--gcn_p_dropout", type=float, default=0.5, help="GCN dropout prob.")

        config = parser.parse_args()
        return Configuration(vars(config))

    @staticmethod
    def from_json(json_path):
        """Load configurations from a JSON file."""
        with open(json_path, "r") as f:
            config = json.load(f)
            return Configuration(config)

    def to_json(self, json_path):
        """Dump configurations to a JSON file."""
        with open(json_path, "w") as f:
            s = json.dumps(vars(self), indent=2, sort_keys=True)
            f.write(s)
