"""
Neural networks for motion prediction.

Copyright ETH Zurich, Manuel Kaufmann
"""

import glob
import importlib
import inspect
from os.path import basename, dirname, isfile, join

from models.dummy_model import DummyModel

scripts = glob.glob(join(dirname(__file__), "*.py"))
scripts = [
    "models." + basename(f)[:-3] for f in scripts if isfile(f) and not basename(f).startswith("_")
]


def create_model(config):
    """Return an instance of the model class with the same name as config.model."""
    print(scripts)
    for script in scripts:
        for classes in inspect.getmembers(importlib.import_module(script), inspect.isclass):
            if classes[0] == config.model:
                return classes[1](config)
    return DummyModel(config)  # noqa F405
