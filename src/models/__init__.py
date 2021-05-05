"""
Neural networks for motion prediction.

Copyright ETH Zurich, Manuel Kaufmann
"""

#Importing all python scripts in the models folder
from os.path import dirname, basename, isfile, join
import sys
import glob
import inspect

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]
from . import *

def create_model(config):
    for script in inspect.getmembers(sys.modules[__name__],inspect.ismodule):
        if script[0] in __all__:
            for classes in inspect.getmembers(script[1],inspect.isclass):
                if classes[0] == config.model:
                    return classes[1](config)
    return dummy_model.DummyModel(config)


