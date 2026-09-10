import importlib
import pickle

import packaging

from setuptools import Distribution


def test_reimport_extern():
    packaging2 = importlib.import_module(packaging.__name__)
    assert packaging is packaging2


def test_distribution_picklable():
    pickle.loads(pickle.dumps(Distribution()))
