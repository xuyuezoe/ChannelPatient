import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "api: tests that call the LLM relay (skipped by default)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("-m") and "api" in config.getoption("-m"):
        return
    skip = pytest.mark.skip(reason="api tests skipped by default (use -m api)")
    for it in items:
        if "api" in it.keywords:
            it.add_marker(skip)
