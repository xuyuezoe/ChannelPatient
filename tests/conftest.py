import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "api: tests that call the LLM relay (skipped by default)")
    config.addinivalue_line("markers", "gpu: tests that need a GPU and model weights (skipped by default)")


def pytest_collection_modifyitems(config, items):
    expr = config.getoption("-m") or ""
    want_api = "api" in expr and "not api" not in expr
    want_gpu = "gpu" in expr and "not gpu" not in expr
    skip = pytest.mark.skip(reason="api tests skipped by default (use -m api)")
    skip_gpu = pytest.mark.skip(reason="gpu tests skipped by default (use -m gpu)")
    for it in items:
        if "api" in it.keywords and not want_api:
            it.add_marker(skip)
        if "gpu" in it.keywords and not want_gpu:
            it.add_marker(skip_gpu)
