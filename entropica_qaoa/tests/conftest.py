import pytest

import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../../')


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption("--notebooks", action="store_true", default=False,
                     help="run notebook tests")
    parser.addoption("--all", action="store_true", default=False,
                     help="run all tests")


def pytest_collection_modifyitems(config, items):
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    skip_nbs = pytest.mark.skip(reason="need --notebooks option to run")
    skip_default = pytest.mark.skip(reason="Added --notebooks or --runslow option")

    if config.getoption("--all"):
        return

    for item in items:
        if config.getoption("--runslow"):
            if "slow" not in item.keywords:
                item.add_marker(skip_default)

        elif config.getoption("--notebooks"):
            if "notebook" not in item.keywords:
                item.add_marker(skip_default)

        else:
            if "notebook" in item.keywords:
                item.add_marker(skip_nbs)
            elif "slow" in item.keywords:
                item.add_marker(skip_slow)
