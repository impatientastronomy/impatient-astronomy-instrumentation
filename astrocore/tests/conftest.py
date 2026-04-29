"""
Shared pytest fixtures and options for the astrocore test suite.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--lib",
        action="store",
        default=None,
        help="Path to libASICamera2 shared library for hardware integration tests",
    )


@pytest.fixture
def lib_path(request):
    import os
    return request.config.getoption("--lib") or os.environ.get("ASI_LIB")
