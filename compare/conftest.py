"""Shared fixtures. Run with the py2 interpreter: python -I -m pytest compare"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "_build"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest  # noqa: E402

import hklrs  # noqa: E402


@pytest.fixture(scope="session", params=sorted(hklrs.presets()))
def preset(request):
    return request.param
