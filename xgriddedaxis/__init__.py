#!/usr/bin/env python
""" Top-level module for xgriddedaxis. """
from pkg_resources import DistributionNotFound, get_distribution

from .remapper import Remapper  # noqa: F401

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    __version__ = '0.0.0'
