# coding: utf-8
#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES).
#
# Copyright (c) 2018-2024 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory,
# National Technology & Engineering Solutions of Sandia, LLC, Carnegie Mellon
# University, West Virginia University Research Corporation, et al.
# All rights reserved.  Please see the files COPYRIGHT.md and LICENSE.md
# for full copyright and license information.
#################################################################################
"""__init__.py for idaes module

Set up logging for the idaes module, and import plugins.
"""
# TODO: Missing doc strings
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

import os
import copy
import logging
from typing import Optional, List

from pyomo.common.fileutils import find_library

from . import config
from .ver import __version__  # noqa


def _handle_optional_compat_activation(
    env_var: str = "IDAES_ACTIVATE_V1_COMPAT",
):
    _log = logging.getLogger("idaes_v1_compat")
    found_in_env = os.environ.get(env_var, None)
    if found_in_env:
        _log.warning(
            "Found environment variable %s=%s. Activating IDAES V1 compatibility.",
            env_var,
            found_in_env,
        )
        try:
            # Only need to import this if required
            # pylint: disable=import-outside-toplevel
            from _idaes_v1_compat import activate
        except ImportError:
            _log.error("Required package _idaes_v1_compat not found")
        else:
            activate()


_handle_optional_compat_activation()

_log = logging.getLogger(__name__)

# Standard locations for config file, binary libraries and executables, ...
data_directory, bin_directory, testing_directory = config.get_data_directory()
# To avoid a circular import the config module doesn't import idaes, but
# some functions in the config module that are executed later use this
# these directories are static from here on.
config.data_directory = data_directory
config.bin_directory = bin_directory
config.testing_directory = testing_directory

# Set the path for the global and local config files
if data_directory is not None:
    _global_config_file = os.path.join(data_directory, "idaes.conf")
else:
    _global_config_file = None
_local_config_file = "idaes.conf"

# Create the general IDAES configuration block, with default config
cfg = config._new_idaes_config_block()  # pylint: disable=protected-access
config.reconfig(cfg)
# read global config and overwrite provided config options
config.read_config(_global_config_file, cfg=cfg)
# read local config and overwrite provided config options
config.read_config(_local_config_file, cfg=cfg)

# Setup the environment so solver executables can be run
config.setup_environment(bin_directory, cfg.use_idaes_solvers)

# Debug log for basic testing of the logging config
_log.debug("'idaes' logger debug test")


# TODO: Remove once AMPL bug is fixed
# TODO: https://github.com/ampl/asl/issues/13
# There appears to be a bug in the ASL which causes terminal failures
# if you try to create multiple ASL structs with different external
# functions in the same process. This causes pytest to crash during testing.
# To avoid this, register all known external functions at initialization.
def _ensure_external_functions_libs_in_env(
    ext_funcs: List[str], var_name: str = "AMPLFUNC", sep: str = "\n"
):
    libraries_str = os.environ.get(var_name, "")
    libraries = [lib for lib in libraries_str.split(sep) if lib.strip()]
    for func_name in ext_funcs:
        lib: Optional[str] = find_library(os.path.join(bin_directory, func_name))
        if lib is not None and lib not in libraries:
            libraries.append(lib)
    os.environ[var_name] = sep.join(libraries)


_ensure_external_functions_libs_in_env(
    ["cubic_roots", "general_helmholtz_external", "functions"]
)


def _create_data_dir():
    """Create the IDAES directory to store data files in."""
    config.create_dir(data_directory)


def _create_bin_dir(bd=None):
    """Create the IDAES directory to store executable files in.

    Args:
        bd: alternate binary directory, used for testing
    """
    _create_data_dir()
    if bd is None:
        bd = bin_directory
    config.create_dir(bd)


def _create_testing_dir():
    """Create an idaes testing directory"""
    _create_data_dir()
    config.create_dir(testing_directory)


if data_directory is not None:
    try:
        _create_data_dir()
    except FileNotFoundError:
        pass  # the standard place for this doesn't exist, shouldn't be a show stopper

    try:
        _create_bin_dir()
    except FileNotFoundError:
        pass  # the standard place for this doesn't exist, shouldn't be a show stopper

    try:
        _create_testing_dir()
    except FileNotFoundError:
        pass  # the standard place for this doesn't exist, shouldn't be a show stopper


def reconfig():
    return config.reconfig(cfg)


def read_config(val):
    return config.read_config(val=val, cfg=cfg)


def write_config(path, default=False):
    _cfg = None if default else cfg
    return config.write_config(path=path, cfg=_cfg)


class temporary_config_ctx(object):
    def __enter__(self):
        self.orig_config = copy.deepcopy(cfg)

    def __exit__(self, exc_type, exc_value, traceback):
        global cfg  # pylint: disable=global-statement
        cfg = self.orig_config
        reconfig()
