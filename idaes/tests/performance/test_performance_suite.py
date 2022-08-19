#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES), and is copyright (c) 2018-2021
# by the software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia University
# Research Corporation, et al.  All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and
# license information.
#################################################################################
"""
Author: Andrew Lee
"""
import pytest
import gc

import pyomo.common.unittest as unittest
from pyomo.util.check_units import assert_units_consistent
from pyomo.common.timing import TicTocTimer

from idaes.core.solvers import get_solver

from idaes.models.properties.modular_properties.examples.tests.test_HC_PR import (
    HC_PR_Model,
)
from idaes.models.unit_models.tests.test_heat_exchanger_1D import HX1D_Model
