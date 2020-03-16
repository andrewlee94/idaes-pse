##############################################################################
# Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework) Copyright (c) 2018-2019, by the
# software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia
# University Research Corporation, et al. All rights reserved.
#
# Please see the files COPYRIGHT.txt and LICENSE.txt for full copyright and
# license information, respectively. Both files are also available online
# at the URL "https://github.com/IDAES/idaes-pse".
##############################################################################
"""
Tests for Caprese helper utility functions.
"""

import pytest
from pyomo.environ import (Block, ConcreteModel,  Constraint, Expression,
                           Set, SolverFactory, Var, value, Objective,
                           TransformationFactory, TerminationCondition,
                           Reference)
from pyomo.network import Arc
from pyomo.kernel import ComponentSet
from pyomo.core.expr.visitor import identify_variables
from pyomo.dae.flatten import flatten_dae_variables

from idaes.core import (FlowsheetBlock, MaterialBalanceType, EnergyBalanceType,
        MomentumBalanceType)
from idaes.core.util.model_statistics import (degrees_of_freedom, 
        activated_equalities_generator, unfixed_variables_generator)
from idaes.core.util.initialization import initialize_by_time_element
from idaes.core.util.exceptions import ConfigurationError
from idaes.generic_models.unit_models import CSTR, Mixer, MomentumMixingType
from idaes.dynamic.cappresse.util import *
from idaes.dynamic.cappresse.nmpc import find_comp_in_block
import idaes.logger as idaeslog
from cstr_for_testing import make_model

__author__ = "Robert Parker"


# See if ipopt is available and set up solver
solver_available = SolverFactory('ipopt').available()
if solver_available:
    solver = SolverFactory('ipopt')
    solver.options = {'tol': 1e-6,
                      'mu_init': 1e-8,
                      'bound_push': 1e-8,
                      'halt_on_ampl_error': 'yes'}
else:
    solver = None


# @ pytest something...?
def test_find_comp_in_block():
    m1 = ConcreteModel()

    @m1.Block([1,2,3])
    def b1(b):
        b.v = Var([1,2,3])

    m2 = ConcreteModel()

    @m2.Block([1,2,3])
    def b1(b):
        b.v = Var([1,2,3,4])

    @m2.Block([1,2,3])
    def b2(b):
        b.v = Var([1,2,3])

    v1 = m1.b1[1].v[1]

    assert find_comp_in_block(m2, m1, v1) is m2.b1[1].v[1]

    v2 = m2.b2[1].v[1]
    v3 = m2.b1[3].v[4]

    # These should result in Attribute/KeyErrors
    #find_comp_in_block(m1, m2, v2)
    #find_comp_in_block(m1, m2, v3)
    assert find_comp_in_block(m1, m2, v2, allow_miss=True) is None
    assert find_comp_in_block(m1, m2, v3, allow_miss=True) is None


def test_VarLocator():
    m = ConcreteModel()
    m.v = Var([1,2,3], ['a','b','c'])

    varlist = [Reference(m.v[:,'a']),
               Reference(m.v[:,'b']),
               Reference(m.v[:,'b'])]

    locator = VarLocator('variable', varlist, 0, is_ic=True)

    assert locator.category == 'variable'
    assert locator.container is varlist
    assert locator.location == 0
    assert locator.is_ic == True


def test_copy_values():
    # Define m1
    m1 = ConcreteModel()
    m1.time = Set(initialize=[1,2,3,4,5])

    m1.v1 = Var(m1.time, initialize=1)
    
    @m1.Block(m1.time)
    def blk(b, t):
        b.v2 = Var(initialize=1)

    # Define m2
    m2 = ConcreteModel()
    m2.time = Set(initialize=[1,2,3,4,5])

    m2.v1 = Var(m2.time, initialize=2)
    
    @m2.Block(m2.time)
    def blk(b, t):
        b.v2 = Var(initialize=2)

    ###

    scalar_vars_1, dae_vars_1 = flatten_dae_variables(m1, m1.time)
    scalar_vars_2, dae_vars_2 = flatten_dae_variables(m2, m2.time)

    m2.v1[2].set_value(5)
    m2.blk[2].v2.set_value(5)

    copy_values_at_time(dae_vars_1, dae_vars_2, 1, 2)

    for t in m1.time:
        if t != 1:
            assert m1.v1[t].value == 1
            assert m1.blk[t].v2.value == 1
        else:
            assert m1.v1[t].value == 5
            assert m1.blk[t].v2.value == 5


def test_find_slices_in_model():
    # Define m1
    m1 = ConcreteModel()
    m1.time = Set(initialize=[1,2,3,4,5])

    m1.v1 = Var(m1.time, initialize=1)
    
    @m1.Block(m1.time)
    def blk(b, t):
        b.v2 = Var(initialize=1)

    # Define m2
    m2 = ConcreteModel()
    m2.time = Set(initialize=[1,2,3,4,5])

    m2.v1 = Var(m2.time, initialize=2)
    
    @m2.Block(m2.time)
    def blk(b, t):
        b.v2 = Var(initialize=2)

    ###

    scalar_vars_1, dae_vars_1 = flatten_dae_variables(m1, m1.time)
    scalar_vars_2, dae_vars_2 = flatten_dae_variables(m2, m2.time)

    t0_tgt = m1.time.first()
    locator = {id(var[t0_tgt]): VarLocator('variable', dae_vars_1, i)
                                for i, var in enumerate(dae_vars_1)}

    tgt_slices = find_slices_in_model(m1, m2, locator, dae_vars_2)

    dae_var_set_1 = ComponentSet(dae_vars_1)
    assert len(dae_var_set_1) == len(tgt_slices)
    assert len(tgt_slices) == len(dae_vars_2)
    for i, _slice in enumerate(tgt_slices):
        assert dae_vars_2[i].name == _slice.name
        assert _slice in dae_var_set_1

    
if __name__ == '__main__':
    test_find_comp_in_block()
    test_VarLocator()
    test_copy_values()
    test_find_slices_in_model()

