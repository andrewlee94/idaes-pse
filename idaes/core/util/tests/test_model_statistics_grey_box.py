#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES).
#
# Copyright (c) 2018-2026 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory,
# National Technology & Engineering Solutions of Sandia, LLC, Carnegie Mellon
# University, West Virginia University Research Corporation, et al.
# All rights reserved.  Please see the files COPYRIGHT.md and LICENSE.md
# for full copyright and license information.
#################################################################################
"""
This module contains tests for the model statistics functions with the
presence of grey-box components.
"""

import pytest

import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
from pyomo.contrib.pynumero.interfaces.tests.external_grey_box_models import PressureDropTwoEqualitiesTwoOutputsWithHessian

from idaes.core.util.model_statistics import *


@pytest.fixture
def model():
    m = pyo.ConcreteModel()

    # Add a Block containing a Grey Box model, with linking variables and constraints
    m.b1 = pyo.Block()
    m.b1.egb = ExternalGreyBoxBlock()
    m.b1.egb.set_external_model(PressureDropTwoEqualitiesTwoOutputsWithHessian(), build_implicit_constraint_objects=True)

    # Add Vars and linking constraints to m
    m.b1.Pin = pyo.Var(initialize=101325, bounds=(0, 1e8))  # Not near bounds
    m.b1.c = pyo.Var(initialize=1e-8, bounds=(0, 1))  # Near bounds
    m.b1.F = pyo.Var()
    m.b1.P1 = pyo.Var()
    m.b1.P3 = pyo.Var()
    m.b1.P2 = pyo.Var()
    m.b1.Pout = pyo.Var()

    m.b1.link_Pin = pyo.Constraint(expr=m.b1.Pin == m.b1.egb.inputs['Pin'])
    m.b1.link_c = pyo.Constraint(expr=m.b1.c == m.b1.egb.inputs['c'])
    m.b1.link_F = pyo.Constraint(expr=m.b1.F == m.b1.egb.inputs['F'])
    m.b1.link_P1 = pyo.Constraint(expr=m.b1.P1 == m.b1.egb.inputs['P1'])
    m.b1.link_P3 = pyo.Constraint(expr=m.b1.P3 == m.b1.egb.inputs['P3'])
    m.b1.link_P2 = pyo.Constraint(expr=m.b1.P2 == m.b1.egb.outputs['P2'])
    m.b1.link_Pout = pyo.Constraint(expr=m.b1.Pout == m.b1.egb.outputs['Pout'])

    # Add a second, unrelated Block to ensure that the model statistics functions are correctly
    m.b2 = pyo.Block()
    m.b2.v1 = pyo.Var()
    m.b2.c1 = pyo.Constraint(expr=m.b2.v1 == 1)

    # Add two inequalities to confirm behaviour
    m.b1.ineq = pyo.Constraint(expr=m.b1.Pin >= 0)
    m.b2.ineq = pyo.Constraint(expr=m.b2.v1 >= 0)

    # Set some values and bounds in the grey box to confirm that grey box variables are included in the variable counts
    # Include both inputs and outputs to confirm that both are included in the variable counts
    m.b1.egb.inputs['Pin'].set_value(101325)
    m.b1.egb.inputs['Pin'].setlb(0)
    m.b1.egb.inputs['Pin'].setub(1e8)
    m.b1.egb.inputs['c'].set_value(1e-8)
    m.b1.egb.inputs['c'].setlb(0)
    m.b1.egb.inputs['c'].setub(1)
    m.b1.egb.outputs['P2'].set_value(90000)
    m.b1.egb.outputs['P2'].setlb(90000)
    m.b1.egb.outputs['P2'].setub(1e8)

    return m


class TestBlockStatisticsGreyBox:
    @pytest.mark.unit
    def test_total_blocks_set_w_grey_box(self, model):
        # Test that the total_blocks_set function correctly counts the number of blocks in the model
        # Grey Box is not included as a normal block
        assert len(total_blocks_set(model)) == 3
        for b in total_blocks_set(model):
            assert b in [model, model.b1, model.b2]

    @pytest.mark.unit
    def test_number_total_blocks_w_grey_box(self, model):
        assert number_total_blocks(model) == 3

    @pytest.mark.unit
    def test_activated_blocks_set_w_grey_box(self, model):
        # Test that the activated_blocks_set function correctly counts the number of activated blocks in the model
        # Grey Box is not included as a normal block
        assert len(activated_blocks_set(model)) == 3
        for b in activated_blocks_set(model):
            assert b in [model, model.b1, model.b2]
        
        # Deactivate b1 and test again
        model.b1.deactivate()
        assert len(activated_blocks_set(model)) == 2
        for b in activated_blocks_set(model):
            assert b in [model, model.b2]

    @pytest.mark.unit
    def test_greybox_block_set_w_grey_box(self, model):
        # Test that the grey_box_set function correctly identifies the Grey Box block in the model
        gbs = greybox_block_set(model)
        assert len(gbs) == 1
        assert model.b1.egb in gbs

    @pytest.mark.unit
    def test_activated_greybox_block_set_w_grey_box(self, model):
        # Test that the activated_greybox_block_set function correctly identifies the activated Grey Box block in the model
        agbs = activated_greybox_block_set(model)
        assert len(agbs) == 1
        assert model.b1.egb in agbs

        # Deactivate b1 and test again
        model.b1.deactivate()
        agbs = activated_greybox_block_set(model)
        assert len(agbs) == 0

    @pytest.mark.unit
    def test_deactivated_greybox_block_set_w_grey_box(self, model):
        # Test that the deactivated_greybox_block_set function correctly identifies the deactivated Grey Box block in the model
        dgb = deactivated_greybox_block_set(model)
        assert len(dgb) == 0

        # Deactivate the grey box and test again
        model.b1.egb.deactivate()
        dgb = deactivated_greybox_block_set(model)
        assert len(dgb) == 1
        assert model.b1.egb in dgb
    
    @pytest.mark.unit
    def test_number_deactivated_greybox_block_w_grey_box(self, model):
        # Test that the number_deactivated_greybox_block function correctly counts the number of deactivated Grey Box blocks in the model
        assert number_deactivated_greybox_block(model) == 0

        # Deactivate the grey box and test again
        model.b1.egb.deactivate()
        assert number_deactivated_greybox_block(model) == 1

    @pytest.mark.unit
    def test_number_greybox_blocks_w_grey_box(self, model):
        # Test that the number_greybox_blocks function correctly counts the number of Grey Box blocks in the model
        assert number_greybox_blocks(model) == 1

        # Deactivate the grey box and test again (should not change the count)
        model.b1.egb.deactivate()
        assert number_greybox_blocks(model) == 1

    @pytest.mark.unit
    def test_number_activated_greybox_blocks_w_grey_box(self, model):
        # Test that the number_activated_greybox_blocks function correctly counts the number of activated Grey Box blocks in the model
        assert number_activated_greybox_blocks(model) == 1

        # Deactivate the grey box and test again
        model.b1.egb.deactivate()
        assert number_activated_greybox_blocks(model) == 0

    @pytest.mark.unit
    def test_number_activated_blocks_w_grey_box(self, model):
        # Test that the number_activated_blocks function correctly counts the number of activated blocks in the model
        assert number_activated_blocks(model) == 3

        # Deactivate b2 and test again
        model.b2.deactivate()
        assert number_activated_blocks(model) == 2

    @pytest.mark.unit
    def test_deactivated_blocks_set_w_grey_box(self, model):
        # Test that the deactivated_blocks_set function correctly identifies the deactivated blocks in the model
        dbs = deactivated_blocks_set(model)
        assert len(dbs) == 0

        # Deactivate b2 and test again
        model.b2.deactivate()
        dbs = deactivated_blocks_set(model)
        assert len(dbs) == 1
        assert model.b2 in dbs

    @pytest.mark.unit
    def test_number_deactivated_blocks_w_grey_box(self, model):
        # Test that the number_deactivated_blocks function correctly counts the number of deactivated blocks in the model
        assert number_deactivated_blocks(model) == 0

        # Deactivate b2 and test again
        model.b2.deactivate()
        assert number_deactivated_blocks(model) == 1


class TestConstraintStatisticsGreyBox:
    @pytest.mark.unit
    def test_total_constraints_set_w_grey_box(self, model):
        # Test that the total_constraints_set function correctly counts the number of constraints in the model
        # First, test with include_greybox = False
        tcs = total_constraints_set(model, include_greybox=False)
        assert len(tcs) == 10
        for c in tcs:
            assert c in [
                model.b1.link_Pin,
                model.b1.link_c,
                model.b1.link_F,
                model.b1.link_P1,
                model.b1.link_P3,
                model.b1.link_P2,
                model.b1.link_Pout,
                model.b2.c1,
                model.b1.ineq,
                model.b2.ineq,
            ]
        
        # Next, test with include_greybox = True
        tcs = total_constraints_set(model, include_greybox=True)
        assert len(tcs) == 14
        for c in tcs:
            assert c in [
                model.b1.link_Pin,
                model.b1.link_c,
                model.b1.link_F,
                model.b1.link_P1,
                model.b1.link_P3,
                model.b1.link_P2,
                model.b1.link_Pout,
                model.b2.c1,
                model.b1.egb.P2_constraint,
                model.b1.egb.Pout_constraint,
                model.b1.egb.pdrop1,
                model.b1.egb.pdrop3,
                model.b1.ineq,
                model.b2.ineq,
            ]

    @pytest.mark.unit
    def test_number_total_constraints_w_grey_box(self, model):
        # Test that the number_total_constraints function correctly counts the number of constraints in the model
        # First, test with include_greybox = False
        assert number_total_constraints(model, include_greybox=False) == 10

        # Next, test with include_greybox = True
        assert number_total_constraints(model, include_greybox=True) == 14

    @pytest.mark.unit
    def test_activated_constraints_generator_w_grey_box(self, model):
        # Test that the activated_constraints_generator function correctly identifies the activated constraints in the model
        # First, test with include_greybox = False
        acg = activated_constraints_generator(model, include_greybox=False)
        acg_list = list(acg)
        assert len(acg_list) == 10
        for c in acg_list:
            assert c in [
                model.b1.link_Pin,
                model.b1.link_c,
                model.b1.link_F,
                model.b1.link_P1,
                model.b1.link_P3,
                model.b1.link_P2,
                model.b1.link_Pout,
                model.b2.c1,
                model.b1.ineq,
                model.b2.ineq,
            ]
        
        # Next, test with include_greybox = True
        acg = activated_constraints_generator(model, include_greybox=True)
        acg_list = list(acg)
        assert len(acg_list) == 14
        for c in acg_list:
            assert c in [
                model.b1.link_Pin,
                model.b1.link_c,
                model.b1.link_F,
                model.b1.link_P1,
                model.b1.link_P3,
                model.b1.link_P2,
                model.b1.link_Pout,
                model.b2.c1,
                model.b1.egb.P2_constraint,
                model.b1.egb.Pout_constraint,
                model.b1.egb.pdrop1,
                model.b1.egb.pdrop3,
                model.b1.ineq,
                model.b2.ineq,
            ]
        
        # Now deactivate the grey box and test again with include_greybox = True (should not include grey box constraints)
        model.b1.egb.deactivate()
        acg = activated_constraints_generator(model, include_greybox=True)
        acg_list = list(acg)
        assert len(acg_list) == 10
        for c in acg_list:
            assert c in [
                model.b1.link_Pin,
                model.b1.link_c,
                model.b1.link_F,
                model.b1.link_P1,
                model.b1.link_P3,
                model.b1.link_P2,
                model.b1.link_Pout,
                model.b2.c1,
                model.b1.ineq,
                model.b2.ineq,
            ]
    
    @pytest.mark.unit
    def test_activated_constraints_set_w_grey_box(self, model):
        # Test that the activated_constraints_set function correctly identifies the activated constraints in the model
        # First, test with include_greybox = False
        acs = activated_constraints_set(model, include_greybox=False)
        assert len(acs) == 10
        for c in acs:
            assert c in [
                model.b1.link_Pin,
                model.b1.link_c,
                model.b1.link_F,
                model.b1.link_P1,
                model.b1.link_P3,
                model.b1.link_P2,
                model.b1.link_Pout,
                model.b2.c1,
                model.b1.ineq,
                model.b2.ineq,
            ]
        
        # Next, test with include_greybox = True
        acs = activated_constraints_set(model, include_greybox=True)
        assert len(acs) == 14
        for c in acs:
            assert c in [
                model.b1.link_Pin,
                model.b1.link_c,
                model.b1.link_F,
                model.b1.link_P1,
                model.b1.link_P3,
                model.b1.link_P2,
                model.b1.link_Pout,
                model.b2.c1,
                model.b1.egb.P2_constraint,
                model.b1.egb.Pout_constraint,
                model.b1.egb.pdrop1,
                model.b1.egb.pdrop3,
                model.b1.ineq,
                model.b2.ineq,
            ]
        
        # Now deactivate the grey box and test again with include_greybox = True (should not include grey box constraints)
        model.b1.egb.deactivate()
        acs = activated_constraints_set(model, include_greybox=True)
        assert len(acs) == 10
        for c in acs:
            assert c in [
                model.b1.link_Pin,
                model.b1.link_c,
                model.b1.link_F,
                model.b1.link_P1,
                model.b1.link_P3,
                model.b1.link_P2,
                model.b1.link_Pout,
                model.b2.c1,
                model.b1.ineq,
                model.b2.ineq,
            ]
    
    @pytest.mark.unit
    def test_number_activated_constraints_w_grey_box(self, model):
        # Test that the number_activated_constraints function correctly counts the number of activated constraints in the model
        # First, test with include_greybox = False
        assert number_activated_constraints(model, include_greybox=False) == 10

        # Next, test with include_greybox = True
        assert number_activated_constraints(model, include_greybox=True) == 14

        # Now deactivate the grey box and test again with include_greybox = True (should not include grey box constraints)
        model.b1.egb.deactivate()
        assert number_activated_constraints(model, include_greybox=True) == 10

    @pytest.mark.unit
    def test_deactivated_constraints_generator_w_grey_box(self, model):
        # Test that the deactivated_constraints_generator function correctly identifies the deactivated constraints in the model
        # First, test with include_greybox = False
        dcg = deactivated_constraints_generator(model, include_greybox=False)
        dcg_list = list(dcg)
        assert len(dcg_list) == 0
        
        # Next, test with include_greybox = True
        dcg = deactivated_constraints_generator(model, include_greybox=True)
        dcg_list = list(dcg)
        assert len(dcg_list) == 0
        
        # Now deactivate the grey box and test again with include_greybox = True (should include grey box constraints)
        model.b1.egb.deactivate()
        dcg = deactivated_constraints_generator(model, include_greybox=True)
        dcg_list = list(dcg)
        assert len(dcg_list) == 4
        for c in dcg_list:
            assert c in [
                model.b1.egb.P2_constraint,
                model.b1.egb.Pout_constraint,
                model.b1.egb.pdrop1,
                model.b1.egb.pdrop3,
            ]

    @pytest.mark.unit
    def test_deactivated_constraints_set_w_grey_box(self, model):
        # Test that the deactivated_constraints_set function correctly identifies the deactivated constraints in the model
        # First, test with include_greybox = False
        dcs = deactivated_constraints_set(model, include_greybox=False)
        assert len(dcs) == 0
        
        # Next, test with include_greybox = True
        dcs = deactivated_constraints_set(model, include_greybox=True)
        assert len(dcs) == 0
        
        # Now deactivate the grey box and test again with include_greybox = True (should include grey box constraints)
        model.b1.egb.deactivate()
        dcs = deactivated_constraints_set(model, include_greybox=True)
        assert len(dcs) == 4
        for c in dcs:
            assert c in [
                model.b1.egb.P2_constraint,
                model.b1.egb.Pout_constraint,
                model.b1.egb.pdrop1,
                model.b1.egb.pdrop3,
            ]
    
    @pytest.mark.unit
    def test_number_deactivated_constraints_w_grey_box(self, model):
        # Test that the number_deactivated_constraints function correctly counts the number of deactivated constraints in the model
        # First, test with include_greybox = False
        assert number_deactivated_constraints(model, include_greybox=False) == 0

        # Next, test with include_greybox = True
        assert number_deactivated_constraints(model, include_greybox=True) == 0

        # Now deactivate the grey box and test again with include_greybox = True (should include grey box constraints)
        model.b1.egb.deactivate()
        assert number_deactivated_constraints(model, include_greybox=True) == 4

    @pytest.mark.unit
    def test_total_equalities_generator_w_grey_box(self, model):
        # Test that the total_equalities_generator function correctly identifies the equality constraints in the model
        # First, test with include_greybox = False
        teg = total_equalities_generator(model, include_greybox=False)
        teg_list = list(teg)
        assert len(teg_list) == 8
        for c in teg_list:
            assert c in [
                model.b1.link_Pin,
                model.b1.link_c,
                model.b1.link_F,
                model.b1.link_P1,
                model.b1.link_P3,
                model.b1.link_P2,
                model.b1.link_Pout,
                model.b2.c1,
            ]
        
        # Next, test with include_greybox = True
        teg = total_equalities_generator(model, include_greybox=True)
        teg_list = list(teg)
        assert len(teg_list) == 12
        for c in teg_list:
            assert c in [
                model.b1.link_Pin,
                model.b1.link_c,
                model.b1.link_F,
                model.b1.link_P1,
                model.b1.link_P3,
                model.b1.link_P2,
                model.b1.link_Pout,
                model.b1.egb.P2_constraint,
                model.b1.egb.Pout_constraint,
                model.b1.egb.pdrop1,
                model.b1.egb.pdrop3,
                model.b2.c1,
            ]

    @pytest.mark.unit
    def test_total_equalities_set_w_grey_box(self, model):
        # Test that the total_equalities_set function correctly identifies the equality constraints in the model
        # First, test with include_greybox = False
        tes = total_equalities_set(model, include_greybox=False)
        assert len(tes) == 8
        for c in tes:
            assert c in [
                model.b1.link_Pin,
                model.b1.link_c,
                model.b1.link_F,
                model.b1.link_P1,
                model.b1.link_P3,
                model.b1.link_P2,
                model.b1.link_Pout,
                model.b2.c1,
            ]
        
        # Next, test with include_greybox = True
        tes = total_equalities_set(model, include_greybox=True)
        assert len(tes) == 12
        for c in tes:
            assert c in [
                model.b1.link_Pin,
                model.b1.link_c,
                model.b1.link_F,
                model.b1.link_P1,
                model.b1.link_P3,
                model.b1.link_P2,
                model.b1.link_Pout,
                model.b2.c1,
                model.b1.egb.P2_constraint,
                model.b1.egb.Pout_constraint,
                model.b1.egb.pdrop1,
                model.b1.egb.pdrop3,
            ]
        
        # Now deactivate the grey box and test again with include_greybox = True
        # Greybox constraints should still be included in the count as we are not checking active
        model.b1.egb.deactivate()
        tes = total_equalities_set(model, include_greybox=True)
        assert len(tes) == 12
        for c in tes:
            assert c in [
                model.b1.link_Pin,
                model.b1.link_c,
                model.b1.link_F,
                model.b1.link_P1,
                model.b1.link_P3,
                model.b1.link_P2,
                model.b1.link_Pout,
                model.b2.c1,
                model.b1.egb.P2_constraint,
                model.b1.egb.Pout_constraint,
                model.b1.egb.pdrop1,
                model.b1.egb.pdrop3,
            ]
    
    @pytest.mark.unit
    def test_number_total_equalities_w_grey_box(self, model):
        # Test that the number_total_equalities function correctly counts the number of equality constraints in the model
        # First, test with include_greybox = False
        assert number_total_equalities(model, include_greybox=False) == 8

        # Next, test with include_greybox = True
        assert number_total_equalities(model, include_greybox=True) == 12

        # Now deactivate the grey box and test again with include_greybox = True
        # Greybox constraints should still be included in the count as we are not checking active
        model.b1.egb.deactivate()
        assert number_total_equalities(model, include_greybox=True) == 12

    @pytest.mark.unit
    def test_activated_equalities_generator_w_grey_box(self, model):
        # Test that the activated_equalities_generator function correctly identifies the activated equality constraints in the model
        # First, test with include_greybox = False
        aeg = activated_equalities_generator(model, include_greybox=False)
        aeg_list = list(aeg)
        assert len(aeg_list) == 8
        for c in aeg_list:
            assert c in [
                model.b1.link_Pin,
                model.b1.link_c,
                model.b1.link_F,
                model.b1.link_P1,
                model.b1.link_P3,
                model.b1.link_P2,
                model.b1.link_Pout,
                model.b2.c1,
            ]
        
        # Next, test with include_greybox = True
        aeg = activated_equalities_generator(model, include_greybox=True)
        aeg_list = list(aeg)
        assert len(aeg_list) == 12
        for c in aeg_list:
            assert c in [
                model.b1.link_Pin,
                model.b1.link_c,
                model.b1.link_F,
                model.b1.link_P1,
                model.b1.link_P3,
                model.b1.link_P2,
                model.b1.link_Pout,
                model.b2.c1,
                model.b1.egb.P2_constraint,
                model.b1.egb.Pout_constraint,
                model.b1.egb.pdrop1,
                model.b1.egb.pdrop3,
            ]
        
        # Now deactivate the grey box and test again with include_greybox = True (should not include grey box constraints)
        model.b1.egb.deactivate()
        aeg = activated_equalities_generator(model, include_greybox=True)
        aeg_list = list(aeg)
        assert len(aeg_list) == 8
        for c in aeg_list:
            assert c in [
                model.b1.link_Pin,
                model.b1.link_c,
                model.b1.link_F,
                model.b1.link_P1,
                model.b1.link_P3,
                model.b1.link_P2,
                model.b1.link_Pout,
                model.b2.c1,
            ]

    @pytest.mark.unit
    def test_activated_equalities_set_w_grey_box(self, model):
        # Test that the activated_equalities_set function correctly identifies the activated equality constraints in the model
        # First, test with include_greybox = False
        aes = activated_equalities_set(model, include_greybox=False)
        assert len(aes) == 8
        for c in aes:
            assert c in [
                model.b1.link_Pin,
                model.b1.link_c,
                model.b1.link_F,
                model.b1.link_P1,
                model.b1.link_P3,
                model.b1.link_P2,
                model.b1.link_Pout,
                model.b2.c1,
            ]
        
        # Next, test with include_greybox = True
        aes = activated_equalities_set(model, include_greybox=True)
        assert len(aes) == 12
        for c in aes:
            assert c in [
                model.b1.link_Pin,
                model.b1.link_c,
                model.b1.link_F,
                model.b1.link_P1,
                model.b1.link_P3,
                model.b1.link_P2,
                model.b1.link_Pout,
                model.b2.c1,
                model.b1.egb.P2_constraint,
                model.b1.egb.Pout_constraint,
                model.b1.egb.pdrop1,
                model.b1.egb.pdrop3,
            ]
        
        # Now deactivate the grey box and test again with include_greybox = True (should not include grey box constraints)
        model.b1.egb.deactivate()
        aes = activated_equalities_set(model, include_greybox=True)
        assert len(aes) == 8
        for c in aes:
            assert c in [
                model.b1.link_Pin,
                model.b1.link_c,
                model.b1.link_F,
                model.b1.link_P1,
                model.b1.link_P3,
                model.b1.link_P2,
                model.b1.link_Pout,
                model.b2.c1,
            ]
    
    @pytest.mark.unit
    def test_number_activated_equalities_w_grey_box(self, model):
        # Test that the number_activated_equalities function correctly counts the number of activated equality constraints in the model
        # First, test with include_greybox = False
        assert number_activated_equalities(model, include_greybox=False) == 8

        # Next, test with include_greybox = True
        assert number_activated_equalities(model, include_greybox=True) == 12

        # Now deactivate the grey box and test again with include_greybox = True (should not include grey box constraints)
        model.b1.egb.deactivate()
        assert number_activated_equalities(model, include_greybox=True) == 8

    @pytest.mark.unit
    def test_number_activated_greybox_equalities_w_grey_box(self, model):
        # Test that the number_activated_greybox_equalities function correctly counts the number of activated equality constraints in the Grey Box blocks in the model
        assert number_activated_greybox_equalities(model) == 4

        # Now deactivate the grey box and test again (should be 0)
        model.b1.egb.deactivate()
        assert number_activated_greybox_equalities(model) == 0

    @pytest.mark.unit
    def test_number_deactivated_equalities_w_grey_box(self, model):
        # Test that the number_deactivated_equalities function correctly counts the number of deactivated equality constraints in the Grey Box blocks in the model
        assert number_deactivated_greybox_equalities(model) == 0

        # Now deactivate the grey box and test again (should be 4)
        model.b1.egb.deactivate()
        assert number_deactivated_greybox_equalities(model) == 4

    @pytest.mark.unit
    def test_deactivated_equalities_generator_w_grey_box(self, model):
        # Test that the deactivated_equalities_generator function correctly identifies the deactivated equality constraints in the Grey Box blocks in the model
        deg = deactivated_equalities_generator(model, include_greybox=True)
        deg_list = list(deg)
        assert len(deg_list) == 0
        
        # Now deactivate the grey box and test again (should include grey box constraints)
        model.b1.egb.deactivate()
        deg = deactivated_equalities_generator(model, include_greybox=True)
        deg_list = list(deg)
        assert len(deg_list) == 4
        for c in deg_list:
            assert c in [
                model.b1.egb.P2_constraint,
                model.b1.egb.Pout_constraint,
                model.b1.egb.pdrop1,
                model.b1.egb.pdrop3,
            ]

    @pytest.mark.unit
    def test_deactivated_equalities_set_w_grey_box(self, model):
        # Test that the deactivated_equalities_set function correctly identifies the deactivated equality constraints in the Grey Box blocks in the model
        decs = deactivated_equalities_set(model, include_greybox=True)
        assert len(decs) == 0
        
        # Now deactivate the grey box and test again (should include grey box constraints)
        model.b1.egb.deactivate()
        decs = deactivated_equalities_set(model, include_greybox=True)
        assert len(decs) == 4
        for c in decs:
            assert c in [
                model.b1.egb.P2_constraint,
                model.b1.egb.Pout_constraint,
                model.b1.egb.pdrop1,
                model.b1.egb.pdrop3,
            ]
    
    @pytest.mark.unit
    def test_number_deactivated_greybox_equalities_w_grey_box(self, model):
        # Test that the number_deactivated_greybox_equalities function correctly counts the number of deactivated equality constraints in the Grey Box blocks in the model
        assert number_deactivated_greybox_equalities(model) == 0

        # Now deactivate the grey box and test again (should be 4)
        model.b1.egb.deactivate()
        assert number_deactivated_greybox_equalities(model) == 4
    
    # Check inequality methods to ensure they work wit ha grey box present
    @pytest.mark.unit
    def test_total_inequalities_generator_w_grey_box(self, model):
        tig = total_inequalities_generator(model)
        tig_list = list(tig)
        assert len(tig_list) == 2
        for c in tig_list:
            assert c in [model.b1.ineq, model.b2.ineq]
    
    @pytest.mark.unit
    def test_total_inequalities_set_w_grey_box(self, model):
        tis = total_inequalities_set(model)
        assert len(tis) == 2
        for c in tis:
            assert c in [model.b1.ineq, model.b2.ineq]
    
    @pytest.mark.unit
    def test_number_total_inequalities_w_grey_box(self, model):
        assert number_total_inequalities(model) == 2
    
    @pytest.mark.unit
    def test_activated_inequalities_generator_w_grey_box(self, model):
        aig = activated_inequalities_generator(model)
        aig_list = list(aig)
        assert len(aig_list) == 2
        for c in aig_list:
            assert c in [model.b1.ineq, model.b2.ineq]
    
    @pytest.mark.unit
    def test_activated_inequalities_set_w_grey_box(self, model):
        aig = activated_inequalities_set(model)
        assert len(aig) == 2
        for c in aig:
            assert c in [model.b1.ineq, model.b2.ineq]
    
    @pytest.mark.unit
    def test_number_activated_inequalities_w_grey_box(self, model):
        assert number_activated_inequalities(model) == 2
    
    @pytest.mark.unit
    def test_deactivated_inequalities_generator_w_grey_box(self, model):
        dig = deactivated_inequalities_generator(model)
        dig_list = list(dig)
        assert len(dig_list) == 0

        # Deactivate one of the inequalities and test again
        model.b1.ineq.deactivate()
        dig = deactivated_inequalities_generator(model)
        dig_list = list(dig)
        assert len(dig_list) == 1
        for c in dig_list:
            assert c in [model.b1.ineq]
    
    @pytest.mark.unit
    def test_deactivated_inequalities_set_w_grey_box(self, model):
        dis = deactivated_inequalities_set(model)
        assert len(dis) == 0

        # Deactivate one of the inequalities and test again
        model.b1.ineq.deactivate()
        dis = deactivated_inequalities_set(model)
        assert len(dis) == 1
        for c in dis:
            assert c in [model.b1.ineq]
    
    @pytest.mark.unit
    def test_number_deactivated_inequalities_w_grey_box(self, model):
        assert number_deactivated_inequalities(model) == 0

        # Deactivate one of the inequalities and test again
        model.b1.ineq.deactivate()
        assert number_deactivated_inequalities(model) == 1


class TestVariableStatisticsGreyBox:
    @pytest.mark.unit
    def test_variables_generator_w_grey_box(self, model):
        # Start with include_greybox = False to confirm that we are not including grey box variables
        vg = variables_generator(model, include_greybox=False)
        vg_list = list(vg)
        assert len(vg_list) == 8
        for v in vg_list:
            assert v.name in [
                "b1.Pin",
                "b1.c",
                "b1.F",
                "b1.P1",
                "b1.P3",
                "b1.P2",
                "b1.Pout",
                "b2.v1",
            ]
        
        # Now test with include_greybox = True to confirm that we are including grey box variables
        vg = variables_generator(model, include_greybox=True)
        vg_list = list(vg)
        assert len(vg_list) == 15
        for v in vg_list:
            print(v.name)
            assert v.name in [
                "b1.Pin",
                "b1.c",
                "b1.F",
                "b1.P1",
                "b1.P3",
                "b1.P2",
                "b1.Pout",
                "b2.v1",
                "b1.egb.inputs[Pin]",
                "b1.egb.inputs[c]",
                "b1.egb.inputs[F]",
                "b1.egb.inputs[P1]",
                "b1.egb.outputs[P2]",
                "b1.egb.inputs[P3]",
                "b1.egb.outputs[Pout]",
            ]
    
    @pytest.mark.unit
    def test_variables_set_w_grey_box(self, model):
        # Start with include_greybox = False to confirm that we are not including grey box variables
        vs = variables_set(model, include_greybox=False)
        assert len(vs) == 8
        for v in vs:
            assert v.name in [
                "b1.Pin",
                "b1.c",
                "b1.F",
                "b1.P1",
                "b1.P3",
                "b1.P2",
                "b1.Pout",
                "b2.v1",
            ]
        
        # Now test with include_greybox = True to confirm that we are including grey box variables
        vs = variables_set(model, include_greybox=True)
        assert len(vs) == 15
        for v in vs:
            assert v.name in [
                "b1.Pin",
                "b1.c",
                "b1.F",
                "b1.P1",
                "b1.P3",
                "b1.P2",
                "b1.Pout",
                "b2.v1",
                "b1.egb.inputs[Pin]",
                "b1.egb.inputs[c]",
                "b1.egb.inputs[F]",
                "b1.egb.inputs[P1]",
                "b1.egb.outputs[P2]",
                "b1.egb.inputs[P3]",
                "b1.egb.outputs[Pout]",
            ]
    
    @pytest.mark.unit
    def test_number_variables_w_grey_box(self, model):
        # Start with include_greybox = False to confirm that we are not including grey box variables
        assert number_variables(model, include_greybox=False) == 8

        # Now test with include_greybox = True to confirm that we are including grey box variables
        assert number_variables(model, include_greybox=True) == 15
    
    @pytest.mark.unit
    def test_fixed_variables_generator(self, model):
        # Test that the fixed_variables_generator function correctly identifies the fixed variables in the model
        fvg = fixed_variables_generator(model)
        fvg_list = list(fvg)
        assert len(fvg_list) == 0

        # Fix some of the variables and test again
        model.b1.Pin.fix(1)
        model.b1.egb.inputs["Pin"].fix(1)
        fvg = fixed_variables_generator(model)
        fvg_list = list(fvg)
        assert len(fvg_list) == 2
        for v in fvg_list:
            assert v.name in ["b1.Pin", "b1.egb.inputs[Pin]"]
        
        # Test again with include_greybox = False to confirm that we are not including grey box variables
        fvg = fixed_variables_generator(model, include_greybox=False)
        fvg_list = list(fvg)
        assert len(fvg_list) == 1
        for v in fvg_list:
            assert v.name in ["b1.Pin"]
    
    @pytest.mark.unit
    def test_fixed_variables_set(self, model):
        # Test that the fixed_variables_set function correctly identifies the fixed variables in the model
        fvs = fixed_variables_set(model)
        assert len(fvs) == 0

        # Fix some of the variables and test again
        model.b1.Pin.fix(1)
        model.b1.egb.inputs["Pin"].fix(1)
        fvs = fixed_variables_set(model)
        assert len(fvs) == 2
        for v in fvs:
            assert v.name in ["b1.Pin", "b1.egb.inputs[Pin]"]
        
        # Test again with include_greybox = False to confirm that we are not including grey box variables
        fvs = fixed_variables_set(model, include_greybox=False)
        assert len(fvs) == 1
        for v in fvs:
            assert v.name in ["b1.Pin"]
    
    @pytest.mark.unit
    def test_number_fixed_variables(self, model):
        # Test that the number_fixed_variables function correctly counts the number of fixed variables in the model
        assert number_fixed_variables(model) == 0

        # Fix some of the variables and test again
        model.b1.Pin.fix(1)
        model.b1.egb.inputs["Pin"].fix(1)
        assert number_fixed_variables(model) == 2

        # Test again with include_greybox = False to confirm that we are not including grey box variables
        assert number_fixed_variables(model, include_greybox=False) == 1
    
    @pytest.mark.unit
    def test_unfixed_variables_generator(self, model):
        # Test that the unfixed_variables_generator function correctly identifies the unfixed variables in the model
        uv = unfixed_variables_generator(model)
        uv_list = list(uv)
        assert len(uv_list) == 15
        for v in uv_list:
            assert v.name in [
                "b1.Pin",
                "b1.c",
                "b1.F",
                "b1.P1",
                "b1.P3",
                "b1.P2",
                "b1.Pout",
                "b2.v1",
                "b1.egb.inputs[Pin]",
                "b1.egb.inputs[c]",
                "b1.egb.inputs[F]",
                "b1.egb.inputs[P1]",
                "b1.egb.outputs[P2]",
                "b1.egb.inputs[P3]",
                "b1.egb.outputs[Pout]",
            ]
        
        # Now fix some of the variables and test again
        model.b1.Pin.fix(1)
        model.b1.egb.inputs["Pin"].fix(1)
        uv = unfixed_variables_generator(model)
        uv_list = list(uv)
        assert len(uv_list) == 13
        for v in uv_list:
            assert v.name in [
                "b1.c",
                "b1.F",
                "b1.P1",
                "b1.P3",
                "b1.P2",
                "b1.Pout",
                "b2.v1",
                "b1.egb.inputs[c]",
                "b1.egb.inputs[F]",
                "b1.egb.inputs[P1]",
                "b1.egb.outputs[P2]",
                "b1.egb.inputs[P3]",
                "b1.egb.outputs[Pout]",
            ]
        
        # Test again with include_greybox = False to confirm that we are not including grey box variables
        uv = unfixed_variables_generator(model, include_greybox=False)
        uv_list = list(uv)
        assert len(uv_list) == 7
        for v in uv_list:
            assert v.name in [
                "b1.c",
                "b1.F",
                "b1.P1",
                "b1.P3",
                "b1.P2",
                "b1.Pout",
                "b2.v1",
            ]
    
    @pytest.mark.unit
    def test_unfixed_variables_set(self, model):
        # Test that the unfixed_variables_set function correctly identifies the unfixed variables in the model
        uvs = unfixed_variables_set(model)
        assert len(uvs) == 15
        for v in uvs:
            assert v.name in [
                "b1.Pin",
                "b1.c",
                "b1.F",
                "b1.P1",
                "b1.P3",
                "b1.P2",
                "b1.Pout",
                "b2.v1",
                "b1.egb.inputs[Pin]",
                "b1.egb.inputs[c]",
                "b1.egb.inputs[F]",
                "b1.egb.inputs[P1]",
                "b1.egb.outputs[P2]",
                "b1.egb.inputs[P3]",
                "b1.egb.outputs[Pout]",
            ]
        
        # Now fix some of the variables and test again
        model.b1.Pin.fix(1)
        model.b1.egb.inputs["Pin"].fix(1)
        uvs = unfixed_variables_set(model)
        assert len(uvs) == 13
        for v in uvs:
            assert v.name in [
                "b1.c",
                "b1.F",
                "b1.P1",
                "b1.P3",
                "b1.P2",
                "b1.Pout",
                "b2.v1",
                "b1.egb.inputs[c]",
                "b1.egb.inputs[F]",
                "b1.egb.inputs[P1]",
                "b1.egb.outputs[P2]",
                "b1.egb.inputs[P3]",
                "b1.egb.outputs[Pout]",
            ]
        
        # Test again with include_greybox = False to confirm that we are not including grey box variables
        uvs = unfixed_variables_set(model, include_greybox=False)
        assert len(uvs) == 7
        for v in uvs:
            assert v.name in [
                "b1.c",
                "b1.F",
                "b1.P1",
                "b1.P3",
                "b1.P2",
                "b1.Pout",
                "b2.v1",
            ]
    
    @pytest.mark.unit
    def test_number_unfixed_variables(self, model):
        # Test that the number_unfixed_variables function correctly counts the number of unfixed variables in the model
        assert number_unfixed_variables(model) == 15

        # Now fix some of the variables and test again
        model.b1.Pin.fix(1)
        model.b1.egb.inputs["Pin"].fix(1)
        assert number_unfixed_variables(model) == 13

        # Test again with include_greybox = False to confirm that we are not including grey box variables
        assert number_unfixed_variables(model, include_greybox=False) == 7
    
    @pytest.mark.unit
    def test_variables_near_bounds_generator(self, model):
        # Test that the variables_near_bounds_generator function correctly identifies the variables that are near their bounds in the model
        vnbg = variables_near_bounds_generator(model, tol=1e-6)
        vnbg_list = list(vnbg)
        assert len(vnbg_list) == 3
        for v in vnbg_list:
            assert v.name in ["b1.c", "b1.egb.inputs[c]", "b1.egb.outputs[P2]"]
        
        # Now test with include_greybox = False to confirm that we are not including grey box variables
        vnbg = variables_near_bounds_generator(model, tol=1e-6, include_greybox=False)
        vnbg_list = list(vnbg)
        assert len(vnbg_list) == 1
        for v in vnbg_list:
            assert v.name in ["b1.c"]

    @pytest.mark.unit
    def test_variables_near_bounds_set(self, model):
        # Test that the variables_near_bounds_set function correctly identifies the variables that are near their bounds in the model
        vnbs = variables_near_bounds_set(model, tol=1e-6)
        assert len(vnbs) == 3
        for v in vnbs:
            assert v.name in ["b1.c", "b1.egb.inputs[c]", "b1.egb.outputs[P2]"]
        
        # Now test with include_greybox = False to confirm that we are not including grey box variables
        vnbs = variables_near_bounds_set(model, tol=1e-6, include_greybox=False)
        assert len(vnbs) == 1
        for v in vnbs:
            assert v.name in ["b1.c"]
    
    @pytest.mark.unit
    def test_number_variables_near_bounds(self, model):
        # Test that the number_variables_near_bounds function correctly counts the number of variables that are near their bounds in the model
        assert number_variables_near_bounds(model, tol=1e-6) == 3

        # Now test with include_greybox = False to confirm that we are not including grey box variables
        assert number_variables_near_bounds(model, tol=1e-6, include_greybox=False) == 1
    
    @pytest.mark.unit
    def test_variables_in_activated_constraints_set_w_grey_box(self, model):
        # Test that the variables_in_activated_constraints_set function correctly identifies the variables that are in the activated constraints in the model
        # First, test with include_greybox = False
        # We should see all the variables, including those in the grey box, as they appear
        # in the linking constraints.
        vics = variables_in_activated_constraints_set(model, include_greybox=False)
        assert len(vics) == 15
        for v in vics:
            assert v.name in [
                "b1.Pin",
                "b1.c",
                "b1.F",
                "b1.P1",
                "b1.P3",
                "b1.P2",
                "b1.Pout",
                "b2.v1",
                "b1.egb.inputs[Pin]",
                "b1.egb.inputs[c]",
                "b1.egb.inputs[F]",
                "b1.egb.inputs[P1]",
                "b1.egb.outputs[P2]",
                "b1.egb.inputs[P3]",
                "b1.egb.outputs[Pout]",
            ]
        
        # Next, deactivate the grey box - result should still be the same
        model.b1.egb.deactivate()
        vics = variables_in_activated_constraints_set(model, include_greybox=True)
        assert len(vics) == 15
        for v in vics:
            assert v.name in [
                "b1.Pin",
                "b1.c",
                "b1.F",
                "b1.P1",
                "b1.P3",
                "b1.P2",
                "b1.Pout",
                "b2.v1",
                "b1.egb.inputs[Pin]",
                "b1.egb.inputs[c]",
                "b1.egb.inputs[F]",
                "b1.egb.inputs[P1]",
                "b1.egb.outputs[P2]",
                "b1.egb.inputs[P3]",
                "b1.egb.outputs[Pout]",
            ]
        
        # Now, turn off a linking constraint
        model.b1.link_Pin.deactivate()
        vics = variables_in_activated_constraints_set(model, include_greybox=True)
        assert len(vics) == 14
        for v in vics:
            assert v.name in [
                "b1.Pin",  # This is in the inequality constraint
                "b1.c",
                "b1.F",
                "b1.P1",
                "b1.P3",
                "b1.P2",
                "b1.Pout",
                "b2.v1",
                # "b1.egb.inputs[Pin]",
                "b1.egb.inputs[c]",
                "b1.egb.inputs[F]",
                "b1.egb.inputs[P1]",
                "b1.egb.outputs[P2]",
                "b1.egb.inputs[P3]",
                "b1.egb.outputs[Pout]",
            ]
        
        # Finally, turn the grey box back on
        # All variables should be back, as the grey box variables are all considered
        # to be in active constraints if the grey box is active
        model.b1.egb.activate()
        vics = variables_in_activated_constraints_set(model, include_greybox=True)
        assert len(vics) == 15
        for v in vics:
            assert v.name in [
                "b1.Pin",
                "b1.c",
                "b1.F",
                "b1.P1",
                "b1.P3",
                "b1.P2",
                "b1.Pout",
                "b2.v1",
                "b1.egb.inputs[Pin]",
                "b1.egb.inputs[c]",
                "b1.egb.inputs[F]",
                "b1.egb.inputs[P1]",
                "b1.egb.outputs[P2]",
                "b1.egb.inputs[P3]",
                "b1.egb.outputs[Pout]",
            ]
    
    @pytest.mark.unit
    def test_number_variables_in_activated_constraints_w_grey_box(self, model):
        # Test that the number_variables_in_activated_constraints function correctly counts the number of variables that are in the activated constraints in the model
        # First, test with include_greybox = False
        assert number_variables_in_activated_constraints(model, include_greybox=False) == 15

        # Next, deactivate the grey box - result should still be the same
        model.b1.egb.deactivate()
        assert number_variables_in_activated_constraints(model, include_greybox=True) == 15

        # Now, turn off a linking constraint
        model.b1.link_Pin.deactivate()
        assert number_variables_in_activated_constraints(model, include_greybox=True) == 14

        # Finally, turn the grey box back on - should go back to 15
        model.b1.egb.activate()
        assert number_variables_in_activated_constraints(model, include_greybox=True) == 15
