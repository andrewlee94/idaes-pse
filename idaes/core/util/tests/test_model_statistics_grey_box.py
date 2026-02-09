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
    m.b1.Pin = pyo.Var()
    m.b1.c = pyo.Var()
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

