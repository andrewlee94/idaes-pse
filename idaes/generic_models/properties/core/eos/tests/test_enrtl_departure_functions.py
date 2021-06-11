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
Tests for eNRTL temperature derivative methods

Author: Andrew Lee
"""
import pytest

from pyomo.environ import (ConcreteModel,
                           units as pyunits,
                           value)
from pyomo.util.check_units import assert_units_equivalent

from idaes.core import (AqueousPhase,
                        Solvent,
                        Apparent,
                        Anion,
                        Cation)
from idaes.generic_models.properties.core.eos.enrtl import ENRTL
from idaes.generic_models.properties.core.generic.generic_property import (
        GenericParameterBlock, StateIndex)
from idaes.generic_models.properties.core.state_definitions import FTPx
from idaes.generic_models.properties.core.pure.electrolyte import \
    relative_permittivity_constant
from idaes.generic_models.properties.core.eos.enrtl_parameters import \
    LinearAlpha

from idaes.generic_models.properties.core.eos.enrtl_departure_functions import *


class density(object):
    def return_expression(*ags, **kwargs):
        return 1000/18e-3*pyunits.mol/pyunits.m**3

    def dT_expression(*args, **kwargs):
        return 1*pyunits.mol/pyunits.m**3/pyunits.K


configuration = {
    "components": {
        "H2O": {"type": Solvent,
                "dens_mol_liq_comp": density,
                "relative_permittivity_liq_comp":
                    relative_permittivity_constant,
                "parameter_data": {
                    "mw": (18E-3, pyunits.kg/pyunits.mol),
                    "relative_permittivity_liq_comp": 78.54}},
        "NaCl": {"type": Apparent,
                 "dissociation_species": {"Na+": 1, "Cl-": 1}},
        "Na+": {"type": Cation,
                "charge": +1},
        "Cl-": {"type": Anion,
                "charge": -1}},
    "phases": {
        "Liq": {"type": AqueousPhase,
                "equation_of_state": ENRTL}},
    "base_units": {"time": pyunits.s,
                   "length": pyunits.m,
                   "mass": pyunits.kg,
                   "amount": pyunits.mol,
                   "temperature": pyunits.K},
    "state_definition": FTPx,
    "state_components": StateIndex.true,
    "pressure_ref": 1e5,
    "temperature_ref": 300,
    "parameter_data": {
        "Liq_tau": {
            ("H2O", "Na+, Cl-"): 8.885,  # Table 1, [1]
            ("Na+, Cl-", "H2O"): -4.549}}}  # Table 1, [1]


class TestPDHTerm():
    @pytest.fixture(scope="class")
    def model(self):
        m = ConcreteModel()
        m.params = GenericParameterBlock(default=configuration)

        m.state = m.params.build_state_block([1])

        # Need to set a value of T for checking expressions later
        m.state[1].temperature.set_value(298.15)

        return m

    @pytest.mark.unit
    def test_dv_dT(self, model):
        e = dv_dT(model.state[1], "Liq")
        assert value(e) == pytest.approx(3.23e-10, rel=1e-5)
        assert_units_equivalent(e, pyunits.m**3/pyunits.mol/pyunits.K)

    @pytest.mark.unit
    def test_deps_dT(self, model):
        e = deps_dT(model.state[1], "Liq")
        assert value(e) == 0
        assert_units_equivalent(e, pyunits.dimensionless)

    @pytest.mark.unit
    def test_dA_dT(self, model):
        A1 = value(model.state[1].Liq_A_DH)
        dA1 = dA_dt(model.state[1], "Liq")

        # Add a minor disturbance to T
        delT = 1e-8
        model.state[1].temperature.set_value(298.15+delT)
        A2 = value(model.state[1].Liq_A_DH)

        assert value(dA1) == pytest.approx((A2-A1)/delT, rel=2e-3)

        # Repeat for a higher T
        model.state[1].temperature.set_value(500)

        A1 = value(model.state[1].Liq_A_DH)
        dA1 = dA_dt(model.state[1], "Liq")

        # Add a minor disturbance to T
        delT = 1e-8
        model.state[1].temperature.set_value(500+delT)
        A2 = value(model.state[1].Liq_A_DH)

        assert value(dA1) == pytest.approx((A2-A1)/delT, rel=3e-3)


class TestLCTermConstant():
    @pytest.fixture(scope="class")
    def model(self):
        m = ConcreteModel()
        m.params = GenericParameterBlock(default=configuration)

        m.state = m.params.build_state_block([1])

        # Need to set a value of T for checking expressions later
        m.state[1].temperature.set_value(298.15)

        return m

    @pytest.mark.unit
    def test_dalpha_dT(self, model):
        for i, j in model.state[1].Liq_alpha:
            assert value(dalpha_dT(model.state[1], "Liq", i, j)) == 0

    @pytest.mark.unit
    def test_dG_dT(self, model):
        for i, j in model.state[1].Liq_G:
            assert value(dG_dT(model.state[1], "Liq", i, j)) == 0

    @pytest.mark.unit
    def test_dtau_dT(self, model):
        for i, j in model.state[1].Liq_tau:
            assert value(dtau_dT(model.state[1], "Liq", i, j)) == 0


class TestLCTermVariable():
    cls_config = dict(configuration)
    cls_config["phase"]["Liq"]["equation_of_state_options"] = {
        "alpha_rule": LinearAlpha}

    @pytest.fixture(scope="class")
    def model(self):
        m = ConcreteModel()
        m.params = GenericParameterBlock(default=configuration)

        m.state = m.params.build_state_block([1])

        # Need to set a value of T for checking expressions later
        m.state[1].temperature.set_value(298.15)

        return m

    @pytest.mark.unit
    def test_dalpha_dT(self, model):
        for i, j in model.state[1].Liq_alpha:
            assert value(dalpha_dT(model.state[1], "Liq", i, j)) == 0

    @pytest.mark.unit
    def test_dG_dT(self, model):
        for i, j in model.state[1].Liq_G:
            assert value(dG_dT(model.state[1], "Liq", i, j)) == 0

    @pytest.mark.unit
    def test_dtau_dT(self, model):
        for i, j in model.state[1].Liq_tau:
            assert value(dtau_dT(model.state[1], "Liq", i, j)) == 0
