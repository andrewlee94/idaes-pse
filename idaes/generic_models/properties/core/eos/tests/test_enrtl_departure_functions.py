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
from copy import deepcopy
from math import log

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
    LinearAlpha, InverseTau
from idaes.generic_models.properties.core.eos.enrtl_reference_states import \
    Symmetric

from idaes.generic_models.properties.core.eos.enrtl_departure_functions import *


def between(y, x1, x2):
    return 0 > (y-x1)*(y-x2)


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
    "temperature_ref": 298.15,
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

    @pytest.mark.unit
    def test_dlngamma_dT(self, model):
        delT = 1e-8
        for i in model.state[1].Liq_log_gamma:
            model.state[1].temperature.set_value(298.15)
            lng_1 = value(model.state[1].Liq_log_gamma[i])

            model.state[1].temperature.set_value(298.15+delT)
            lng_2 = value(model.state[1].Liq_log_gamma[i])

            dlng_dT = (lng_2 - lng_1)/delT
            assert pytest.approx(dlng_dT, rel=2e-3) == value(
                dlngamma_dT(model.state[1], "Liq", i, Symmetric))

            # Repeat for a higher T
            model.state[1].temperature.set_value(500)
            lng_1 = value(model.state[1].Liq_log_gamma[i])

            model.state[1].temperature.set_value(500+delT)
            lng_2 = value(model.state[1].Liq_log_gamma[i])

            dlng_dT = (lng_2 - lng_1)/delT
            assert pytest.approx(dlng_dT, rel=4e-3) == value(
                dlngamma_dT(model.state[1], "Liq", i, Symmetric))


class TestLCTermVariable():
    cls_config = deepcopy(configuration)
    cls_config["phases"]["Liq"]["equation_of_state_options"] = {
        "alpha_rule": LinearAlpha,
        "tau_rule": InverseTau}
    cls_config["parameter_data"] = {
        "Liq_alpha_2": {
            ("Na+, Cl-", "H2O"): 0.1,
            ("H2O", "H2O"): 0.2},
        "Liq_tau_2": {
            ("Na+, Cl-", "H2O"): 10,
            ("H2O", "Na+, Cl-"): 20,
            ("H2O", "H2O"): 30}}

    @pytest.fixture(scope="class")
    def model(self):
        m = ConcreteModel()
        m.params = GenericParameterBlock(default=TestLCTermVariable.cls_config)

        m.state = m.params.build_state_block([1])

        # Need to set a value of T for checking expressions later
        m.state[1].temperature.set_value(298.15)

        return m

    @pytest.mark.unit
    def test_dalpha_dT(self, model):
        delT = 1e-8
        for i, j in model.state[1].Liq_alpha:
            model.state[1].temperature.set_value(298.15)
            a1 = value(model.state[1].Liq_alpha[i, j])
            v = value(dalpha_dT(model.state[1], "Liq", i, j))

            model.state[1].temperature.set_value(298.15+delT)
            a2 = value(model.state[1].Liq_alpha[i, j])
            num_a_dT = (a2-a1)/delT

            if i == "H2O" and j == "H2O":
                assert v == 0.2
            elif i == "Na+" and j == "Cl-":
                assert v == 0
            elif i == "Cl-" and j == "Na+":
                assert v == 0
            else:
                assert v == 0.1

            assert v == pytest.approx(num_a_dT, rel=1e-3)

            # Repeat for higher T
            model.state[1].temperature.set_value(500)
            a1 = value(model.state[1].Liq_alpha[i, j])
            v = value(dalpha_dT(model.state[1], "Liq", i, j))

            model.state[1].temperature.set_value(500+delT)
            a2 = value(model.state[1].Liq_alpha[i, j])
            num_a_dT = (a2-a1)/delT

            if i == "H2O" and j == "H2O":
                assert v == 0.2
            elif i == "Na+" and j == "Cl-":
                assert v == 0
            elif i == "Cl-" and j == "Na+":
                assert v == 0
            else:
                assert v == 0.1

            assert v == pytest.approx(num_a_dT, rel=1e-3)

    @pytest.mark.unit
    def test_dG_dT(self, model):
        delT = 1e-8
        for i, j in model.state[1].Liq_G:
            model.state[1].temperature.set_value(298.15)
            G1 = value(model.state[1].Liq_G[i, j])
            a = value(model.state[1].Liq_alpha[i, j])
            t = value(model.state[1].Liq_tau[i, j])
            v = value(dG_dT(model.state[1], "Liq", i, j))

            model.state[1].temperature.set_value(298.15+delT)
            G2 = value(model.state[1].Liq_G[i, j])
            num_G_dT = (G2-G1)/delT

            if i == j:
                assert v == 0
            elif i == "H2O" and j in ["Na+", "Cl-"]:
                assert v == pytest.approx(
                    -G1*(a*(-20/298.15**2) + t*0.1), rel=1e-6)
            elif i in ["Na+", "Cl-"] and j == "H2O":
                assert v == pytest.approx(
                    -G1*(a*(-10/298.15**2) + t*0.1), rel=1e-6)
            else:
                assert v == 0

            assert v == pytest.approx(num_G_dT, rel=1e-3)

            # Repeat for higher T
            model.state[1].temperature.set_value(500)
            G1 = value(model.state[1].Liq_G[i, j])
            a = value(model.state[1].Liq_alpha[i, j])
            t = value(model.state[1].Liq_tau[i, j])
            v = value(dG_dT(model.state[1], "Liq", i, j))

            model.state[1].temperature.set_value(500+delT)
            G2 = value(model.state[1].Liq_G[i, j])
            num_G_dT = (G2-G1)/delT

            if i == j:
                assert v == 0
            elif i == "H2O" and j in ["Na+", "Cl-"]:
                assert v == pytest.approx(
                    -G1*(a*(-20/500**2) + t*0.1), rel=1e-6)
            elif i in ["Na+", "Cl-"] and j == "H2O":
                assert v == pytest.approx(
                    -G1*(a*(-10/500**2) + t*0.1), rel=1e-6)
            else:
                assert v == 0

            assert v == pytest.approx(num_G_dT, rel=1e-3)

    @pytest.mark.unit
    def test_dtau_dT(self, model):
        delT = 1e-8
        for i, j in model.state[1].Liq_tau:
            model.state[1].temperature.set_value(298.15)
            t1 = value(model.state[1].Liq_tau[i, j])
            G = value(model.state[1].Liq_G[i, j])
            a = value(model.state[1].Liq_alpha[i, j])
            v = value(dtau_dT(model.state[1], "Liq", i, j))

            model.state[1].temperature.set_value(298.15+delT)
            t2 = value(model.state[1].Liq_tau[i, j])
            num_tau_dT = (t2-t1)/delT

            if i == "H2O" and j == "H2O":
                assert v == -30/298.15**2
            elif i == "H2O" and j in ["Na+", "Cl-"]:
                dG_dT = -G*(a*(-20/298.15**2) + t1*1)
                assert v == pytest.approx(
                    ((1*log(G) - a*dG_dT/G)/a**2), rel=1e-6)
            elif i in ["Na+", "Cl-"] and j == "H2O":
                dG_dT = -G*(a*(-10/298.15**2) + t1*1)
                assert v == pytest.approx(
                    ((1*log(G) - a*dG_dT/G)/a**2), rel=1e-6)
            else:
                assert v == 0

            assert v == pytest.approx(num_tau_dT, rel=1e-3)

            # Repeat for higher T
            model.state[1].temperature.set_value(500)

            t1 = value(model.state[1].Liq_tau[i, j])
            G = value(model.state[1].Liq_G[i, j])
            a = value(model.state[1].Liq_alpha[i, j])
            v = value(dtau_dT(model.state[1], "Liq", i, j))

            model.state[1].temperature.set_value(500+delT)
            t2 = value(model.state[1].Liq_tau[i, j])
            num_tau_dT = (t2-t1)/delT

            if i == "H2O" and j == "H2O":
                assert v == -30/500**2
            elif i == "H2O" and j in ["Na+", "Cl-"]:
                dG_dT = -G*(a*(-20/500**2) + t1*1)
                assert v == pytest.approx(
                    ((1*log(G) - a*dG_dT/G)/a**2), rel=1e-6)
            elif i in ["Na+", "Cl-"] and j == "H2O":
                dG_dT = -G*(a*(-10/500**2) + t1*1)
                assert v == pytest.approx(
                    ((1*log(G) - a*dG_dT/G)/a**2), rel=1e-6)
            else:
                assert v == 0

            assert v == pytest.approx(num_tau_dT, rel=1e-3)

    @pytest.mark.unit
    def test_dlngamma_dT(self, model):
        delT = 1e-8
        dtol = 1e-2

        for i in model.state[1].Liq_log_gamma:
            if i != "H2O":
                continue
            T = 298.15

            model.state[1].temperature.set_value(T-delT)
            lng_1 = value(model.state[1].Liq_log_gamma[i])
            model.state[1].temperature.set_value(T)
            lng_2 = value(model.state[1].Liq_log_gamma[i])
            model.state[1].temperature.set_value(T+delT)
            lng_3 = value(model.state[1].Liq_log_gamma[i])

            dlng_dT_m = (lng_2 - lng_1)/delT
            dlng_dT_p = (lng_3 - lng_2)/delT

            model.state[1].temperature.set_value(T)
            dlng = value(dlngamma_dT(model.state[1], "Liq", i, Symmetric))
            print(dlng_dT_m, dlng, dlng_dT_p)
            assert (pytest.approx(dlng_dT_m, rel=dtol) == dlng or
                    pytest.approx(dlng_dT_p, rel=dtol) == dlng or
                    (between(dlng, dlng_dT_m, dlng_dT_p) and
                     (pytest.approx(dlng_dT_m, rel=10*dtol) == dlng or
                      pytest.approx(dlng_dT_p, rel=10*dtol) == dlng)))

            # Repeat for a higher T
            T = 500

            model.state[1].temperature.set_value(T-delT)
            lng_1 = value(model.state[1].Liq_log_gamma[i])
            model.state[1].temperature.set_value(T)
            lng_2 = value(model.state[1].Liq_log_gamma[i])
            model.state[1].temperature.set_value(T+delT)
            lng_3 = value(model.state[1].Liq_log_gamma[i])

            dlng_dT_m = (lng_2 - lng_1)/delT
            dlng_dT_p = (lng_3 - lng_2)/delT

            model.state[1].temperature.set_value(T)
            dlng = value(dlngamma_dT(model.state[1], "Liq", i, Symmetric))
            print(dlng_dT_m, dlng, dlng_dT_p)
            assert (pytest.approx(dlng_dT_m, rel=dtol) == dlng or
                    pytest.approx(dlng_dT_p, rel=dtol) == dlng or
                    (between(dlng, dlng_dT_m, dlng_dT_p) and
                     (pytest.approx(dlng_dT_m, rel=10*dtol) == dlng or
                      pytest.approx(dlng_dT_p, rel=10*dtol) == dlng)))
        assert False