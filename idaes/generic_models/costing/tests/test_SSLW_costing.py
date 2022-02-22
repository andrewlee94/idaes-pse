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
Tests for costing package based on methods from:

    Process and Product Design Principles: Synthesis, Analysis, and
    Evaluation
    Seider, Seader, Lewin, Windagdo, 3rd Ed. John Wiley and Sons
    Chapter 22. Cost Accounting and Capital Cost Estimation
    22.2 Cost Indexes and Capital Investment
"""
import pytest

from pyomo.environ import (Block,
                           check_optimal_termination,
                           ConcreteModel,
                           Constraint,
                           Param,
                           units as pyunits,
                           value,
                           Var)
from pyomo.util.check_units import assert_units_consistent

from idaes.core import FlowsheetBlock
from idaes.generic_models.costing import FlowsheetCostingBlock
from idaes.core.util import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom

from idaes.generic_models.costing.SSLW import (SSLWCosting,
                                               HXMaterial,
                                               HXTubeLength,
                                               HXType,
                                               VesselMaterial,
                                               TrayType,
                                               TrayMaterial)


# Some more information about this module
__author__ = "Andrew Lee"


solver = get_solver()


@pytest.fixture
def model():
    m = ConcreteModel()

    m.fs = FlowsheetBlock()

    m.fs.costing = FlowsheetCostingBlock(
        default={"costing_package": SSLWCosting})

    # Add a placeholder to represent a unit model
    m.fs.unit = Block()

    return m


@pytest.mark.unit
def test_global_definitions(model):
    assert "USD_500" in pyunits.pint_registry

    CEI = {"USD2010": 550.8,
           "USD2011": 585.7,
           "USD2012": 584.6,
           "USD2013": 567.3,
           "USD2014": 576.1,
           "USD2015": 556.8,
           "USD2016": 541.7,
           "USD2017": 567.5,
           "USD2018": 671.1,
           "USD2019": 680.0}

    for c, conv in CEI.items():
        assert c in pyunits.pint_registry

        assert pytest.approx(conv/500, rel=1e-10) == pyunits.convert_value(
            1, pyunits.USD_500, getattr(pyunits, c))


@pytest.mark.component
@pytest.mark.parametrize("material", HXMaterial)
@pytest.mark.parametrize("hxtype", HXType)
@pytest.mark.parametrize("tube_length", HXTubeLength)
def test_cost_heat_exchanger(model, material, hxtype, tube_length):
    model.fs.unit.area = Param(initialize=1000,
                               units=pyunits.m**2)
    model.fs.unit.tube = Block()
    model.fs.unit.tube.properties_in = Block(model.fs.time)
    model.fs.unit.tube.properties_in[0].pressure = Param(
        initialize=2, units=pyunits.atm)

    model.fs.costing.cost_unit(model.fs.unit,
                               SSLWCosting.cost_heat_exchanger,
                               hx_type=hxtype,
                               material_type=material,
                               tube_length=tube_length)

    assert isinstance(model.fs.unit_costing["unit"].base_cost_per_unit, Var)
    assert isinstance(model.fs.unit_costing["unit"].capital_cost, Var)
    assert isinstance(model.fs.unit_costing["unit"].number_of_units, Var)
    assert isinstance(model.fs.unit_costing["unit"].pressure_factor, Var)
    assert isinstance(model.fs.unit_costing["unit"].material_factor, Var)

    assert isinstance(model.fs.unit_costing["unit"].capital_cost_constraint,
                      Constraint)
    assert isinstance(model.fs.unit_costing["unit"].hx_material_eqn,
                      Constraint)
    assert isinstance(model.fs.unit_costing["unit"].p_factor_eq,
                      Constraint)

    assert degrees_of_freedom(model) == 0
    assert_units_consistent(model.fs.unit_costing)

    res = solver.solve(model)

    assert check_optimal_termination(res)

    # Test solution for one known case
    if (material == HXMaterial.SS_SS and
            hxtype == HXType.Utube and
            tube_length == HXTubeLength.TwelveFoot):
        assert pytest.approx(87704.6, 1e-5) == value(
            model.fs.unit_costing["unit"].base_cost_per_unit)
        assert pytest.approx(0.982982, 1e-5) == value(
            model.fs.unit_costing["unit"].pressure_factor)
        assert pytest.approx(4.08752, 1e-5) == value(
            model.fs.unit_costing["unit"].material_factor)

        assert pytest.approx(529737, 1e-5) == value(pyunits.convert(
            model.fs.unit_costing["unit"].capital_cost,
            to_units=pyunits.USD2018))


# number_of_trays=None,
# tray_material=TrayMaterial.CS,
# tray_type=TrayType.Sieve,

# TODO: Some arguments only apply to vertical vessels
@pytest.mark.component
@pytest.mark.parametrize("material_type", VesselMaterial)
@pytest.mark.parametrize("weight_limit", [1, 2])
@pytest.mark.parametrize("aspect_ratio_range", [1, 2])
@pytest.mark.parametrize("include_pl", [True, False])
def test_cost_vessel(model,
                     material_type,
                     weight_limit,
                     aspect_ratio_range,
                     include_pl):
    model.fs.unit.length = Param(initialize=0.00075,
                                 units=pyunits.m)
    model.fs.unit.diameter = Param(initialize=2,
                                   units=pyunits.m)

    model.fs.costing.cost_unit(model.fs.unit,
                               SSLWCosting.cost_vessel,
                               vertical=True,
                               material_type=material_type,
                               weight_limit=weight_limit,
                               aspect_ratio_range=aspect_ratio_range,
                               include_platforms_ladders=include_pl)

    assert isinstance(model.fs.unit_costing["unit"].shell_thickness, Param)
    assert isinstance(model.fs.unit_costing["unit"].material_factor, Param)
    assert isinstance(model.fs.unit_costing["unit"].material_density, Param)

    assert isinstance(model.fs.unit_costing["unit"].base_cost_per_unit, Var)
    assert isinstance(model.fs.unit_costing["unit"].capital_cost, Var)
    assert isinstance(model.fs.unit_costing["unit"].weight, Var)

    assert isinstance(model.fs.unit_costing["unit"].capital_cost_constraint,
                      Constraint)
    assert isinstance(model.fs.unit_costing["unit"].base_cost_constraint,
                      Constraint)
    assert isinstance(model.fs.unit_costing["unit"].weight_eq,
                      Constraint)

    # Platforms and ladders
    if include_pl:
        assert isinstance(
            model.fs.unit_costing["unit"].base_cost_platforms_ladders, Var)

        assert isinstance(
            model.fs.unit_costing["unit"].cost_platforms_ladders_eq,
            Constraint)

    assert degrees_of_freedom(model) == 0
    assert_units_consistent(model.fs.unit_costing)

    res = solver.solve(model)

    assert check_optimal_termination(res)

    # Test solution for one known case
    if (material_type == VesselMaterial.CS and
            weight_limit == 1 and aspect_ratio_range == 1 and include_pl):
        assert pytest.approx(40012.3, 1e-5) == value(pyunits.convert(
            model.fs.unit_costing["unit"].capital_cost,
            to_units=pyunits.USD2018))


@pytest.mark.component
@pytest.mark.parametrize("tray_material", TrayMaterial)
@pytest.mark.parametrize("tray_type", TrayType)
def test_cost_vessel_trays(model,
                           tray_material,
                           tray_type):
    model.fs.unit.length = Param(initialize=0.00075,
                                 units=pyunits.m)
    model.fs.unit.diameter = Param(initialize=2,
                                   units=pyunits.m)

    model.fs.costing.cost_unit(model.fs.unit,
                               SSLWCosting.cost_vessel,
                               vertical=True,
                               number_of_trays=10,
                               tray_material=tray_material,
                               tray_type=tray_type)

    assert isinstance(model.fs.unit_costing["unit"].shell_thickness, Param)
    assert isinstance(model.fs.unit_costing["unit"].material_factor, Param)
    assert isinstance(model.fs.unit_costing["unit"].material_density, Param)

    assert isinstance(model.fs.unit_costing["unit"].base_cost_per_unit, Var)
    assert isinstance(model.fs.unit_costing["unit"].capital_cost, Var)
    assert isinstance(model.fs.unit_costing["unit"].weight, Var)

    assert isinstance(model.fs.unit_costing["unit"].capital_cost_constraint,
                      Constraint)
    assert isinstance(model.fs.unit_costing["unit"].base_cost_constraint,
                      Constraint)
    assert isinstance(model.fs.unit_costing["unit"].weight_eq,
                      Constraint)

    assert isinstance(
        model.fs.unit_costing["unit"].base_cost_platforms_ladders, Var)

    assert isinstance(
        model.fs.unit_costing["unit"].cost_platforms_ladders_eq,
        Constraint)

    assert isinstance(model.fs.unit_costing["unit"].tray_type_factor,
                      Param)

    assert isinstance(model.fs.unit_costing["unit"].base_cost_trays, Var)
    assert isinstance(model.fs.unit_costing["unit"].tray_material_factor,
                      Var)
    assert isinstance(model.fs.unit_costing["unit"].number_trays_factor,
                      Var)
    assert isinstance(model.fs.unit_costing["unit"].base_cost_per_tray,
                      Var)

    assert isinstance(
        model.fs.unit_costing["unit"].tray_material_factor_eq, Constraint)
    assert isinstance(
        model.fs.unit_costing["unit"].num_tray_factor_constraint,
        Constraint)
    assert isinstance(
        model.fs.unit_costing["unit"].single_tray_cost_constraint,
        Constraint)
    assert isinstance(
        model.fs.unit_costing["unit"].tray_costing_constraint,
        Constraint)

    assert degrees_of_freedom(model) == 0
    assert_units_consistent(model.fs.unit_costing)

    res = solver.solve(model)

    assert check_optimal_termination(res)