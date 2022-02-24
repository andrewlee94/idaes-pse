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

from idaes.core import FlowsheetBlock, UnitModelBlock
from idaes.generic_models.costing import \
    FlowsheetCostingBlock, UnitModelCostingBlock
from idaes.core.util import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom

from idaes.generic_models.costing.SSLW import (SSLWCosting,
                                               HXMaterial,
                                               HXTubeLength,
                                               HXType,
                                               VesselMaterial,
                                               TrayType,
                                               TrayMaterial,
                                               HeaterMaterial,
                                               HeaterSource)


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
    m.fs.unit = UnitModelBlock()

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

    model.fs.unit.costing = UnitModelCostingBlock(default={
        "flowsheet_costing_block": model.fs.costing,
        "costing_method": SSLWCosting.cost_heat_exchanger,
        "costing_method_arguments": {"hx_type": hxtype,
                                     "material_type": material,
                                     "tube_length": tube_length}})

    assert isinstance(model.fs.unit.costing.base_cost_per_unit, Var)
    assert isinstance(model.fs.unit.costing.capital_cost, Var)
    assert isinstance(model.fs.unit.costing.number_of_units, Var)
    assert isinstance(model.fs.unit.costing.pressure_factor, Var)
    assert isinstance(model.fs.unit.costing.material_factor, Var)

    assert isinstance(model.fs.unit.costing.capital_cost_constraint,
                      Constraint)
    assert isinstance(model.fs.unit.costing.hx_material_eqn,
                      Constraint)
    assert isinstance(model.fs.unit.costing.p_factor_eq,
                      Constraint)

    assert degrees_of_freedom(model) == 0
    assert_units_consistent(model.fs.unit.costing)

    res = solver.solve(model)

    assert check_optimal_termination(res)

    # Test solution for one known case
    if (material == HXMaterial.SS_SS and
            hxtype == HXType.Utube and
            tube_length == HXTubeLength.TwelveFoot):
        assert pytest.approx(87704.6, 1e-5) == value(
            model.fs.unit.costing.base_cost_per_unit)
        assert pytest.approx(0.982982, 1e-5) == value(
            model.fs.unit.costing.pressure_factor)
        assert pytest.approx(4.08752, 1e-5) == value(
            model.fs.unit.costing.material_factor)

        assert pytest.approx(529737, 1e-5) == value(pyunits.convert(
            model.fs.unit.costing.capital_cost,
            to_units=pyunits.USD2018))


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

    model.fs.unit.costing = UnitModelCostingBlock(default={
        "flowsheet_costing_block": model.fs.costing,
        "costing_method": SSLWCosting.cost_vessel,
        "costing_method_arguments": {
            "vertical": True,
            "material_type": material_type,
            "weight_limit": weight_limit,
            "aspect_ratio_range": aspect_ratio_range,
            "include_platforms_ladders": include_pl}})

    assert isinstance(model.fs.unit.costing.shell_thickness, Param)
    assert isinstance(model.fs.unit.costing.material_factor, Param)
    assert isinstance(model.fs.unit.costing.material_density, Param)

    assert isinstance(model.fs.unit.costing.base_cost_per_unit, Var)
    assert isinstance(model.fs.unit.costing.capital_cost, Var)
    assert isinstance(model.fs.unit.costing.weight, Var)

    assert isinstance(model.fs.unit.costing.capital_cost_constraint,
                      Constraint)
    assert isinstance(model.fs.unit.costing.base_cost_constraint,
                      Constraint)
    assert isinstance(model.fs.unit.costing.weight_eq,
                      Constraint)

    # Platforms and ladders
    if include_pl:
        assert isinstance(
            model.fs.unit.costing.base_cost_platforms_ladders, Var)

        assert isinstance(
            model.fs.unit.costing.cost_platforms_ladders_eq,
            Constraint)

    assert degrees_of_freedom(model) == 0
    assert_units_consistent(model.fs.unit.costing)

    res = solver.solve(model)

    assert check_optimal_termination(res)

    # Test solution for one known case
    if (material_type == VesselMaterial.CS and
            weight_limit == 1 and aspect_ratio_range == 1 and include_pl):
        assert pytest.approx(40012.3, 1e-5) == value(pyunits.convert(
            model.fs.unit.costing.capital_cost,
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

    model.fs.unit.costing = UnitModelCostingBlock(default={
        "flowsheet_costing_block": model.fs.costing,
        "costing_method": SSLWCosting.cost_vessel,
        "costing_method_arguments": {
            "vertical": True,
            "number_of_trays": 10,
            "tray_material": tray_material,
            "tray_type": tray_type}})

    assert isinstance(model.fs.unit.costing.shell_thickness, Param)
    assert isinstance(model.fs.unit.costing.material_factor, Param)
    assert isinstance(model.fs.unit.costing.material_density, Param)

    assert isinstance(model.fs.unit.costing.base_cost_per_unit, Var)
    assert isinstance(model.fs.unit.costing.capital_cost, Var)
    assert isinstance(model.fs.unit.costing.weight, Var)

    assert isinstance(model.fs.unit.costing.capital_cost_constraint,
                      Constraint)
    assert isinstance(model.fs.unit.costing.base_cost_constraint,
                      Constraint)
    assert isinstance(model.fs.unit.costing.weight_eq,
                      Constraint)

    assert isinstance(
        model.fs.unit.costing.base_cost_platforms_ladders, Var)

    assert isinstance(
        model.fs.unit.costing.cost_platforms_ladders_eq,
        Constraint)

    assert isinstance(model.fs.unit.costing.tray_type_factor,
                      Param)

    assert isinstance(model.fs.unit.costing.base_cost_trays, Var)
    assert isinstance(model.fs.unit.costing.tray_material_factor,
                      Var)
    assert isinstance(model.fs.unit.costing.number_trays_factor,
                      Var)
    assert isinstance(model.fs.unit.costing.base_cost_per_tray,
                      Var)

    assert isinstance(
        model.fs.unit.costing.tray_material_factor_eq, Constraint)
    assert isinstance(
        model.fs.unit.costing.num_tray_factor_constraint,
        Constraint)
    assert isinstance(
        model.fs.unit.costing.single_tray_cost_constraint,
        Constraint)
    assert isinstance(
        model.fs.unit.costing.tray_costing_constraint,
        Constraint)

    assert degrees_of_freedom(model) == 0
    assert_units_consistent(model.fs.unit.costing)

    res = solver.solve(model)

    assert check_optimal_termination(res)


@pytest.mark.component
@pytest.mark.parametrize("material_type", VesselMaterial)
def test_cost_vessel_horizontal(model, material_type):
    model.fs.unit.length = Param(initialize=0.00075,
                                 units=pyunits.m)
    model.fs.unit.diameter = Param(initialize=2,
                                   units=pyunits.m)

    model.fs.unit.costing = UnitModelCostingBlock(default={
        "flowsheet_costing_block": model.fs.costing,
        "costing_method": SSLWCosting.cost_vessel,
        "costing_method_arguments": {
            "vertical": False,
            "material_type": material_type,
            "weight_limit": 1,
            "include_platforms_ladders": False}})

    assert isinstance(model.fs.unit.costing.shell_thickness, Param)
    assert isinstance(model.fs.unit.costing.material_factor, Param)
    assert isinstance(model.fs.unit.costing.material_density, Param)

    assert isinstance(model.fs.unit.costing.base_cost_per_unit, Var)
    assert isinstance(model.fs.unit.costing.capital_cost, Var)
    assert isinstance(model.fs.unit.costing.weight, Var)

    assert isinstance(model.fs.unit.costing.capital_cost_constraint,
                      Constraint)
    assert isinstance(model.fs.unit.costing.base_cost_constraint,
                      Constraint)
    assert isinstance(model.fs.unit.costing.weight_eq,
                      Constraint)

    assert degrees_of_freedom(model) == 0
    assert_units_consistent(model.fs.unit.costing)

    res = solver.solve(model)

    assert check_optimal_termination(res)


@pytest.mark.component
@pytest.mark.parametrize("material_type", HeaterMaterial)
@pytest.mark.parametrize("heat_source", HeaterSource)
def test_cost_fired_heater(model, material_type, heat_source):
    model.fs.unit.heat_duty = Param([0],
                                    initialize=1000,
                                    units=pyunits.kJ/pyunits.s)
    model.fs.unit.control_volume = Block()
    model.fs.unit.control_volume.properties_in = Block(model.fs.time)
    model.fs.unit.control_volume.properties_in[0].pressure = Param(
        initialize=2, units=pyunits.atm)

    model.fs.unit.costing = UnitModelCostingBlock(default={
        "flowsheet_costing_block": model.fs.costing,
        "costing_method": SSLWCosting.cost_fired_heater,
        "costing_method_arguments": {
            "material_type": material_type,
            "heat_source": heat_source}})

    assert isinstance(model.fs.unit.costing.pressure_factor, Var)
    assert isinstance(model.fs.unit.costing.base_cost_per_unit, Var)
    assert isinstance(model.fs.unit.costing.capital_cost, Var)

    assert isinstance(model.fs.unit.costing.capital_cost_constraint,
                      Constraint)
    assert isinstance(model.fs.unit.costing.base_cost_per_unit_eq,
                      Constraint)
    assert isinstance(model.fs.unit.costing.pressure_factor_eq,
                      Constraint)

    assert degrees_of_freedom(model) == 0
    assert_units_consistent(model.fs.unit.costing)

    res = solver.solve(model)

    assert check_optimal_termination(res)


@pytest.mark.component
# @pytest.mark.parametrize("material_type", HeaterMaterial)
# @pytest.mark.parametrize("heat_source", HeaterSource)
def test_cost_pump(model):
    model.fs.unit.work_mechanical = Param([0],
                                          initialize=1000,
                                          units=pyunits.kJ/pyunits.s)
    model.fs.unit.deltaP = Param([0],
                                 initialize=1,
                                 units=pyunits.atm)
    model.fs.unit.control_volume = Block()
    model.fs.unit.control_volume.properties_in = Block(model.fs.time)
    model.fs.unit.control_volume.properties_in[0].dens_mass = Param(
        initialize=1000, units=pyunits.kg/pyunits.m**3)
    model.fs.unit.control_volume.properties_in[0].flow_vol = Param(
        initialize=1, units=pyunits.m**3/pyunits.s)

    model.fs.unit.costing = UnitModelCostingBlock(default={
        "flowsheet_costing_block": model.fs.costing,
        "costing_method": SSLWCosting.cost_pump,
        "costing_method_arguments": {}})

    # assert isinstance(model.fs.unit.costing.pressure_factor, Var)
    # assert isinstance(model.fs.unit.costing.base_cost_per_unit, Var)
    # assert isinstance(model.fs.unit.costing.capital_cost, Var)

    # assert isinstance(model.fs.unit.costing.capital_cost_constraint,
    #                   Constraint)
    # assert isinstance(model.fs.unit.costing.base_cost_per_unit_eq,
    #                   Constraint)
    # assert isinstance(model.fs.unit.costing.pressure_factor_eq,
    #                   Constraint)

    assert degrees_of_freedom(model) == 0
    assert_units_consistent(model.fs.unit.costing)

    res = solver.solve(model)

    assert check_optimal_termination(res)