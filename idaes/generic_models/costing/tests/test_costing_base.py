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
Tests for costing base classes
"""
import pytest

from pyomo.environ import ConcreteModel, Constraint, Set, units as pyunits, Var
from pyomo.util.check_units import (assert_units_consistent,
                                    assert_units_equivalent)

from idaes.core import declare_process_block_class, UnitModelBlockData
from idaes.core.util.exceptions import ConfigurationError
from idaes.core.util.misc import register_units_of_measurement
from idaes.core.util.model_statistics import degrees_of_freedom

from idaes.generic_models.costing import (CostingPackageBase,
                                          FlowsheetCostingBlock,
                                          UnitModelCostingBlock)

# TODO : Tests for cases with multiple costing packages
register_units_of_measurement("USD2010 = [currency]")


class TestCostingPackageBase:
    @pytest.mark.unit
    def test_basic_attributes(self):
        CP = CostingPackageBase

        assert CP.currency_units == []
        assert CP.base_currency is None
        assert CP.base_period is pyunits.year
        assert CP.defined_flows == {}
        assert CP.unit_mapping == {}

    @pytest.mark.unit
    def test_build_global_params(self):
        with pytest.raises(NotImplementedError,
                           match="Derived CostingPackage class has not "
                           "defined a build_global_params method"):
            CostingPackageBase.build_global_params(self)

    @pytest.mark.unit
    def test_build_process_costs(self):
        with pytest.raises(NotImplementedError,
                           match="Derived CostingPackage class has not "
                           "defined a build_process_costs method"):
            CostingPackageBase.build_process_costs(self)

    @pytest.mark.unit
    def test_initialize(self):
        with pytest.raises(NotImplementedError,
                           match="Derived CostingPackage class has not "
                           "defined an initialize method"):
            CostingPackageBase.initialize(self)


# Create some dummy classes to represent inherited unit models
@declare_process_block_class("TypeA")
class TypeAData(UnitModelBlockData):
    def build(self):
        self.class_type = "A"


@declare_process_block_class("TypeB")
class TypeBData(TypeAData):
    def build(self):
        self.class_type = "B"


@declare_process_block_class("TypeC")
class TypeCData(TypeBData):
    def build(self):
        self.class_type = "C"


@declare_process_block_class("TypeD")
class TypeDData(TypeAData):
    def build(self):
        self.class_type = "D"


@declare_process_block_class("TypeE")
class TypeEData(UnitModelBlockData):
    def build(self):
        self.class_type = "E"


class TestCostingPackage(CostingPackageBase):
    defined_flows = {"test_flow_1": 0.2*pyunits.J}

    base_currency = pyunits.USD2010
    base_period = pyunits.year

    @staticmethod
    def build_global_params(self):
        self._bgp = True

    @staticmethod
    def build_process_costs(self):
        self._bpc = True

    @staticmethod
    def initialize(self):
        self._init = True

    def method_1(blk):
        blk.cost_method = 1

    def method_2(blk):
        blk.cost._method = 2

    def method_3(blk):
        blk.cost_method = 3

    def method_4(blk):
        blk.cost_method = 4

    unit_mapping = {TypeA: method_1,
                    TypeB: method_2,
                    TypeC: method_3}


class TestFlowsheetCostingBlock:
    @pytest.mark.unit
    def test_invalid_costing_package(self):
        m = ConcreteModel()

        with pytest.raises(ConfigurationError,
                           match="costing - no costing_package was assigned - "
                           "all FlowsheetCostingBlocks must be assigned a "
                           "costing package."):
            m.costing = FlowsheetCostingBlock()

        class foo:
            pass

        with pytest.raises(
                ValueError,
                match="invalid value for configuration 'costing_package'"):
            m.costing = FlowsheetCostingBlock(
                default={"costing_package": foo})

    @pytest.mark.unit
    def test_costing_package_no_base_currency(self):
        class TestCostingPackage2(TestCostingPackage):
            base_currency = None

        m = ConcreteModel()

        with pytest.raises(
                ValueError,
                match="costing - costing package has not specified the base "
                "currency units to use for costing."):
            m.costing = FlowsheetCostingBlock(
                default={"costing_package": TestCostingPackage2})

    @pytest.fixture(scope="class")
    def costing(self):
        m = ConcreteModel()
        m.costing = FlowsheetCostingBlock(
            default={"costing_package": TestCostingPackage})

        return m

    @pytest.mark.unit
    def test_basic_attributes(self, costing):
        assert costing.costing._registered_unit_models == []
        assert isinstance(costing.costing.flow_types, Set)
        assert len(costing.costing.flow_types) == 1
        assert "test_flow_1" in costing.costing.flow_types
        assert costing.costing._registered_flows == {
            "test_flow_1": []}

        assert costing.costing._costing_methods_map == {
            TypeAData: TestCostingPackage.method_1,
            TypeBData: TestCostingPackage.method_2,
            TypeCData: TestCostingPackage.method_3}

        # Check that test_flow_1 was properly defined
        assert isinstance(costing.costing.test_flow_1_cost, Var)
        assert costing.costing.test_flow_1_cost.value == 0.2
        assert_units_equivalent(costing.costing.test_flow_1_cost.get_units(),
                                pyunits.J)

        # Test that build_global_parameters was called successfully
        assert costing.costing._bgp

    @pytest.mark.unit
    def test_register_flow_type(self, costing):
        costing.costing.register_flow_type(
            "test_flow", 42*pyunits.USD2010/pyunits.mol)

        assert isinstance(costing.costing.test_flow_cost, Var)
        assert costing.costing.test_flow_cost.value == 42
        assert_units_equivalent(costing.costing.test_flow_cost.get_units(),
                                pyunits.USD2010/pyunits.mol)
        assert "test_flow" in costing.costing.flow_types

        assert costing.costing._registered_flows == {
            "test_flow_1": [],
            "test_flow": []}

    @pytest.mark.unit
    def test_cost_flow_invalid_type(self, costing):
        with pytest.raises(ValueError,
                           match="foo is not a recognized flow type. Please "
                           "check your spelling and that the flow type has "
                           "been registered with the FlowsheetCostingBlock."):
            costing.costing.cost_flow(42, "foo")

    @pytest.mark.unit
    def test_cost_flow_indexed_var(self, costing):
        costing.indexed_var = Var([1, 2, 3],
                                  initialize=1,
                                  units=pyunits.mol/pyunits.s)
        with pytest.raises(TypeError,
                           match="indexed_var is an indexed Var. Flow costing "
                           "only supports unindexed Vars."):
            costing.costing.cost_flow(costing.indexed_var, "test_flow")

    @pytest.mark.unit
    def test_cost_flow_unbounded_var(self, costing):
        with pytest.raises(ValueError,
                           match="indexed_var\[1\] has a lower bound of less "
                           "than zero. Costing requires that all flows have a "
                           "lower bound equal to or greater than zero to "
                           "avoid negative costs."):
            costing.costing.cost_flow(costing.indexed_var[1], "test_flow")

    @pytest.mark.unit
    def test_cost_flow_var(self, costing):
        costing.indexed_var[1].setlb(0)

        costing.costing.cost_flow(costing.indexed_var[1], "test_flow")

        assert (costing.indexed_var[1] in
                costing.costing._registered_flows["test_flow"])

    @pytest.mark.unit
    def test_cost_flow_unbounded_expr(self, costing):
        with pytest.raises(ValueError,
                           match="flow_expr is an expression with a lower "
                           "bound of less than zero. Costing requires that "
                           "all flows have a lower bound equal to or greater "
                           "than zero to avoid negative costs."):
            costing.costing.cost_flow(-costing.indexed_var[2], "test_flow")

    @pytest.mark.unit
    def test_cost_flow_expr(self, costing):
        costing.indexed_var[2].setub(0)

        costing.costing.cost_flow(-costing.indexed_var[2], "test_flow")

        assert str(-costing.indexed_var[2]) == str(
                costing.costing._registered_flows["test_flow"][-1])

    @pytest.mark.unit
    def test_get_costing_method_for(self, costing):
        costing.unit_a = TypeA()
        costing.unit_b = TypeB()
        costing.unit_c = TypeC()
        costing.unit_d = TypeD()
        costing.unit_e = TypeE()

        assert isinstance(costing.unit_a, TypeAData)

        assert costing.costing._get_costing_method_for(costing.unit_a) is \
            TestCostingPackage.method_1

        assert costing.costing._get_costing_method_for(costing.unit_b) is \
            TestCostingPackage.method_2

        assert costing.costing._get_costing_method_for(costing.unit_c) is \
            TestCostingPackage.method_3

        # TypeD not registered with property package, but inherits from TypeA
        # Should get method_1
        assert costing.costing._get_costing_method_for(costing.unit_d) is \
            TestCostingPackage.method_1

        # TypeE not registered with property package and no inheritance
        # Should return RuntimeError
        with pytest.raises(RuntimeError,
                           match="Could not identify default costing method "
                           "for unit_e. This implies the unit model's class "
                           "and parent classes do not exist in the default "
                           "mapping provided by the costing package. Please "
                           "provide a specific costing method for this unit."):
            costing.costing._get_costing_method_for(costing.unit_e)

    @pytest.mark.unit
    def test_cost_unit_first(self, costing):
        assert not hasattr(costing, "unit_costing_set")
        assert not hasattr(costing, "unit_costing")

        costing.costing.cost_unit(costing.unit_a)

        assert isinstance(costing.unit_costing_set, Set)
        assert len(costing.unit_costing_set) == 1
        assert "unit_a" in costing.unit_costing_set

        assert isinstance(costing.unit_costing, UnitModelCostingBlock)
        assert "unit_a" in costing.unit_costing
        assert costing.unit_costing["unit_a"].cost_method == 1

        assert costing.unit_a in costing.costing._registered_unit_models

    @pytest.mark.unit
    def test_cost_unit_duplicate(self, costing):
        with pytest.raises(RuntimeError,
                           match="Unit model unit_a already appears in the "
                           "Set of costed units. Each unit model can only be "
                           "costed once."):
            costing.costing.cost_unit(costing.unit_a)

    @pytest.mark.unit
    def test_del_unit_costing(self, costing):
        costing.costing.del_unit_costing(costing.unit_a)

        assert isinstance(costing.unit_costing_set, Set)
        assert len(costing.unit_costing_set) == 0

        assert isinstance(costing.unit_costing, UnitModelCostingBlock)
        assert len(costing.unit_costing) == 0

        assert costing.unit_a not in costing.costing._registered_unit_models

    @pytest.mark.unit
    def test_del_unit_costing_unregistered(self, costing):
        with pytest.raises(RuntimeError,
                           match="unit_a was not registered with this "
                           "FlowsheetCostingBlock. del_unit_costing can only "
                           "be used from the block with which the unit model "
                           "is registered for costing."):
            costing.costing.del_unit_costing(costing.unit_a)

    @pytest.mark.unit
    def test_cost_unit_custom_method(self, costing):
        def custom_method(blk):
            blk.capital_cost = Var(initialize=1,
                                   bounds=(0, 1e10),
                                   units=pyunits.USD2010)
            blk.fixed_operating_cost = Var(initialize=1,
                                           bounds=(0, 1e10),
                                           units=pyunits.USD2010/pyunits.year)
            blk.variable_operating_cost = Var(
                initialize=1,
                bounds=(0, 1e10),
                units=pyunits.USD2010/pyunits.year)

            blk.capital_cost_constraint = Constraint(
                expr=blk.capital_cost == 4.2e6*pyunits.USD2010)
            blk.fixed_operating_cost_constraint = Constraint(
                expr=blk.fixed_operating_cost ==
                1e2*pyunits.USD2010/pyunits.year)
            blk.variable_operating_cost_constraint = Constraint(
                expr=blk.variable_operating_cost ==
                7e4*pyunits.USD2010/pyunits.year)

        costing.costing.cost_unit(costing.unit_a, method=custom_method)

        assert isinstance(costing.unit_costing_set, Set)
        assert len(costing.unit_costing_set) == 1
        assert "unit_a" in costing.unit_costing_set

        assert isinstance(costing.unit_costing, UnitModelCostingBlock)
        assert "unit_a" in costing.unit_costing

        assert costing.unit_a in costing.costing._registered_unit_models

        assert isinstance(costing.unit_costing["unit_a"].capital_cost, Var)
        assert isinstance(
            costing.unit_costing["unit_a"].variable_operating_cost, Var)
        assert isinstance(
            costing.unit_costing["unit_a"].fixed_operating_cost, Var)

    @pytest.mark.unit
    def test_cost_unit_capital_cost_not_var(self, costing):
        def dummy_method(blk):
            blk.capital_cost = "foo"

        with pytest.raises(TypeError,
                           match="unit_b capital_cost component must be a "
                           "Var. Please check the costingpackage you are "
                           "using to ensure that all costing components are "
                           "declared as variables."):
            costing.costing.cost_unit(costing.unit_b, method=dummy_method)

        # Clean up for next test
        costing.costing.del_unit_costing(costing.unit_b)

    @pytest.mark.unit
    def test_cost_unit_capital_cost_lb(self, costing):
        def dummy_method(blk):
            blk.capital_cost = Var()

        with pytest.raises(ValueError,
                           match="unit_b capital_cost component has a lower "
                           "bound less than zero. All costing components are "
                           "required to have lower bounds of 0 or greater to "
                           "avoid negative costs."):
            costing.costing.cost_unit(costing.unit_b, method=dummy_method)

        # Clean up for next test
        costing.costing.del_unit_costing(costing.unit_b)

    @pytest.mark.unit
    def test_cost_unit_fixed_operating_cost_not_var(self, costing):
        def dummy_method(blk):
            blk.fixed_operating_cost = "foo"

        with pytest.raises(TypeError,
                           match="unit_b fixed_operating_cost component must "
                           "be a Var. Please check the costingpackage you are "
                           "using to ensure that all costing components are "
                           "declared as variables."):
            costing.costing.cost_unit(costing.unit_b, method=dummy_method)

        # Clean up for next test
        costing.costing.del_unit_costing(costing.unit_b)

    @pytest.mark.unit
    def test_cost_unit_fixed_operating_cost_lb(self, costing):
        def dummy_method(blk):
            blk.fixed_operating_cost = Var()

        with pytest.raises(ValueError,
                           match="unit_b fixed_operating_cost component has "
                           "a lower bound less than zero. All costing "
                           "components are required to have lower bounds of 0 "
                           "or greater to avoid negative costs."):
            costing.costing.cost_unit(costing.unit_b, method=dummy_method)

        # Clean up for next test
        costing.costing.del_unit_costing(costing.unit_b)

    @pytest.mark.unit
    def test_cost_unit_variable_operating_cost_not_var(self, costing):
        def dummy_method(blk):
            blk.variable_operating_cost = "foo"

        with pytest.raises(TypeError,
                           match="unit_b variable_operating_cost component "
                           "must be a Var. Please check the costingpackage "
                           "you are using to ensure that all costing "
                           "components are declared as variables."):
            costing.costing.cost_unit(costing.unit_b, method=dummy_method)

        # Clean up for next test
        costing.costing.del_unit_costing(costing.unit_b)

    @pytest.mark.unit
    def test_cost_unit_variable_operating_cost_lb(self, costing):
        def dummy_method(blk):
            blk.variable_operating_cost = Var()

        with pytest.raises(ValueError,
                           match="unit_b variable_operating_cost component "
                           "has a lower bound less than zero. All costing "
                           "components are required to have lower bounds of 0 "
                           "or greater to avoid negative costs."):
            costing.costing.cost_unit(costing.unit_b, method=dummy_method)

        # Clean up for next test
        costing.costing.del_unit_costing(costing.unit_b)

    @pytest.mark.unit
    def test_cost_process(self, costing):
        costing.costing.cost_process()

        # Check that build_process_costs was called from costing package
        assert costing.costing._bpc

        # Then check aggregation
        assert isinstance(costing.costing.aggregate_capital_cost, Var)
        assert str(costing.costing.aggregate_capital_cost.get_units()) == str(
            pyunits.USD2010)
        assert isinstance(costing.costing.aggregate_capital_cost_constraint,
                          Constraint)

        assert isinstance(costing.costing.aggregate_fixed_operating_cost, Var)
        assert str(pyunits.USD2010/pyunits.year) == str(
            costing.costing.aggregate_fixed_operating_cost.get_units())
        assert isinstance(
            costing.costing.aggregate_fixed_operating_cost_constraint,
            Constraint)

        assert isinstance(
            costing.costing.aggregate_variable_operating_cost, Var)
        assert str(pyunits.USD2010/pyunits.year) == str(
            costing.costing.aggregate_variable_operating_cost.get_units())
        assert isinstance(
            costing.costing.aggregate_variable_operating_cost_constraint,
            Constraint)

        assert isinstance(
            costing.costing.aggregate_flow_test_flow, Var)
        assert str(pyunits.mol/pyunits.s) == str(
            costing.costing.aggregate_flow_test_flow.get_units())
        assert isinstance(
            costing.costing.aggregate_flow_test_flow_constraint,
            Constraint)

        # We also have a test_flow_1 type registered, but no flows costed
        # This should have been skipped
        assert not hasattr(costing.costing, "aggregate_flow_test_flow_1")
        assert not hasattr(costing.costing,
                           "aggregate_flow_test_flow_1_constraint")

        assert isinstance(
            costing.costing.aggregate_flow_costs, Var)
        assert str(pyunits.USD2010/pyunits.year) == str(
            costing.costing.aggregate_flow_costs.get_units())
        assert len(costing.costing.aggregate_flow_costs) == 2
        assert isinstance(costing.costing.aggregate_flow_costs_constraint,
                          Constraint)
        assert len(costing.costing.aggregate_flow_costs_constraint) == 2

    @pytest.mark.unit
    def test_unit_consistency(self, costing):
        assert_units_consistent(costing)

    @pytest.mark.unit
    def test_degrees_of_freedom(self, costing):
        costing.indexed_var[1].fix(2)
        costing.indexed_var[2].fix(-3)
        costing.indexed_var[3].fix(0)

        assert degrees_of_freedom(costing) == 0

    @pytest.mark.unit
    def test_initialize(self, costing):
        costing.costing.initialize()

        # Check that initialize was called from costing package
        assert costing.costing._init

        # Check that unit-level vars were initialized
        assert costing.unit_costing["unit_a"].capital_cost.value == 4.2e6
        assert costing.unit_costing["unit_a"].fixed_operating_cost.value == \
            100
        assert \
            costing.unit_costing["unit_a"].variable_operating_cost.value == 7e4

        # Check that aggregate vars were initialized
        # Capital and operating costs should equal the unit level ones
        assert costing.costing.aggregate_capital_cost.value == 4.2e6
        assert costing.costing.aggregate_fixed_operating_cost.value == \
            100
        assert \
            costing.costing.aggregate_variable_operating_cost.value == 7e4

        assert costing.costing.aggregate_flow_test_flow.value == 5

        assert pytest.approx(
            costing.costing.aggregate_flow_costs["test_flow"].value,
            rel=1e-12) == (
                pyunits.convert_value(5*42,
                                      from_units=1/pyunits.s,
                                      to_units=1/pyunits.year))
        assert costing.costing.aggregate_flow_costs["test_flow_1"].value == (
            0)


@pytest.mark.unit
def test_UnitModelCostingBlock():
    m = ConcreteModel()
    m.costing = UnitModelCostingBlock([1, 2, 3])

    assert isinstance(m.costing, UnitModelCostingBlock)
    assert m.costing.is_indexed()
    for i in [1, 2, 3]:
        assert i in m.costing