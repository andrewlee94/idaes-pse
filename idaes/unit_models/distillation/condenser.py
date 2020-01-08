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
Condenser model for distillation.

While the condenser model (both total and partial), is fairly simple, a major
portion of this code has gone into making this generic and be able to handle
different state variables and the associated splits.
"""

__author__ = "Jaffer Ghouse"

import logging
from pandas import DataFrame
from enum import Enum

# Import Pyomo libraries
from pyomo.common.config import ConfigBlock, ConfigValue, In
from pyomo.network import Port
from pyomo.environ import Reference, Expression, Var, Constraint, value, \
    TerminationCondition

# Import IDAES cores
import idaes.logger as idaeslog
from idaes.core import (ControlVolume0DBlock,
                        declare_process_block_class,
                        EnergyBalanceType,
                        MomentumBalanceType,
                        MaterialBalanceType,
                        UnitModelBlockData,
                        useDefault)
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.config import is_physical_parameter_block
from idaes.core.util.misc import add_object_reference
from idaes.core.util.exceptions import PropertyPackageError, \
    ConfigurationError, PropertyNotSupportedError

_log = idaeslog.getLogger(__name__)


class CondenserType(Enum):
    totalCondenser = 0
    partialCondenser = 1


class TemperatureSpec(Enum):
    atBubblePoint = 0
    customTemperature = 1


@declare_process_block_class("Condenser")
class CondenserData(UnitModelBlockData):
    """
    Condenser unit for distillation model.
    Unit model to condense (total/partial) the vapor from the top tray of
    the distillation column.
    """
    CONFIG = UnitModelBlockData.CONFIG()
    CONFIG.declare("condenser_type", ConfigValue(
        default=CondenserType.totalCondenser,
        domain=In(CondenserType),
        description="Type of condenser flag",
        doc="""Indicates what type of condenser should be constructed,
**default** - CondenserType.totalCondenser.
**Valid values:** {
**CondenserType.totalCondenser** - Incoming vapor from top tray is condensed
to all liquid,
**CondenserType.partialCondenser** - Incoming vapor from top tray is
partially condensed to a vapor and liquid stream.}"""))
    CONFIG.declare("temperature_spec", ConfigValue(
        default=None,
        domain=In(TemperatureSpec),
        description="Temperature spec for the condenser",
        doc="""Temperature specification for the condenser,
**default** - TemperatureSpec.none
**Valid values:** {
**TemperatureSpec.none** - No spec is selected,
**TemperatureSpec.atBubblePoint** - Condenser temperature set at
bubble point i.e. total condenser,
**TemperatureSpec.customTemperature** - Condenser temperature at
user specified temperature.}"""))
    CONFIG.declare("material_balance_type", ConfigValue(
        default=MaterialBalanceType.useDefault,
        domain=In(MaterialBalanceType),
        description="Material balance construction flag",
        doc="""Indicates what type of mass balance should be constructed,
**default** - MaterialBalanceType.componentPhase.
**Valid values:** {
**MaterialBalanceType.none** - exclude material balances,
**MaterialBalanceType.componentPhase** - use phase component balances,
**MaterialBalanceType.componentTotal** - use total component balances,
**MaterialBalanceType.elementTotal** - use total element balances,
**MaterialBalanceType.total** - use total material balance.}"""))
    CONFIG.declare("energy_balance_type", ConfigValue(
        default=EnergyBalanceType.useDefault,
        domain=In(EnergyBalanceType),
        description="Energy balance construction flag",
        doc="""Indicates what type of energy balance should be constructed,
**default** - EnergyBalanceType.enthalpyTotal.
**Valid values:** {
**EnergyBalanceType.none** - exclude energy balances,
**EnergyBalanceType.enthalpyTotal** - single enthalpy balance for material,
**EnergyBalanceType.enthalpyPhase** - enthalpy balances for each phase,
**EnergyBalanceType.energyTotal** - single energy balance for material,
**EnergyBalanceType.energyPhase** - energy balances for each phase.}"""))
    CONFIG.declare("momentum_balance_type", ConfigValue(
        default=MomentumBalanceType.pressureTotal,
        domain=In(MomentumBalanceType),
        description="Momentum balance construction flag",
        doc="""Indicates what type of momentum balance should be constructed,
**default** - MomentumBalanceType.pressureTotal.
**Valid values:** {
**MomentumBalanceType.none** - exclude momentum balances,
**MomentumBalanceType.pressureTotal** - single pressure balance for material,
**MomentumBalanceType.pressurePhase** - pressure balances for each phase,
**MomentumBalanceType.momentumTotal** - single momentum balance for material,
**MomentumBalanceType.momentumPhase** - momentum balances for each phase.}"""))
    CONFIG.declare("has_pressure_change", ConfigValue(
        default=False,
        domain=In([True, False]),
        description="Pressure change term construction flag",
        doc="""Indicates whether terms for pressure change should be
constructed,
**default** - False.
**Valid values:** {
**True** - include pressure change terms,
**False** - exclude pressure change terms.}"""))
    CONFIG.declare("property_package", ConfigValue(
        default=useDefault,
        domain=is_physical_parameter_block,
        description="Property package to use for control volume",
        doc="""Property parameter object used to define property calculations,
**default** - useDefault.
**Valid values:** {
**useDefault** - use default package from parent model or flowsheet,
**PropertyParameterObject** - a PropertyParameterBlock object.}"""))
    CONFIG.declare("property_package_args", ConfigBlock(
        implicit=True,
        description="Arguments to use for constructing property packages",
        doc="""A ConfigBlock with arguments to be passed to a property block(s)
and used when constructing these,
**default** - None.
**Valid values:** {
see property package for documentation.}"""))

    def build(self):
        """Build the model.

        Args:
            None
        Returns:
            None
        """
        # Call UnitModel.build to setup dynamics
        super(CondenserData, self).build()

        # Check config arguments
        if self.config.temperature_spec is None:
            raise ConfigurationError("temperature_spec config argument "
                                     "has not been specified. Please select "
                                     "a valid option.")
        if (self.config.condenser_type == CondenserType.partialCondenser) and \
                (self.config.temperature_spec ==
                 TemperatureSpec.atBubblePoint):
            raise ConfigurationError("condenser_type set to partial but "
                                     "temperature_spec set to atBubblePoint. "
                                     "Select customTemperature and specify "
                                     "outlet temperature.")

        # Add Control Volume for the condenser
        self.control_volume = ControlVolume0DBlock(default={
            "dynamic": self.config.dynamic,
            "has_holdup": self.config.has_holdup,
            "property_package": self.config.property_package,
            "property_package_args": self.config.property_package_args})

        self.control_volume.add_state_blocks(
            has_phase_equilibrium=True)

        self.control_volume.add_material_balances(
            balance_type=self.config.material_balance_type,
            has_phase_equilibrium=True)

        self.control_volume.add_energy_balances(
            balance_type=self.config.energy_balance_type,
            has_heat_transfer=True)

        self.control_volume.add_momentum_balances(
            balance_type=self.config.momentum_balance_type,
            has_pressure_change=self.config.has_pressure_change)

        self._make_ports()

        if self.config.condenser_type == CondenserType.totalCondenser:

            self._make_splits_total_condenser()

            if (self.config.temperature_spec == TemperatureSpec.atBubblePoint):
                # Option 1: if true, condition for total condenser
                # (T_cond = T_bubble)
                # Option 2: if this is false, then user has selected
                # custom temperature spec and needs to fix an outlet
                # temperature.
                def rule_total_cond(self, t):
                    return self.control_volume.properties_out[t].\
                        temperature == self.control_volume.properties_out[t].\
                        temperature_bubble
                self.eq_total_cond_spec = Constraint(self.flowsheet().time,
                                                     rule=rule_total_cond)

        else:
            self._make_splits_partial_condenser()

        # Add object reference to variables of the control volume
        # Reference to the heat duty
        add_object_reference(self, "heat_duty", self.control_volume.heat)

        # Reference to the pressure drop (if set to True)
        if self.config.has_pressure_change:
            add_object_reference(self, "deltaP", self.control_volume.deltaP)

    def _make_ports(self):

        # Add Ports for the condenser
        # Inlet port (the vapor from the top tray)
        self.add_inlet_port()

        # Outlet ports that always exist irrespective of condenser type
        self.reflux = Port(noruleinit=True, doc="Reflux stream that is"
                           " returned to the top tray.")
        self.distillate = Port(noruleinit=True, doc="Distillate stream that is"
                               " the top product.")

        if self.config.condenser_type == CondenserType.partialCondenser:
            self.vapor_outlet = Port(noruleinit=True,
                                     doc="Vapor outlet port from a "
                                     "partial condenser")
        # Add codnenser specific variables
        self.reflux_ratio = Var(initialize=1, doc="reflux ratio for "
                                "the condenser")

    def _make_splits_total_condenser(self):
        # Get dict of Port members and names
        member_list = self.control_volume.\
            properties_out[0].define_port_members()

        # Create references and populate the reflux, distillate ports
        for k in member_list:
            # Create references and populate the intensive variables
            if "flow" not in member_list[k].local_name:
                if not member_list[k].is_indexed():
                    var = self.control_volume.properties_out[:].\
                        component(member_list[k].local_name)
                else:
                    var = self.control_volume.properties_out[:].\
                        component(member_list[k].local_name)[...]

                # add the reference and variable name to the reflux port
                self.reflux.add(Reference(var), k)

                # add the reference and variable name to the distillate port
                self.distillate.add(Reference(var), k)

            elif "flow" in member_list[k].local_name:
                # Create references and populate the extensive variables
                # This is for vars that are not indexed
                if not member_list[k].is_indexed():
                    # Expression for reflux flow and relation to the
                    # reflux_ratio variable

                    def rule_reflux_flow(self, t):
                        return self.control_volume.properties_out[t].\
                            component(member_list[k].local_name) * \
                            (self.reflux_ratio / (1 + self.reflux_ratio))
                    self.e_reflux_flow = Expression(self.flowsheet().time,
                                                    rule=rule_reflux_flow)
                    self.reflux.add(self.e_reflux_flow, k)

                    # Expression for distillate flow and relation to the
                    # reflux_ratio variable
                    def rule_distillate_flow(self, t):
                        return self.control_volume.properties_out[t].\
                            component(member_list[k].local_name) / \
                            (1 + self.reflux_ratio)
                    self.e_distillate_flow = Expression(
                        self.flowsheet().time, rule=rule_distillate_flow)
                    self.distillate.add(self.e_distillate_flow, k)
                else:
                    # Create references and populate the extensive variables
                    # This is for vars that are indexed by phase, comp or both.
                    index_set = member_list[k].index_set()

                    def rule_reflux_flow(self, t, *args):
                        return self.control_volume.properties_out[t].\
                            component(member_list[k].local_name)[args] * \
                            (self.reflux_ratio / (1 + self.reflux_ratio))
                    self.e_reflux_flow = Expression(self.flowsheet().time,
                                                    index_set,
                                                    rule=rule_reflux_flow)
                    self.reflux.add(self.e_reflux_flow, k)

                    def rule_distillate_flow(self, t, *args):
                        return self.control_volume.properties_out[t].\
                            component(member_list[k].local_name)[args] / \
                            (1 + self.reflux_ratio)
                    self.e_distillate_flow = Expression(
                        self.flowsheet().time, index_set,
                        rule=rule_distillate_flow)
                    self.distillate.add(self.e_distillate_flow, k)

            else:
                raise PropertyNotSupportedError(
                    "Unrecognized names for flow variables encountered while "
                    "building the condenser ports.")

    def _make_splits_partial_condenser(self):
        # Get dict of Port members and names
        member_list = self.control_volume.\
            properties_out[0].define_port_members()

        # Create references and populate the reflux, distillate ports
        for k in member_list:
            # Create references and populate the intensive variables
            if "flow" not in k and "frac" not in k and "enth" not in k:
                if not member_list[k].is_indexed():
                    var = self.control_volume.properties_out[:].\
                        component(member_list[k].local_name)
                else:
                    var = self.control_volume.properties_out[:].\
                        component(member_list[k].local_name)[...]

                # add the reference and variable name to the reflux port
                self.reflux.add(Reference(var), k)

                # add the reference and variable name to the distillate port
                self.distillate.add(Reference(var), k)

                # add the reference and variable name to the
                # vapor outlet port
                self.vapor_outlet.add(Reference(var), k)

            elif "frac" in k and ("mole" in k or "mass" in k):

                # Mole/mass frac is typically indexed
                index_set = member_list[k].index_set()

                # if state var is not mole/mass frac by phase
                if "phase" not in k:
                    # Assuming the state block has the var
                    # "mole_frac_phase_comp". Valid if VLE is supported
                    # Create a string "mole_frac_phase_comp" or
                    # "mass_frac_phase_comp". Cannot directly append phase
                    # to k as the naming convention is phase followed
                    # by comp
                    str_split = k.split('_')
                    local_name = '_'.join(str_split[0:2]) + \
                        "_phase" + "_" + str_split[2]

                    # Rule for liquid fraction
                    def rule_liq_frac(self, t, i):
                        return self.control_volume.properties_out[t].\
                            component(local_name)["Liq", i]
                    self.e_liq_frac = Expression(
                        self.flowsheet().time, index_set,
                        rule=rule_liq_frac)

                    # Rule for vapor fraction
                    def rule_vap_frac(self, t, i):
                        return self.control_volume.properties_out[t].\
                            component(local_name)["Vap", i]
                    self.e_vap_frac = Expression(
                        self.flowsheet().time, index_set,
                        rule=rule_vap_frac)

                    # add the reference and variable name to the reflux port
                    self.reflux.add(self.e_liq_frac, k)

                    # add the reference and variable name to the
                    # distillate port
                    self.distillate.add(self.e_liq_frac, k)

                    # add the reference and variable name to the
                    # vapor port
                    self.vapor_outlet.add(self.e_vap_frac, k)
                else:

                    # Assumes mole_frac_phase or mass_frac_phase exist as
                    # state vars in the port and therefore access directly
                    # from the state block.
                    var = self.control_volume.properties_out[:].\
                        component(member_list[k].local_name)[...]

                    # add the reference and variable name to the reflux port
                    self.reflux.add(Reference(var), k)

                    # add the reference and variable name to the distillate port
                    self.distillate.add(Reference(var), k)
            elif "flow" in k:
                if "phase" not in k:

                    # Assumes that here the var is total flow or component
                    # flow. However, need to extract the flow by phase from
                    # the state block. Expects to find the var
                    # flow_mol_phase or flow_mass_phase in the state block.

                    # Check if it is not indexed by component list and this
                    # is total flow
                    if not member_list[k].is_indexed():
                        # if state var is not flow_mol/flow_mass
                        # by phase
                        local_name = str(member_list[k].local_name) + \
                            "_phase"

                        # Rule for vap flow
                        def rule_vap_flow(self, t):
                            return self.control_volume.properties_out[t].\
                                component(local_name)["Vap"]
                        self.e_vap_flow = Expression(
                            self.flowsheet().time,
                            rule=rule_vap_flow)

                        # Rule to link the liq flow to the reflux
                        def rule_reflux_flow(self, t):
                            return self.control_volume.properties_out[t].\
                                component(local_name)["Liq"] * \
                                (self.reflux_ratio / (1 + self.reflux_ratio))
                        self.e_reflux_flow = Expression(
                            self.flowsheet().time,
                            rule=rule_reflux_flow)

                        # Rule to link the liq flow to the distillate
                        def rule_distillate_flow(self, t):
                            return self.control_volume.properties_out[t].\
                                component(local_name)["Liq"] / \
                                (1 + self.reflux_ratio)
                        self.e_distillate_flow = Expression(
                            self.flowsheet().time,
                            rule=rule_distillate_flow)

                    else:
                        # when it is flow comp indexed by component list
                        str_split = \
                            str(member_list[k].local_name).split("_")
                        if len(str_split) == 3 and str_split[-1] == "comp":
                            local_name = str_split[0] + "_" + \
                                str_split[1] + "_phase_" + "comp"

                        # Get the indexing set i.e. component list
                        index_set = member_list[k].index_set()

                        # Rule for vap flow
                        def rule_vap_flow(self, t, i):
                            return self.control_volume.properties_out[t].\
                                component(local_name)["Vap", i]
                        self.e_vap_flow = Expression(
                            self.flowsheet().time, index_set,
                            rule=rule_vap_flow)

                        # Rule to link the liq flow to the reflux
                        def rule_reflux_flow(self, t, i):
                            return self.control_volume.properties_out[t].\
                                component(local_name)["Liq", i] * \
                                (self.reflux_ratio / (1 + self.reflux_ratio))
                        self.e_reflux_flow = Expression(
                            self.flowsheet().time, index_set,
                            rule=rule_reflux_flow)

                        # Rule to link the liq flow to the distillate
                        def rule_distillate_flow(self, t, i):
                            return self.control_volume.properties_out[t].\
                                component(local_name)["Liq", i] / \
                                (1 + self.reflux_ratio)
                        self.e_distillate_flow = Expression(
                            self.flowsheet().time, index_set,
                            rule=rule_distillate_flow)

                    # add the reference and variable name to the reflux port
                    self.reflux.add(self.e_reflux_flow, k)

                    # add the reference and variable name to the
                    # distillate port
                    self.distillate.add(self.e_distillate_flow, k)

                    # add the reference and variable name to the
                    # distillate port
                    self.vapor_outlet.add(self.e_vap_flow, k)
            elif "enth" in k:
                if "phase" not in k:
                    # assumes total mixture enthalpy (enth_mol or enth_mass)
                    # and hence should not be indexed by phase
                    if not member_list[k].is_indexed():
                        # if state var is not enth_mol/enth_mass
                        # by phase, add _phase string to extract the right
                        # value from the state block
                        local_name = str(member_list[k].local_name) + \
                            "_phase"
                    else:
                        raise PropertyPackageError(
                            "Enthalpy is indexed but the variable "
                            "name does not reflect the presence of an index. "
                            "Please follow the naming convention outlined "
                            "in the documentation for state variables.")

                    # Rule for vap enthalpy. Setting the enthalpy to the
                    # enth_mol_phase['Vap'] value from the state block
                    def rule_vap_enth(self, t):
                        return self.control_volume.properties_out[t].\
                            component(local_name)["Vap"]
                    self.e_vap_enth = Expression(
                        self.flowsheet().time,
                        rule=rule_vap_enth)

                    # Rule to link the liq enthalpy to the reflux.
                    # Setting the enthalpy to the
                    # enth_mol_phase['Liq'] value from the state block
                    def rule_reflux_enth(self, t):
                        return self.control_volume.properties_out[t].\
                            component(local_name)["Liq"]
                    self.e_reflux_enth = Expression(
                        self.flowsheet().time,
                        rule=rule_reflux_enth)

                    # Rule to link the liq flow to the distillate.
                    # Setting the enthalpy to the
                    # enth_mol_phase['Liq'] value from the state block
                    def rule_distillate_enth(self, t):
                        return self.control_volume.properties_out[t].\
                            component(local_name)["Liq"]
                    self.e_distillate_enth = Expression(
                        self.flowsheet().time,
                        rule=rule_distillate_enth)

                    # add the reference and variable name to the reflux port
                    self.reflux.add(self.e_reflux_enth, k)

                    # add the reference and variable name to the
                    # distillate port
                    self.distillate.add(self.e_distillate_enth, k)

                    # add the reference and variable name to the
                    # distillate port
                    self.vapor_outlet.add(self.e_vap_enth, k)
                elif "phase" in k:
                    # assumes enth_mol_phase or enth_mass_phase.
                    # This is an intensive property, you create a direct
                    # reference irrespective of the reflux, distillate and
                    # vap_outlet

                    # Rule for vap flow
                    if not member_list[k].is_indexed():
                        var = self.control_volume.properties_out[:].\
                            component(member_list[k].local_name)
                    else:
                        var = self.control_volume.properties_out[:].\
                            component(member_list[k].local_name)[...]

                    # add the reference and variable name to the reflux port
                    self.reflux.add(Reference(var), k)

                    # add the reference and variable name to the distillate port
                    self.distillate.add(Reference(var), k)

                    # add the reference and variable name to the
                    # vapor outlet port
                    self.vapor_outlet.add(Reference(var), k)
                else:
                    raise PropertyNotSupportedError(
                        "Unrecognized enthalpy state variable encountered "
                        "while building ports for the condenser. Only total "
                        "mixture enthalpy or enthalpy by phase are supported.")

    def initialize(self, solver=None, outlvl=idaeslog.NOTSET):

        # TODO: Fix the inlets to the condenser to the vapor flow from
        # the top tray or take it as an argument to this method.

        init_log = idaeslog.getInitLogger(self.name, outlvl)
        solve_log = idaeslog.getSolveLogger(self.name, outlvl)

        if self.config.temperature_spec == TemperatureSpec.customTemperature:
            if degrees_of_freedom(self) != 0:
                raise ConfigurationError(
                    "Degrees of freedom is not 0 during initialization. "
                    "Check if outlet temperature has been fixed in addition "
                    "to the other inputs required as customTemperature was "
                    "selected for temperature_spec config argument."
                )

        if self.config.condenser_type == CondenserType.totalCondenser:
            self.eq_total_cond_spec.deactivate()

        # Initialize the inlet and outlet state blocks
        self.control_volume.initialize(outlvl=outlvl)

        # Activate the total condenser spec
        if self.config.condenser_type == CondenserType.totalCondenser:
            self.eq_total_cond_spec.activate()

        if solver is not None:
            with solver_log(solve_log, idaeslog.DEBUG) as slc:
                res = solver.solve(self, tee=slc.tee)
            init_log.unit(
                "Initialisation Complete, {}.".format(idaeslog.condition(res))
            )

    def _get_performance_contents(self, time_point=0):
        var_dict = {}
        if hasattr(self, "heat_duty"):
            var_dict["Heat Duty"] = self.heat_duty[time_point]
        if hasattr(self, "deltaP"):
            var_dict["Pressure Change"] = self.deltaP[time_point]

        return {"vars": var_dict}

    def _get_stream_table_contents(self, time_point=0):
        stream_attributes = {}

        if self.config.condenser_type == CondenserType.totalCondenser:
            stream_dict = {"Inlet": "inlet",
                           "Reflux": "reflux",
                           "Distillate": "distillate"}
        else:
            stream_dict = {"Inlet": "inlet",
                           "Vapor Outlet": "vapor_outlet",
                           "Reflux": "reflux",
                           "Distillate": "distillate"}

        for n, v in stream_dict.items():
            port_obj = getattr(self, v)

            stream_attributes[n] = {}

            for k in port_obj.vars:
                for i in port_obj.vars[k].keys():
                    if isinstance(i, float):
                        stream_attributes[n][k] = value(
                            port_obj.vars[k][time_point])
                    else:
                        if len(i) == 2:
                            kname = str(i[1])
                        else:
                            kname = str(i[1:])
                        stream_attributes[n][k + " " + kname] = \
                            value(port_obj.vars[k][time_point, i[1:]])

        return DataFrame.from_dict(stream_attributes, orient="columns")
