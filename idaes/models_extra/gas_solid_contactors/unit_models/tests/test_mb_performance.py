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
"""

import pytest

import pyomo.environ as pyo
from pyomo.dae.flatten import flatten_dae_components
from pyomo.environ import (
    ConcreteModel,
    check_optimal_termination,
    SolverStatus,
    value,
    Var,
    Constraint,
)
from pyomo.util.check_units import assert_units_consistent
from pyomo.common.config import ConfigBlock
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from idaes.core import (
    FlowsheetBlock,
    MaterialBalanceType,
    EnergyBalanceType,
    MomentumBalanceType,
)
from idaes.core.util.model_statistics import (
    degrees_of_freedom,
    number_variables,
    number_total_constraints,
    number_unused_variables,
    unused_variables_set,
    large_residuals_set,
)
from idaes.core.util.testing import initialization_tester
from idaes.core.util import scaling as iscale
from idaes.core.solvers import get_solver
from idaes.core.util.exceptions import InitializationError

# Import MBR unit model
from idaes.models_extra.gas_solid_contactors.unit_models.moving_bed import MBR

# Import property packages
from idaes.models_extra.gas_solid_contactors.properties.methane_iron_OC_reduction.gas_phase_thermo import (
    GasPhaseParameterBlock,
)
from idaes.models_extra.gas_solid_contactors.properties.methane_iron_OC_reduction.solid_phase_thermo import (
    SolidPhaseParameterBlock,
)
from idaes.models_extra.gas_solid_contactors.properties.methane_iron_OC_reduction.hetero_reactions import (
    HeteroReactionParameterBlock,
)


def build_model(
    dynamic=False,
    #ntfe=10,
    #ntcp=3,
):
    m = ConcreteModel()
    if dynamic:
        m.fs = FlowsheetBlock(
            default={
                "dynamic": dynamic,
                "time_set": [0, 600],
                "time_units": pyo.units.s,
            },
        )
    else:
        m.fs = FlowsheetBlock(default={"dynamic": dynamic})

    # Set up thermo props and reaction props
    m.fs.gas_properties = GasPhaseParameterBlock()
    m.fs.solid_properties = SolidPhaseParameterBlock()

    m.fs.hetero_reactions = HeteroReactionParameterBlock(
        default={
            "solid_property_package": m.fs.solid_properties,
            "gas_property_package": m.fs.gas_properties,
        }
    )

    m.fs.MB = MBR(
        default={
            "has_holdup": True,
            "transformation_method": "dae.collocation",
            "gas_phase_config": {"property_package": m.fs.gas_properties},
            "solid_phase_config": {
                "property_package": m.fs.solid_properties,
                "reaction_package": m.fs.hetero_reactions,
            },
        }
    )
    return m


def initialize_model(m, dynamic=False, ntfe=10, ntcp=3):
    if dynamic:
        disc = pyo.TransformationFactory("dae.collocation")
        disc.apply_to(
            m, wrt=m.fs.time, nfe=ntfe, ncp=ntcp, scheme="LAGRANGE-RADAU"
        )

    # Fix bed geometry variables
    m.fs.MB.bed_diameter.fix(6.5)  # m
    m.fs.MB.bed_height.fix(5)  # m

    # Fix inlet port variables for gas and solid
    m.fs.MB.gas_inlet.flow_mol[:].fix(128.20513)  # mol/s
    m.fs.MB.gas_inlet.temperature[:].fix(298.15)  # K
    m.fs.MB.gas_inlet.pressure[:].fix(2.00e5)  # Pa = 1E5 bar
    m.fs.MB.gas_inlet.mole_frac_comp[:, "CO2"].fix(0.02499)
    m.fs.MB.gas_inlet.mole_frac_comp[:, "H2O"].fix(0.00001)
    m.fs.MB.gas_inlet.mole_frac_comp[:, "CH4"].fix(0.975)

    m.fs.MB.solid_inlet.flow_mass[:].fix(591.4)  # kg/s
    # Particle porosity:
    # The porosity of the OC particle at the inlet is calculated from the
    # known bulk density of the fresh OC particle (3251.75 kg/m3), and the
    # skeletal density of the fresh OC particle (calculated from the known
    # composition of the fresh particle, and the skeletal density of its
    # components [see the solids property package])
    m.fs.MB.solid_inlet.particle_porosity[:].fix(0.27)
    m.fs.MB.solid_inlet.temperature[:].fix(1183.15)  # K
    m.fs.MB.solid_inlet.mass_frac_comp[:, "Fe2O3"].fix(0.45)
    m.fs.MB.solid_inlet.mass_frac_comp[:, "Fe3O4"].fix(1e-9)
    m.fs.MB.solid_inlet.mass_frac_comp[:, "Al2O3"].fix(0.55)

    # Fix initial conditions
    if dynamic:
        t0 = m.fs.time.first()
        x0 = m.fs.MB.gas_phase.length_domain.first()
        xf = m.fs.MB.gas_phase.length_domain.last()

        m.fs.MB.gas_phase.material_holdup[t0, ...].fix()
        m.fs.MB.gas_phase.energy_holdup[t0, ...].fix()
        m.fs.MB.solid_phase.material_holdup[t0, ...].fix()
        m.fs.MB.solid_phase.energy_holdup[t0, ...].fix()

        m.fs.MB.gas_phase.material_holdup[t0, x0, ...].unfix()
        m.fs.MB.gas_phase.energy_holdup[t0, x0, ...].unfix()
        m.fs.MB.solid_phase.material_holdup[t0, xf, ...].unfix()
        m.fs.MB.solid_phase.energy_holdup[t0, xf, ...].unfix()

    assert degrees_of_freedom(m) == 0

    # Initialize fuel reactor

    # State arguments for initializing property state blocks
    # Gas phase temperature is initialized at solid
    # temperature because thermal mass of solid >> thermal mass of gas
    # Particularly useful for initialization if reaction takes place
    blk = m.fs.MB
    gas_phase_state_args = {
        "flow_mol": blk.gas_inlet.flow_mol[0].value,
        "temperature": blk.solid_inlet.temperature[0].value,
        "pressure": blk.gas_inlet.pressure[0].value,
        "mole_frac": {
            "CH4": blk.gas_inlet.mole_frac_comp[0, "CH4"].value,
            "CO2": blk.gas_inlet.mole_frac_comp[0, "CO2"].value,
            "H2O": blk.gas_inlet.mole_frac_comp[0, "H2O"].value,
        },
    }
    solid_phase_state_args = {
        "flow_mass": blk.solid_inlet.flow_mass[0].value,
        "particle_porosity": blk.solid_inlet.particle_porosity[0].value,
        "temperature": blk.solid_inlet.temperature[0].value,
        "mass_frac": {
            "Fe2O3": blk.solid_inlet.mass_frac_comp[0, "Fe2O3"].value,
            "Fe3O4": blk.solid_inlet.mass_frac_comp[0, "Fe3O4"].value,
            "Al2O3": blk.solid_inlet.mass_frac_comp[0, "Al2O3"].value,
        },
    }

    # Scale the model by applying scaling transformation
    # This reduces ill conditioning of the model
    iscale.calculate_scaling_factors(m)

    if dynamic:
        length = m.fs.MB.gas_phase.length_domain
        nfe = length.get_discretization_info()["nfe"]
        ncp = length.get_discretization_info().get("ncp", 1)
        steady = build_model(dynamic=False)
        initialize_model(steady)
        scalar_vars, dae_vars = flatten_dae_components(
            steady, steady.fs.time, pyo.Var
        )
        scalar_data = [
            (pyo.ComponentUID(var), var.value) for var in scalar_vars
        ]
        dae_data = [
            (pyo.ComponentUID(var.referent), var[t0].value)
            for var in dae_vars
        ]
        for cuid, val in scalar_data:
            m.find_component(cuid).set_value(val)
        for cuid, val in dae_data:
            m.find_component(cuid)[:].set_value(val)

        m.fs.MB.gas_phase.material_accumulation[...].set_value(0.0)
        m.fs.MB.gas_phase.energy_accumulation[...].set_value(0.0)
        m.fs.MB.solid_phase.material_accumulation[...].set_value(0.0)
        m.fs.MB.solid_phase.energy_accumulation[...].set_value(0.0)

    else:
        m.fs.MB.initialize(
            optarg={"tol": 1e-5},
            gas_phase_state_args=gas_phase_state_args,
            solid_phase_state_args=solid_phase_state_args,
        )


def solve_model(m):
    solver = get_solver()
    results = solver.solve(m.fs.MB, tee=True, options={"tol": 1e-5})
    pyo.assert_optimal_termination(results)


def build_dynamic_model():
    return build_model(dynamic=True)


def initialize_dynamic_model(model):
    initialize_model(model, dynamic=True, ntfe=10, ntcp=1)


if __name__ == "__main__":
    dynamic = True
    kwds = {"ntfe": 10, "ntcp": 1}
    model = build_model(dynamic=dynamic)
    initialize_model(model, dynamic=dynamic, **kwds)
    solve_model(model)
