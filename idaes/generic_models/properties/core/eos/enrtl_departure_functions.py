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
DEparture function sub-methods for eNRTL activity coefficient method.

Only applicable to liquid/electrolyte phases

Mostly needed to calculate d/dT for excess enthalpy

Reference:

Song, Y. and Chen, C.-C., Symmetric Electrolyte Nonrandom Two-Liquid Activity
Coefficient Model, Ind. Eng. Chem. Res., 2009, Vol. 48, pgs. 7788–7797
"""
from pyomo.environ import units as pyunits
from idaes.core.util.constants import Constants
from idaes.generic_models.properties.core.generic.utility import (
    get_component_object as cobj)


def dv_dT(b, pname):
    # From Eqn 77
    if len(b.params.solvent_set) == 1:
        s = b.params.solvent_set.first()
        dens_mol_cls = b.params.get_component(s).config["dens_mol_liq_comp"]
        r = dens_mol_cls.return_expression(b, cobj(b, s), b.temperature)
        dr_dT = dens_mol_cls.dT_expression(b, cobj(b, s), b.temperature)
        dv_dT = dr_dT/r**2
        return pyunits.convert(dv_dT, pyunits.m**3/pyunits.mol/pyunits.K)
    else:
        n = 0
        d = 0
        for s in b.params.solvent_set:
            dens_mol_cls = b.params.get_component(s).config[
                "dens_mol_liq_comp"]
            r = dens_mol_cls.return_expression(b, cobj(b, s), b.temperature)
            dr_dT = dens_mol_cls.dT_expression(b, cobj(b, s), b.temperature)
            dv_dT = dr_dT/r**2

            n += b.mole_frac_phase_comp_true[pname, s]*dv_dT
            d += b.mole_frac_phase_comp_true[pname, s]
        return pyunits.convert(n/d, pyunits.m**3/pyunits.mol/pyunits.K)


def deps_dT(b, pname):
    # From Eqn 78
    if len(b.params.solvent_set) == 1:
        s = b.params.solvent_set.first()
        eps_cls = b.params.get_component(s).config[
            "relative_permittivity_liq_comp"]
        return eps_cls.dT_expression(b, cobj(b, s), b.temperature)
    else:
        n = 0
        d = 0
        for s in b.params.solvent_set:
            eps_cls = b.params.get_component(s).config[
                "relative_permittivity_liq_comp"]
            n += (b.mole_frac_phase_comp_true[pname, s] *
                  eps_cls.dT_expression(b, cobj(b, s), b.temperature) *
                  b.params.get_component(s).mw)
            d += (b.mole_frac_phase_comp_true[pname, s] *
                  b.params.get_component(s).mw)
        return n/d


def dA_dt(b, pname):
    # Partial derivative of Debye-Huckel parameter with temperature
    v = pyunits.convert(getattr(b, pname+"_vol_mol_solvent"),
                        pyunits.m**3/pyunits.mol)
    eps = getattr(b, pname+"_relative_permittivity_solvent")
    eps0 = Constants.vacuum_electric_permittivity

    x = (2*Constants.pi*Constants.avogadro_number/v)**(1/2)
    dx_dT = -((Constants.pi*Constants.avogadro_number)**(1/2) /
              (2**(1/2)*v**(3/2)) *
              dv_dT(b, pname))

    y = (Constants.elemental_charge**2 /
         (4*Constants.pi*eps*eps0 *
          Constants.boltzmann_constant*b.temperature))**(3/2)
    dy_dT = (-3/(16*Constants.pi**(3/2)) *
             eps0*Constants.boltzmann_constant*Constants.elemental_charge**3 *
             (deps_dT(b, pname)*b.temperature + eps) *
             (1/(eps*eps0*Constants.boltzmann_constant*b.temperature))**(5/2))

    return (1/3)*(x*dy_dT + dx_dT*y)
