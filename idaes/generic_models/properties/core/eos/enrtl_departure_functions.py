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
from pyomo.environ import exp, log, units as pyunits

from idaes.core.util.constants import Constants
from idaes.generic_models.properties.core.eos.enrtl import (
    DefaultAlphaRule, DefaultTauRule)
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


def dalpha_dT(b, pname, i, j):
    # Calculate dG/dT terms
    Y = getattr(b, pname+"_Y")
    molecular_set = b.params.solvent_set | b.params.solute_set
    pobj = b.params.get_phase(pname)

    # Check options for alpha rule
    if (pobj.config.equation_of_state_options is not None and
            "alpha_rule" in pobj.config.equation_of_state_options):
        alpha_dT = pobj.config.equation_of_state_options[
            "alpha_rule"].dT_expression
    else:
        alpha_dT = DefaultAlphaRule.dT_expression

    if ((i in molecular_set) and (j in molecular_set)):
        # alpha equal user provided parameters
        return alpha_dT(b, pobj, i, j, b.temperature)
    elif (i in b.params.cation_set and j in molecular_set):
        # FRom Eqn 32
        return sum(Y[k]*alpha_dT(b, pobj, (i+", "+k), j, b.temperature)
                   for k in b.params.anion_set)
    elif (j in b.params.cation_set and i in molecular_set):
        # From Eqn 32
        return sum(Y[k]*alpha_dT(b, pobj, (j+", "+k), i, b.temperature)
                   for k in b.params.anion_set)
    elif (i in b.params.anion_set and j in molecular_set):
        # From Eqn 33
        return sum(Y[k]*alpha_dT(b, pobj, (k+", "+i), j, b.temperature)
                   for k in b.params.cation_set)
    elif (j in b.params.anion_set and i in molecular_set):
        # From Eqn 33
        return sum(Y[k]*alpha_dT(b, pobj, (k+", "+j), i, b.temperature)
                   for k in b.params.cation_set)
    elif (i in b.params.cation_set and j in b.params.anion_set):
        # From Eqn 34
        if len(b.params.cation_set) > 1:
            return sum(Y[k]*alpha_dT(
                b, pobj, (i+", "+j), (k+", "+j), b.temperature)
                       for k in b.params.cation_set)
        else:
            return 0
    elif (i in b.params.anion_set and j in b.params.cation_set):
        # From Eqn 35
        if len(b.params.anion_set) > 1:
            return sum(Y[k]*alpha_dT(
                b, pobj, (j+", "+i), (j+", "+k), b.temperature)
                       for k in b.params.anion_set)
        else:
            return 0


def dG_dT(b, pname, i, j):
    # Calculate dG/dT terms
    Y = getattr(b, pname+"_Y")
    molecular_set = b.params.solvent_set | b.params.solute_set
    pobj = b.params.get_phase(pname)

    # Check options for alpha rule
    if (pobj.config.equation_of_state_options is not None and
            "alpha_rule" in pobj.config.equation_of_state_options):
        alpha = pobj.config.equation_of_state_options[
            "alpha_rule"].return_expression
        alpha_dT = pobj.config.equation_of_state_options[
            "alpha_rule"].dT_expression
    else:
        alpha = DefaultAlphaRule.return_expression
        alpha_dT = DefaultAlphaRule.dT_expression

    # Check options for tau rule
    if (pobj.config.equation_of_state_options is not None and
            "tau_rule" in pobj.config.equation_of_state_options):
        tau = pobj.config.equation_of_state_options[
            "tau_rule"].return_expression
        tau_dT = pobj.config.equation_of_state_options[
            "tau_rule"].dT_expression
    else:
        tau = DefaultTauRule.return_expression
        tau_dT = DefaultTauRule.dT_expression

    def _G(b, pobj, i, j, T):  # Eqn 23
        if i != j:
            return exp(-alpha(b, pobj, i, j, T) * tau(b, pobj, i, j, T))
        else:
            return 1

    def _G_dT(b, pobj, i, j, T):  # From Eqn 23
        if i != j:
            return (-_G(b, pobj, i, j, T) *
                    (alpha(b, pobj, i, j, T)*tau_dT(b, pobj, i, j, T) +
                     alpha_dT(b, pobj, i, j, T)*tau(b, pobj, i, j, T)))
        else:
            return 0

    if ((i in molecular_set) and
            (j in molecular_set)):
        # G comes directly from parameters
        return _G_dT(b, pobj, i, j, b.temperature)
    elif (i in b.params.cation_set and j in molecular_set):
        # From Eqn 38
        return sum(Y[k] * _G_dT(b, pobj, (i+", "+k), j,  b.temperature)
                   for k in b.params.anion_set)
    elif (i in molecular_set and j in b.params.cation_set):
        # From Eqn 40
        return sum(Y[k] * _G_dT(b, pobj, i, (j+", "+k), b.temperature)
                   for k in b.params.anion_set)
    elif (i in b.params.anion_set and j in molecular_set):
        # FRom Eqn 39
        return sum(Y[k] * _G_dT(b, pobj, (k+", "+i), j, b.temperature)
                   for k in b.params.cation_set)
    elif (i in molecular_set and j in b.params.anion_set):
        # From Eqn 41
        return sum(Y[k] * _G_dT(b, pobj, i, (k+", "+j), b.temperature)
                   for k in b.params.cation_set)
    elif (i in b.params.cation_set and j in b.params.anion_set):
        # From Eqn 42
        if len(b.params.cation_set) > 1:
            return sum(Y[k] * _G_dT(
                b, pobj, (i+", "+j), (k+", "+j), b.temperature)
                for k in b.params.cation_set)
        else:
            # This term does not exist for single cation systems
            # However, need a valid result to calculate tau
            return 0
    elif (i in b.params.anion_set and j in b.params.cation_set):
        # From Eqn 43
        if len(b.params.anion_set) > 1:
            return sum(Y[k] * _G_dT(
                b, pobj, (j+", "+i), (j+", "+k), b.temperature)
                for k in b.params.anion_set)
        else:
            # This term does not exist for single anion systems
            # However, need a valid result to calculate tau
            return 0


def dtau_dT(b, pname, i, j):
    # Calculate tau terms
    molecular_set = b.params.solvent_set | b.params.solute_set
    pobj = b.params.get_phase(pname)

    # Check options for tau rule
    if (pobj.config.equation_of_state_options is not None and
            "tau_rule" in pobj.config.equation_of_state_options):
        tau_dT = pobj.config.equation_of_state_options[
            "tau_rule"].dT_expression
    else:
        tau_dT = DefaultTauRule.dT_expression

    if ((i in molecular_set) and (j in molecular_set)):
        # tau equal to parameter
        return tau_dT(b, pobj, i, j, b.temperature)
    else:
        alpha = getattr(b, pname+"_alpha")
        G = getattr(b, pname+"_G")
        # From Eqn 44
        return ((dalpha_dT(b, pname, i, j)*log(G[i, j]) -
                 alpha[i, j]*dG_dT(b, pname, i, j)/G[i, j]) /
                alpha[i, j]**2)
