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

__author__ = "John Eslick"

"""Transformation to replace variables with other variables."""
import pyomo.environ as pyo
from pyomo.core.base.plugin import TransformationFactory
from pyomo.core.plugins.transform.hierarchy import NonIsomorphicTransformation
from pyomo.core.expr import current as EXPR
from pyomo.common.config import ConfigBlock, ConfigValue, add_docstring_list
from pyomo.core.base.var import _GeneralVarData


def _is_var(v):
    return isinstance(v, (_GeneralVarData, pyo.Var))

@TransformationFactory.register(
    'replace_variables',
    doc="Replace variables with other variables.")
class ReplaceVariables(NonIsomorphicTransformation):
    """Replace variables in a model or block with other variables.

    Keyword arguments below are specified for the ``apply_to(instance, **kwargs)``
    method.

    """
    CONFIG = ConfigBlock()
    CONFIG.declare("substitute", ConfigValue(
        default=[],
        description="List-like of tuples where the first item in a tuple is a "
                    "Pyomo variable to be replaced and the second item in the "
                    "tuple is a Pyomo variable to replace it with. This "
                    "transformation is not reversible."
    ))

    __doc__ = add_docstring_list(__doc__, CONFIG)

    @staticmethod
    def replace(instance, substitute):
        # Create the replacement dict. Do some argument validation and indexed
        # var handling
        d = {}
        for r in substitute:
            if not (_is_var(r[0]) and _is_var(r[1])):
                raise TypeError(
                    "Replace only allows variables to be replaced, {} is type {}"
                    " and {} is type {}".format(r[0], type(r[0]), r[1], type(r[1]))
                )
            if r[0].is_indexed() != r[1].is_indexed():
                raise TypeError(
                    "IndexedVars must be replaced by IndexedVars, {} is type {}"
                    " and {} is type {}".format(r[0], type(r[0]), r[1], type(r[1]))
                )
            if r[0].is_indexed() and r[1].is_indexed():
                if not r[0].index_set().issubset(r[1].index_set()):
                    raise ValueError(
                        "The index set of {} must be a subset of"
                        " {}.".format(r[0], r[1])
                    )
                for i in r[0]:
                    d[id(r[0][i])] = r[1][i]
            else:
                #scalar replace
                d[id(r[0])] = r[1]

        # Replacement Visitor
        vis = EXPR.ExpressionReplacementVisitor(substitute=d)

        # Do replacements in Expressions, Constraints, and Objectives
        for c in instance.component_data_objects(pyo.Constraint, descend_into=True):
            c.set_value(expr=vis.dfs_postorder_stack(c.expr))
        for c in instance.component_data_objects((pyo.Expression, pyo.Objective),
                                                 descend_into=True):
            vis.dfs_postorder_stack(c)


    def _apply_to(self, instance, **kwds):
        """
        Apply the transformation.  This is called by ``apply_to`` in the
        superclass, and should not be called directly.  ``apply_to`` takes the
        same arguments.

        Args:
            instance: A block or model to apply the transformation to
            substitute: A list-like of two-element list-likes.  Each two element
                list-like specifies a replacment of the first variable by the
                second.  SimpleVar, IndexedVar, _GeneralVarData, and Reference are
                all accepted types.

        Returns:
            None
        """
        config = self.CONFIG(kwds)
        self.replace(instance, config.substitute)
