# #################################################################################
# # The Institute for the Design of Advanced Energy Systems Integrated Platform
# # Framework (IDAES IP) was produced under the DOE Institute for the
# # Design of Advanced Energy Systems (IDAES).
# #
# # Copyright (c) 2018-2026 by the software owners: The Regents of the
# # University of California, through Lawrence Berkeley National Laboratory,
# # National Technology & Engineering Solutions of Sandia, LLC, Carnegie Mellon
# # University, West Virginia University Research Corporation, et al.
# # All rights reserved.  Please see the files COPYRIGHT.md and LICENSE.md
# # for full copyright and license information.
# #################################################################################
# """
# This module contains tests for the Diagnostics Toolbox using ExternalGreyBox models.
# """
# from io import StringIO
# import re

# import pytest

# import numpy as np
# import pytest
# from scipy.sparse import coo_matrix

# from pyomo.contrib.pynumero.interfaces.external_grey_box import (
#     ExternalGreyBoxBlock,
#     ExternalGreyBoxModel,
# )
# from pyomo.environ import (
#     Block,
#     ConcreteModel,
#     Constraint,
#     SolverFactory,
#     units,
#     Var,
# )

# from idaes.core.util.diagnostics_tools.diagnostics_toolbox import (
#     DiagnosticsToolbox,
# )


# @pytest.fixture()
# def model():
#     # Note: This model is deliberately infeasible and poorly scaled to
#     # test the diagnostics toolbox capabilities.
#     class BasicGrayBox(ExternalGreyBoxModel):
#         def input_names(self):
#             return ["v2", "v5", "v6"]

#         def output_names(self):
#             return ["v1", "v3", "v4", "v7"]

#         def set_input_values(self, input_values):
#             self._input_values = list(input_values)

#         def finalize_block_construction(self, pyomo_block):
#             pyomo_block.outputs["v3"].setlb(0)  # v3
#             pyomo_block.outputs["v3"].setub(5)  # v3
#             pyomo_block.inputs["v5"].setlb(0)  # v5
#             pyomo_block.inputs["v5"].setub(5)  # v5

#         def evaluate_outputs(self):
#             v1 = 10 - self._input_values[0]  # c1
#             # Linear combination of c2 and c3 to eliminate v3 and solve for v4
#             v4 = -2*self._input_values[1] - 2*self._input_values[1]
#             v3 = v4 + self._input_values[1]  # c2
#             v7 = 2e-8 * v1  # c4
#             return [v1, v3, v4, v7]

#         def evaluate_jacobian_outputs(self):
#             row = np.zeros(6)  # output index
#             col = np.zeros(6)  # input index
#             data = np.zeros(6)  # jacobian values

#             # dv1/dv2
#             row[0], col[0], data[0] = (0, 0, -1)
#             # dv4/dv5
#             row[1], col[1], data[1] = (2, 1, -2)
#             # dv4/dv6
#             row[2], col[2], data[2] = (2, 2, -1)
#             # dv3/dv5
#             row[3], col[3], data[3] = (1, 1, -1)
#             # dv3/dv6
#             row[4], col[4], data[4] = (1, 2, -1)
#             # dv7/dv2
#             row[5], col[5], data[5] = (3, 0, -2e-8)

#             return coo_matrix((data, (row, col)), shape=(4, 3))

#     m = ConcreteModel()
#     m.b = Block()

#     m.v1 = Var(units=units.m)  # External variable
#     m.b.v2 = Var(units=units.m)
#     m.b.v3 = Var(bounds=(0, 5))
#     m.b.v4 = Var()
#     m.b.v5 = Var(bounds=(0, 5))
#     m.b.v6 = Var()
#     m.b.v7 = Var(
#         units=units.m, bounds=(0, 1)
#     )  # Poorly scaled variable with lower bound
#     m.b.v8 = Var()  # Unused variable

#     m.b.v2.fix(5)
#     m.b.v5.fix(2)
#     m.b.v6.fix(0)

#     m.b.gb = ExternalGreyBoxBlock(external_model=BasicGrayBox())
#     m.b.gb_link_v1 = Constraint(expr=m.b.gb.outputs["v1"] == m.v1)
#     m.b.gb_link_v2 = Constraint(expr=m.b.gb.inputs["v2"] == m.b.v2)
#     m.b.gb_link_v3 = Constraint(expr=m.b.gb.outputs["v3"] == m.b.v3)
#     m.b.gb_link_v4 = Constraint(expr=m.b.gb.outputs["v4"] == m.b.v4)
#     m.b.gb_link_v5 = Constraint(expr=m.b.gb.inputs["v5"] == m.b.v5)
#     m.b.gb_link_v6 = Constraint(expr=m.b.gb.inputs["v6"] == m.b.v6)
#     m.b.gb_link_v7 = Constraint(expr=m.b.gb.outputs["v7"] == m.b.v7)

#     # Solve model
#     solver = SolverFactory("cyipopt")
#     res = solver.solve(m)

#     return m


# @pytest.fixture()
# def diagnostics_toolbox(model):
#     return DiagnosticsToolbox(model.b)


# @pytest.mark.unit
# def test_display_external_variables(diagnostics_toolbox):
#     stream = StringIO()

#     diagnostics_toolbox.display_external_variables(stream=stream)

#     expected = """====================================================================================
# The following external variable(s) appear in constraints within the model:

#     v1

# ====================================================================================
# """
#     assert stream.getvalue() == expected


# @pytest.mark.unit
# def test_display_unused_variables(diagnostics_toolbox):
#     stream = StringIO()

#     diagnostics_toolbox.display_unused_variables(stream=stream)

#     expected = """====================================================================================
# The following variable(s) do not appear in any activated constraints within the model:

#     b.v8

# ====================================================================================
# """

#     assert stream.getvalue() == expected


# @pytest.mark.unit
# def test_display_variables_fixed_to_zero(diagnostics_toolbox):
#     stream = StringIO()

#     diagnostics_toolbox.display_variables_fixed_to_zero(stream=stream)

#     expected = """====================================================================================
# The following variable(s) are fixed to zero:

#     b.v6

# ====================================================================================
# """

#     assert stream.getvalue() == expected


# @pytest.mark.unit
# def test_display_variables_at_or_outside_bounds(diagnostics_toolbox):
#     stream = StringIO()

#     diagnostics_toolbox.display_variables_at_or_outside_bounds(stream=stream)

#     expected = """====================================================================================
# The following variable(s) have values at or outside their bounds (tol=0.0E+00):

#     b.v3 (free): value=-9.997671315624657e-09 bounds=(0, 5)

# ====================================================================================
# """

#     assert stream.getvalue() == expected


# @pytest.mark.unit
# def test_display_variables_with_none_value(diagnostics_toolbox):
#     stream = StringIO()

#     diagnostics_toolbox.display_variables_with_none_value(stream=stream)

#     expected = """====================================================================================
# The following variable(s) have a value of None:

#     b.v8

# ====================================================================================
# """

#     assert stream.getvalue() == expected


# @pytest.mark.unit
# def test_display_variables_with_none_value_in_activated_constraints(diagnostics_toolbox):
#     stream = StringIO()

#     # Set some variable values to None for testing
#     diagnostics_toolbox._model.v4.set_value(None)
#     diagnostics_toolbox._model.gb.inputs["v5"].set_value(None)

#     diagnostics_toolbox.display_variables_with_none_value_in_activated_constraints(stream=stream)

#     expected = """====================================================================================
# The following variable(s) have a value of None:

#     b.v4
#     b.gb.inputs[v5]

# ====================================================================================
# """

#     assert stream.getvalue() == expected


# @pytest.mark.unit
# def test_display_variables_with_value_near_zero(diagnostics_toolbox):
#     stream = StringIO()

#     diagnostics_toolbox.display_variables_with_value_near_zero(stream=stream)

#     expected = """====================================================================================
# The following variable(s) have a value close to zero (tol=1.0E-08):

#     b.v3: value=-9.997671315624657e-09
#     b.v6: value=0

# ====================================================================================
# """

#     assert stream.getvalue() == expected


# @pytest.mark.unit
# def test_display_variables_with_extreme_values(diagnostics_toolbox):
#     stream = StringIO()

#     diagnostics_toolbox.display_variables_with_extreme_values(stream=stream)

#     expected = """====================================================================================
# The following variable(s) have extreme values (<1.0E-04 or > 1.0E+04):

#     b.v7: 1.0000939326524314e-07

# ====================================================================================
# """

#     assert stream.getvalue() == expected
