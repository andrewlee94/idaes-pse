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
Tests for Metrics object
"""
import pytest
import pandas as pd

from idaes.surrogate.metrics import TrainingMetrics
from idaes.surrogate import AlamoObject


# For this test we will use a simple z = x function and calcuate metrics
# Measured (test) data will include a fixed offset (i.e. z = x + Err)
# Data will be collected symetrically around 0, so that mean z1 == 0
ERR = 0.1  # offset to apply to check metrics
Np = 50  # number of points above 0 to use in check
N = 2*Np+1  # total number of points to use in check


@pytest.fixture
def metrics():
    # Create dataset
    x = []
    z = []
    for i in range(N):
        v = 0.1*i - Np/10
        x.append(v)
        z.append(v+ERR)  # add offset of 0.1
    dataset = pd.DataFrame({"x1": x, "z1": z})

    # Create a dummy ALAMO surrogate to use for testing
    alm_obj = AlamoObject(surrogate_expressions={"z1": "z1 == x1"},
                          input_labels=["x1"],
                          output_labels=["z1"])

    metrics = TrainingMetrics(surrogate=alm_obj, dataframe=dataset)

    return metrics


@pytest.mark.unit
def test_init(metrics):
    assert isinstance(metrics._surrogate, AlamoObject)
    assert metrics._surrogate._surrogate_expressions["z1"] == "z1 == x1"

    assert isinstance(metrics._measured_data, pd.DataFrame)

    assert metrics._evaluated_data is None
    assert metrics._RMSE is None
    assert metrics._MSE is None
    assert metrics._SSE is None
    assert metrics._R2 is None


@pytest.mark.unit
def test_evalaute_surrogate(metrics):
    metrics.evaluate_surrogate()

    assert isinstance(metrics._evaluated_data, pd.DataFrame)

    # Check that z = x in all rows
    for i, r in metrics._evaluated_data.iterrows():
        assert metrics._evaluated_data["z1"][i] == \
            metrics._measured_data["x1"][i]


@pytest.mark.unit
def test_compute_metrics(metrics):
    metrics.compute_metrics()

    # All data should have an offset of ERR, so error is known
    assert metrics.SSE["z1"] == pytest.approx(ERR**2*N, rel=1e-12)
    assert metrics.MSE["z1"] == pytest.approx(ERR**2, rel=1e-12)
    assert metrics.RMSE["z1"] == pytest.approx(ERR, rel=1e-12)

    # Calculate SST
    # Average measured z1 is equal to ERR, thus measured z1-z1_mean = x1
    sst = 0
    for i in range(int(Np)):
        sst += 2*(Np/10-0.1*i)**2

    assert metrics._SST["z1"] == pytest.approx(sst, rel=1e-12)
    assert metrics.R2["z1"] == pytest.approx(1-ERR**2*N/sst, rel=1e-12)
