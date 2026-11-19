"""Guard the new timed LSTM preparation against feature drift."""
from __future__ import annotations

import numpy as np
import pytest

from approaches.lstm.causal_data import prepare_features, transform_test_for_benchmark
from shared.contract import FILES


def test_benchmark_test_features_match_the_export_contract() -> None:
    if not all(path.is_file() for path in FILES.values()):
        pytest.skip("The frozen physical snapshots are local artifacts")
    model1, model2, physical, metadata, state = prepare_features(
        stuck_context=True, return_benchmark_state=True)
    prepared1, prepared2 = transform_test_for_benchmark(physical["test"], state)
    assert prepared1.shape == model1["test"].shape == (31680, 27)
    assert prepared2.shape == model2["test"].shape == (31680, 154)
    assert metadata["data_manifest_id"] == state["data_manifest_id"]
    assert np.array_equal(prepared1, model1["test"])
    assert float(np.max(np.abs(prepared2 - model2["test"]))) < 1e-3
