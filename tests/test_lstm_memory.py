"""The Jetson training path must never materialize a full window tensor."""

from types import SimpleNamespace

import numpy as np

from approaches.lstm import adapter
from approaches.lstm.causal_data import windows


class _Dataset:
    def __init__(self, factory):
        self.factory = factory

    def prefetch(self, count):
        assert count == 1
        return self

    def __iter__(self):
        return iter(self.factory())


def test_training_batches_keep_targets_labels_and_weights_aligned(monkeypatch):
    matrices = {"train": np.arange(30, dtype=np.float32)[:, None]}
    targets = np.arange(19, 30, dtype=np.int64)
    labels = (targets % 2).astype(np.int32)
    monkeypatch.setattr(adapter, "BATCH_SIZE", 3)
    original = adapter.windows

    def bounded_windows(*args, **kwargs):
        assert len(kwargs["targets"]) <= 3
        return original(*args, **kwargs)

    monkeypatch.setattr(adapter, "windows", bounded_windows)
    fake_tf = SimpleNamespace(
        float32=np.float32, int32=np.int32,
        TensorSpec=lambda shape, dtype: (shape, dtype),
        data=SimpleNamespace(Dataset=SimpleNamespace(
            from_generator=lambda factory, output_signature: _Dataset(factory))),
    )
    dataset = adapter._training_dataset(
        fake_tf, matrices, "train", targets, labels, class_weights={0: 0.5, 1: 2.0}
    )
    batches = list(dataset)
    assert len(batches) == 4
    seen = []
    for x, y, weights in batches:
        assert x.shape[0] <= 3 and x.shape[1:] == (20, 1)
        np.testing.assert_array_equal(y, x[:, -1, 0].astype(np.int32) % 2)
        np.testing.assert_array_equal(weights, np.where(y == 0, 0.5, 2.0))
        seen.extend(x[:, -1, 0].astype(int))
    assert sorted(seen) == targets.tolist()


def test_chunked_predictions_match_full_windows_at_split_boundary():
    matrices = {"train": np.arange(25, dtype=np.float32)[:, None],
                "validation": np.arange(25, 30, dtype=np.float32)[:, None],
                "test": np.arange(30, 35, dtype=np.float32)[:, None]}
    targets = np.arange(5)

    class Model:
        output_shape = (None, 1)

        def predict_on_batch(self, x):
            return x.mean(axis=1)

    expected = windows(matrices, "test", targets=targets)[0].mean(axis=1)
    actual = adapter._predict_rows(Model(), matrices, "test", targets, batch_size=2)
    np.testing.assert_array_equal(actual, expected)
