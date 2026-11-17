"""Prevent leakage and verify consistent scaler shape for held-out samples."""
import unittest

import numpy as np


def normalize(train, heldout):
    train = np.asarray(train, dtype=float)
    heldout = np.asarray(heldout, dtype=float)
    if train.ndim != 2 or heldout.ndim != 2 or train.shape[1] != heldout.shape[1]:
        raise ValueError("Expected two feature matrices with matching columns")
    if train.shape[0] == 0:
        raise ValueError("Training samples are required")
    means = train.mean(axis=0)
    spread = train.std(axis=0)
    spread = np.where(spread == 0, 1, spread)
    return (train - means) / spread, (heldout - means) / spread


class NormalizationStabilityTests(unittest.TestCase):
    def test_heldout_extreme_does_not_change_training_scaler(self):
        train = [[1, 3], [3, 3], [5, 3]]
        normalized, heldout = normalize(train, [[1000, 3]])
        np.testing.assert_allclose(normalized.mean(axis=0), [0, 0], atol=1e-12)
        self.assertGreater(heldout[0, 0], 100)
        self.assertEqual(heldout[0, 1], 0)
        np.testing.assert_allclose(train, [[1, 3], [3, 3], [5, 3]])

    def test_mismatched_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            normalize([[1, 2]], [[1, 2, 3]])

    def test_empty_training_set_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize(np.empty((0, 2)), [[1, 2]])


if __name__ == "__main__":
    unittest.main()
