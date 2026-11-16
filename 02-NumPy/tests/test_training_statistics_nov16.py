"""Learning exercise: fit normalization statistics only on training data."""
import unittest

import numpy as np


def standardize(train, validation):
    train = np.asarray(train, dtype=float)
    validation = np.asarray(validation, dtype=float)
    if train.ndim != 2 or validation.ndim != 2 or train.shape[1] != validation.shape[1]:
        raise ValueError("Expected two matrices with matching feature counts")
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    return (train - mean) / scale, (validation - mean) / scale


class TrainingStatisticsContracts(unittest.TestCase):
    def test_validation_outlier_does_not_change_training_scaler(self):
        train = [[1.0, 7.0], [3.0, 7.0], [5.0, 7.0]]
        normalized_train, transformed = standardize(train, [[1000.0, 7.0]])
        np.testing.assert_allclose(normalized_train.mean(axis=0), [0, 0], atol=1e-12)
        np.testing.assert_allclose(normalized_train[:, 1], [0, 0, 0])
        self.assertGreater(transformed[0, 0], 100)
        self.assertEqual(transformed[0, 1], 0.0)

    def test_mismatched_feature_counts_fail(self):
        with self.assertRaises(ValueError):
            standardize([[1, 2]], [[1, 2, 3]])


if __name__ == "__main__":
    unittest.main()
