"""NumPy axis behavior: retained dimensions make broadcasting unambiguous."""
import unittest

import numpy as np


class KeepDimensionsTests(unittest.TestCase):
    def test_row_centers_broadcast_to_original_matrix(self):
        values = np.array([[2.0, 4.0, 6.0], [10.0, 20.0, 30.0]])
        row_means = values.mean(axis=1, keepdims=True)
        self.assertEqual(row_means.shape, (2, 1))
        np.testing.assert_array_equal(
            values - row_means,
            [[-2.0, 0.0, 2.0], [-10.0, 0.0, 10.0]],
        )

    def test_column_sums_preserve_single_row(self):
        values = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(values.sum(axis=0, keepdims=True).shape, (1, 3))
        np.testing.assert_array_equal(
            values.sum(axis=0, keepdims=True), [[5, 7, 9]]
        )

    def test_original_data_is_unchanged(self):
        values = np.array([[1.0, 3.0], [2.0, 4.0]])
        _ = values - values.mean(axis=0, keepdims=True)
        np.testing.assert_array_equal(values, [[1, 3], [2, 4]])


if __name__ == "__main__":
    unittest.main()
