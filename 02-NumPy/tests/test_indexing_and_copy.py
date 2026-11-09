"""Regression checks for NumPy indexing, views, and copies."""
import unittest

import numpy as np


class IndexingAndCopyTests(unittest.TestCase):
    def test_slice_is_view_and_changes_base_array(self):
        values = np.array([1, 2, 3, 4])
        view = values[1:3]
        view[0] = 99
        self.assertEqual(values.tolist(), [1, 99, 3, 4])

    def test_boolean_indexing_creates_independent_result(self):
        values = np.array([1, 2, 3, 4])
        selected = values[values % 2 == 0]
        selected[0] = 99
        self.assertEqual(values.tolist(), [1, 2, 3, 4])
        self.assertEqual(selected.tolist(), [99, 4])

    def test_explicit_copy_is_independent(self):
        values = np.array([[1, 2], [3, 4]])
        copy = values[:, 0].copy()
        copy[0] = 10
        self.assertEqual(values[0, 0], 1)


if __name__ == "__main__":
    unittest.main()
