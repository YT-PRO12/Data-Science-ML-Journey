"""Executable NumPy fundamentals for a learning notebook chapter."""
import unittest
import numpy as np


class NumPyShapeContracts(unittest.TestCase):
    def test_axis_reductions_have_different_shapes(self):
        matrix = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(matrix.sum(axis=0).tolist(), [5, 7, 9])
        self.assertEqual(matrix.sum(axis=1).tolist(), [6, 15])

    def test_column_broadcasting_does_not_mutate_input(self):
        matrix = np.array([[1, 2], [3, 4]])
        result = matrix + np.array([[10], [20]])
        self.assertEqual(result.tolist(), [[11, 12], [23, 24]])
        self.assertEqual(matrix.tolist(), [[1, 2], [3, 4]])

    def test_explicit_copy_is_independent(self):
        source = np.array([1, 2, 3])
        copy = source[:2].copy()
        copy[0] = 99
        self.assertEqual(source.tolist(), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
