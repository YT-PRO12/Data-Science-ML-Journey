"""Independent numerical tests; no dataset downloads or notebook execution."""
import unittest

import numpy as np


class BroadcastingContracts(unittest.TestCase):
    def test_row_offsets_and_column_offsets_follow_shape_rules(self):
        matrix = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(
            (matrix + np.array([10, 20, 30])).tolist(),
            [[11, 22, 33], [14, 25, 36]],
        )
        self.assertEqual(
            (matrix + np.array([[10], [20]])).tolist(),
            [[11, 12, 13], [24, 25, 26]],
        )

    def test_incompatible_trailing_dimensions_raise(self):
        matrix = np.ones((2, 3))
        with self.assertRaises(ValueError):
            _ = matrix + np.ones((4,))

    def test_broadcasting_does_not_mutate_source(self):
        source = np.array([[1, 2], [3, 4]])
        _ = source + np.array([10, 20])
        self.assertEqual(source.tolist(), [[1, 2], [3, 4]])


if __name__ == "__main__":
    unittest.main()
