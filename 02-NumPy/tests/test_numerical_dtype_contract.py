"""Numerical dtype exercises without datasets or models."""
import unittest

import numpy as np


class NumericDtypeContracts(unittest.TestCase):
    def test_int64_sum_preserves_large_integer_total(self):
        values = np.array([2_000_000_000, 2_000_000_000], dtype=np.int32)
        total = values.sum(dtype=np.int64)
        self.assertEqual(int(total), 4_000_000_000)

    def test_float_division_is_not_integer_truncation(self):
        values = np.array([1, 2, 3], dtype=np.int32)
        result = values / 2
        self.assertEqual(result.tolist(), [0.5, 1.0, 1.5])
        self.assertTrue(np.issubdtype(result.dtype, np.floating))

    def test_nan_requires_explicit_missing_value_policy(self):
        values = np.array([1.0, np.nan, 3.0])
        self.assertTrue(np.isnan(np.mean(values)))
        self.assertEqual(float(np.nanmean(values)), 2.0)


if __name__ == "__main__":
    unittest.main()
