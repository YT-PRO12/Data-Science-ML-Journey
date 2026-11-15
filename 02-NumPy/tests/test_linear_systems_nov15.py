"""Numerical linear-algebra contracts for the NumPy learning module."""
import unittest

import numpy as np


class LinearSystemContracts(unittest.TestCase):
    def test_solve_reconstructs_the_original_right_hand_side(self):
        coefficient = np.array([[3.0, 1.0], [1.0, 2.0]])
        right_hand_side = np.array([9.0, 8.0])
        solution = np.linalg.solve(coefficient, right_hand_side)

        np.testing.assert_allclose(solution, [2.0, 3.0], rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            coefficient @ solution, right_hand_side, rtol=0, atol=1e-12
        )
        np.testing.assert_array_equal(
            coefficient, [[3.0, 1.0], [1.0, 2.0]]
        )

    def test_singular_system_raises_instead_of_fabricating_a_solution(self):
        singular = np.array([[1.0, 2.0], [2.0, 4.0]])
        with self.assertRaises(np.linalg.LinAlgError):
            np.linalg.solve(singular, np.array([3.0, 6.0]))


if __name__ == "__main__":
    unittest.main()
