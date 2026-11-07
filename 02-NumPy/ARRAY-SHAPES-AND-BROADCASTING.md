# NumPy shapes and broadcasting

A shape describes array dimensions, not the number of bytes. For a matrix with shape `(2, 3)`, axis 0 groups values down each column; axis 1 groups across each row.

```python
import numpy as np

matrix = np.array([[1, 2, 3], [4, 5, 6]])
assert matrix.shape == (2, 3)
assert matrix.sum(axis=0).tolist() == [5, 7, 9]
assert matrix.sum(axis=1).tolist() == [6, 15]
assert matrix.mean() == 3.5

offsets = np.array([[10], [20]])
assert (matrix + offsets).tolist() == [
    [11, 12, 13],
    [24, 25, 26],
]
```

Broadcasting aligns dimensions **from right to left**. Each pair must match or one of the dimensions must be 1. Here `(2, 3)` and `(2, 1)` are compatible. `(2, 3)` and `(4,)` are not.

## Common mistakes

- An `axis` argument identifies the axis *reduced*, not the one retained.
- Using `reshape` to silence a shape error can change the meaning of the data.
- NumPy integer arrays have fixed-width arithmetic; large products can overflow. For accurate large sums, select an appropriate dtype.
- Slices can be views: editing a slice may change the original array. Use `.copy()` for independent data.

Practice these invariants in a fresh notebook kernel. This guide is teaching material, not a claim of executed model evaluation.
