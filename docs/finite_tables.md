# Finite Boolean table screening

Use a table when the legal inputs for one section can be enumerated. Generate each live feature once from the reference circuit and retain the assignment order. A feature column gives one Boolean value per assignment. All commands assume that the rows and column meanings of the input arrays agree. They do not establish that the supplied table covers a complete circuit domain.

```sh
strata table-screen care partial.npz
strata table-screen fit features.npz
strata table-screen repair donation.npz
```

Each NPZ contains an integer scalar `row_count`. Every truth column is a `uint8` array of length `ceil(row_count / 8)`, packed with `numpy.packbits(..., bitorder="big")`. Array index zero starts at bit 7 of byte zero. The final byte may contain unused low bits. A matrix is organized as one packed row per function. Variable-length, object and pickle data are unsupported.

| Command | Arrays | Result |
|---|---|---|
| `care` | `required_one`, `required_zero`, `candidates` | One verdict per candidate and the first conflicting zero-based row for each failed candidate. The two required sets must be disjoint. Remaining rows are free. |
| `fit` | `required_one`, `required_zero`, `features` | An affine XOR of the feature columns and a constant, or a list of contradictory rows. On those rows each feature and the constant have even parity, while the required output has odd parity. |
| `repair` | `available`, `targets` | The increase in GF(2) rank when target columns are added to the available features and a constant. This lower-bounds paid binary ANDs for a straight-line correction with free affine operations. |

For example, suppose the target is the product of two address bits. Test whether either single address bit, or their XOR, can supply the target. Then give the affine fitter the product as an extra feature and check that it finds the representation.

```python
import numpy as np

a = np.array([0, 0, 1, 1], dtype=np.uint8)
b = np.array([0, 1, 0, 1], dtype=np.uint8)
goal = a & b
common = dict(row_count=4, required_one=np.packbits(goal),
              required_zero=np.packbits(1 ^ goal))
np.savez("partial.npz", **common, candidates=np.packbits([a, b, a ^ b], axis=1))
np.savez("features.npz", **common, features=np.packbits([a, b, goal], axis=1))
np.savez("donation.npz", row_count=4, available=np.packbits([a, b], axis=1),
         targets=np.packbits([goal], axis=1))
```

The repair floor here is one. A two-input AND introduces at most one new linearly independent column. An implementation can cost more, and reordering addresses, changing the signals or computing a different physical table changes the problem. The same table format can represent only a sample of reachable states. A single valid counterexample suffices to reject a proposal, while acceptance on a sample requires a larger check. Phase, dirty outputs, scratch restoration and actual circuit price belong in the boundary verifier.

The Python entry points are `screen_care_table`, `fit_care_table` and `affine_repair_floor` in `strata.table_screen`. Each CLI receipt includes the SHA-256 of the input archive.


## Exact XOR/AND synthesis

For a small explicit table, the ordinary exact synthesis search now bounds the target dimension left after all quadratic input functions are included. Its first paid product is quadratic, so each additional independent target direction outside that span requires a later AND. A bounded search can try one quadratic helper, then add products inside the combined affine span of the inputs, requested outputs and helper. A construction that meets the proved lower bound is optimal in this Boolean model. The emitted selectors are replayed on every declared row.

On a complete cube with quadratic outputs, the report can also include the polar ranks of all nonzero combinations of independent output forms. Any scalar function with polar rank 2r needs at least r ANDs, as each AND can be made affine by at most one affine restriction. With m independent nonlinear outputs, a nonzero combination can cancel the last m−1 gates. These give a lower bound for unrestricted straight-line circuits. The receipt labels a separate and potentially higher bound that applies only to one layer with affine operands. Native measurement cleanup, physical width and reversible targets are priced after synthesis.
