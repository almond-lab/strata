# Affine carrier width at a cut

Use `strata carrier-screen cuts.npz` to test whether given Boolean data and retained helper signals can all be recovered by affine operations from a specified number of carried bits. Run `screen_carrier_cut(live, helpers, row_count=n, width=w)` in `strata.carrier_screen` for a single in-memory cut.

With the all-ones column included, the finite-table matrix of supplied functions has GF(2) rank `r`. Any representation with affine recovery needs at least `r - 1` carried bits. XOR, NOT and constants are free in this bound. For example, if three independent inputs x, y, z are live, retaining x XOR y costs no additional independent dimension. Retaining x AND y and x AND z instead raises the simultaneous bound from three to five on the eight legal inputs.

The result records independent assignment indices in `witness_rows`. Restrict the live and helper truth columns to those indices, include the constant, and recompute rank to replay the obstruction. The width comparison is local to each cut. A redesign can change which signals are needed or use nonlinear recovery; phase, workspace schedules and reachability require separate evidence.

## Packed NPZ input

- `cut_names` is a one-dimensional array of unique, nonempty Unicode names. The position of each name determines the numbered keys below.
- `row_counts` and `widths` each contain one integer per cut. Row counts are positive; widths may be zero.
- `live_i` and `helpers_i` are uint8 arrays of shape `(number_of_functions, ceil(row_counts[i]/8))`, produced by `np.packbits(functions, axis=1, bitorder="big")`. Use a zero-row array for an empty signal set. Padding bits in the final byte are ignored.
- Optional `row_ids_i` maps the assignment indices at cut i to integer or Unicode identifiers, for example the legal input that generated each native trace. The receipt retains both the zero-based witness indices and the identifiers.

The JSON receipt pins the NPZ SHA-256 and lists each cut's live and joint affine width lower bounds, the number of independent helper dimensions, the budget shortfall and the witnesses. A budget at or above the bound has status `within_rank_bound`; it leaves achievability open.

