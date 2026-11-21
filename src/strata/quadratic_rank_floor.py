"""Polar-rank lower bounds for complete-cube quadratic Boolean value maps."""

from collections import Counter
from itertools import islice
from math import comb

from .model import StrataError


def _polar_rank(form, pairs, n_inputs):
    """Rank over GF(2) of the alternating matrix of a quadratic form."""
    matrix = [0] * n_inputs
    while form:
        bit = form & -form
        i, j = pairs[bit.bit_length() - 1]
        matrix[i] ^= 1 << j
        matrix[j] ^= 1 << i
        form ^= bit
    pivots = {}
    for row in matrix:
        while row:
            pivot = row.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = row
                break
            row ^= pivots[pivot]
    return len(pivots)


def quadratic_rank_floor(input_rows, output_rows):
    """Return a certified AND lower bound or None when outside the supported scope.

    Rows follow the same Boolean contract as floors.exact_minimum_on_rows.
    Read at most 256 rows plus one overflow sentinel. Require a complete
    cube of at most eight variables, at most 64 output
    coordinates, degree at most two, and at most eight independent components
    modulo affine functions. All nonzero combinations of the independent
    quadratic components are examined. This proves a floor for general
    XOR/NOT/two-input AND straight-line circuits, including nested ANDs.

    After restricting a circuit to an affine flat on which its first k ANDs
    are affine, both operands of gate k+1 are affine. If either is constant
    the product is already affine. Otherwise, fixing one operand to zero
    costs one more codimension and makes the product zero. Thus a scalar
    computed with k ANDs is affine on some flat of codimension at most k.
    An affine flat for a quadratic has an isotropic direction space for its
    polar matrix. A rank-2r alternating form has isotropic dimension at
    most n-r, so the scalar requires at least r ANDs.

    With m independent outputs modulo affine, write each as an affine
    expression in the inputs and all t AND-gate outputs. A nonzero linear
    combination of the m outputs kills the coefficients of the last m-1
    ANDs. It is still non-affine and requires at least the minimum scalar
    floor among the nonzero combinations. This gives t >= m-1+min_floor.
    The highest single-component floor is also an unrestricted lower bound.
    """
    # Import locally so floors can use this routine without an import cycle.
    from .floors import _canonical_rows

    try:
        input_rows = list(islice(input_rows, 257))
        output_rows = list(islice(output_rows, 257))
        if len(input_rows) > 256 or len(output_rows) > 256:
            return None
        input_rows = [tuple(islice(row, 9)) for row in input_rows]
        output_rows = [tuple(islice(row, 65)) for row in output_rows]
    except TypeError as exc:
        raise StrataError("expected iterables of Boolean input and output rows") from exc
    if any(len(row) > 8 for row in input_rows) or any(len(row) > 64 for row in output_rows):
        return None
    inputs, outputs = _canonical_rows(input_rows, output_rows)
    n_inputs, n_outputs = len(inputs[0]), len(outputs[0])
    if len(inputs) != 1 << n_inputs:
        return None
    coefficients = [0] * len(inputs)
    for source, target in zip(inputs, outputs):
        address = sum(bit << i for i, bit in enumerate(source))
        coefficients[address] = sum(bit << i for i, bit in enumerate(target))
    # Boolean Möbius transform, operating on all output columns together.
    for i in range(n_inputs):
        stride = 1 << i
        for base in range(0, len(coefficients), 2 * stride):
            for low in range(base, base + stride):
                coefficients[low + stride] ^= coefficients[low]
    if any(value and mask.bit_count() > 2 for mask, value in enumerate(coefficients)):
        return None

    pairs = [(i, j) for i in range(n_inputs) for j in range(i + 1, n_inputs)]
    forms = [0] * n_outputs
    for p, (i, j) in enumerate(pairs):
        columns = coefficients[(1 << i) | (1 << j)]
        while columns:
            bit = columns & -columns
            forms[bit.bit_length() - 1] |= 1 << p
            columns ^= bit

    pivots = {}
    basis = []
    independent = []
    for output, form in enumerate(forms):
        residual = form
        while residual:
            pivot = residual.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = residual
                independent.append(output)
                basis.append(form)
                break
            residual ^= pivots[pivot]
        if len(basis) > 8:
            return None

    m = len(basis)
    ranks = Counter()
    combined = 0
    previous = 0
    for index in range(1, 1 << m):
        gray = index ^ (index >> 1)
        combined ^= basis[(gray ^ previous).bit_length() - 1]
        ranks[_polar_rank(combined, pairs, n_inputs)] += 1
        previous = gray
    minimum = min(ranks, default=0) // 2
    maximum = max(ranks, default=0) // 2
    value = max(maximum, m - 1 + minimum, 0)

    # In the more restrictive model with only affine AND operands, the output
    # coefficient vectors form a binary [t,m] code. A combination selecting
    # h products has polar rank <=2h. Apply the sphere-packing bound to the
    # resulting minimum distance. This separate bound never raises 'value'.
    layer_floor = value
    radius = max(0, (minimum - 1) // 2)
    while (1 << m) * sum(comb(layer_floor, i) for i in range(radius + 1)) > 1 << layer_floor:
        layer_floor += 1

    return {
        "method": "complete_cube_quadratic_polar_rank",
        "value": value,
        "model": "unrestricted XOR/NOT/constants and two-input ANDs",
        "inputs": n_inputs,
        "outputs": n_outputs,
        "nonlinear_output_rank": m,
        "independent_output_indices": independent,
        "minimum_nonzero_polar_rank": 2 * minimum,
        "maximum_polar_rank": 2 * maximum,
        "polar_rank_histogram": dict(sorted(ranks.items())),
        "affine_operand_only_bound": {
            "value": layer_floor,
            "minimum_code_distance": minimum,
            "reason": "binary sphere packing for one layer of ANDs with affine operands",
        },
    }

