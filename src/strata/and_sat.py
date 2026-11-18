"""Bounded SAT backend for affine/AND programs on finite care rows."""
from threading import Timer
from time import monotonic


def solve_and_count(available, targets, row_count, gate_count, *, deadline, clause_limit, conflict_limit):
    """Find a witness at this gate count, or prove no such program exists."""
    from pysat.solvers import Glucose4

    variable = 0
    clauses = []

    def fresh():
        nonlocal variable
        variable += 1
        return variable

    def conjunction(a, b, c):
        clauses.extend(([-a, -b, c], [a, -c], [b, -c]))

    def parity(literals):
        result = literals[0]
        for other in literals[1:]:
            output = fresh()
            clauses.extend(([result, other, -output], [-result, -other, -output],
                            [result, -other, output], [-result, other, output]))
            result = output
        return result

    width = len(available)
    # A preflight bounds allocation before constructing the row constraints.
    size = row_count * (gate_count + 2) * (gate_count + len(targets) + 1) * (7 * (width + gate_count) + 1)
    if size > 8 * clause_limit:
        return 'sat_encoding_budget', None, 0

    gate_selectors = []
    for index in range(gate_count):
        gate_selectors.append([[fresh() for _ in range(width + index)] for _ in range(2)])
    for a, b in gate_selectors:
        # Complementing either operand changes the result by an affine
        # combination of already available wires. This symmetry is free.
        clauses.extend(([-a[0]], [-b[0]]))
    output_selectors = [[fresh() for _ in range(width + gate_count)] for _ in targets]

    for row in range(row_count):
        if monotonic() >= deadline:
            return 'search timeout; no minimum claimed', None, len(clauses)
        active_inputs = [i for i, column in enumerate(available) if column >> row & 1]
        products = []

        def row_value(selectors):
            terms = [selectors[i] for i in active_inputs]
            for selector, product in zip(selectors[width:], products):
                combined = fresh()
                conjunction(selector, product, combined)
                terms.append(combined)
            return parity(terms)  # The constant column is always active.

        for left, right in gate_selectors:
            a, b = row_value(left), row_value(right)
            result = fresh()
            conjunction(a, b, result)
            products.append(result)
        for selectors, target in zip(output_selectors, targets):
            wire = row_value(selectors)
            clauses.append([wire if target >> row & 1 else -wire])
        if len(clauses) > clause_limit:
            return 'sat_encoding_budget', None, len(clauses)

    with Glucose4(bootstrap_with=clauses) as solver:
        remaining = deadline - monotonic()
        if remaining <= 0:
            return 'search timeout; no minimum claimed', None, len(clauses)
        solver.conf_budget(min(conflict_limit, 2**31 - 1))
        timer = Timer(remaining, solver.interrupt)
        timer.daemon = True
        timer.start()
        try:
            solved = solver.solve_limited(expect_interrupt=True)
        finally:
            timer.cancel()
            timer.join()
        if solved is None:
            reason = 'search timeout; no minimum claimed' if monotonic() >= deadline else 'sat_conflict_budget'
            return reason, None, len(clauses)
        if solved is False:
            return 'unsat', None, len(clauses)
        model = {literal for literal in solver.get_model() if literal > 0}

    values, witness = list(available), []
    for selectors in gate_selectors:
        operands = []
        for mask in selectors:
            value = 0
            for selector, column in zip(mask, values):
                if selector in model:
                    value ^= column
            operands.append(value)
        a, b = operands
        witness.append((a,b))
        values.append(a & b)
    return 'sat', witness, len(clauses)
