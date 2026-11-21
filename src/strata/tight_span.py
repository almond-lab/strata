"""Bounded constructor for a target span with one additional quadratic helper."""
from itertools import combinations
from time import monotonic

from .floors import _basis, _inside


def quadratic_quotient_floor(initial, targets):
    """After the first AND, every existing wire is still at most quadratic."""
    quadratic = _basis([*initial, *(a & b for a, b in combinations(initial, 2))])
    missing = len(_basis([*quadratic, *targets])) - len(quadratic)
    return missing + 1 if missing else 0


def _span(basis):
    values = [0]
    for row in basis:
        values += [x ^ row for x in values]
    return values


def _remainder(value, basis):
    for row in basis:
        if value & (1 << (row.bit_length() - 1)):
            value ^= row
    return value


def construct_one_helper(initial, targets, *, max_pairs, deadline):
    """Try one affine product before closing within inputs + targets + that helper.

    Each accepted product increases dimension by exactly one. If a returned
    witness meets an independently proved floor, it certifies the minimum.
    A miss leaves solutions with a later helper or other schedules open.
    """
    output_basis = _basis([*initial, *targets])
    if len(output_basis) + 1 > 11 or len(initial) > 7 or max_pairs <= 0:
        return None, 0
    # Complementing a factor changes its product only by an available affine
    # signal. The constant is present, so use one representative per pair.
    forms = [x for x in _span(initial) if not x & 1]
    helpers = {}
    checked = 0
    for a, b in combinations(forms, 2):
        if checked >= max_pairs or (checked % 256 == 0 and monotonic() >= deadline):
            return None, checked
        checked += 1
        term = a & b
        if not _inside(term, output_basis):
            helpers.setdefault(_remainder(term, initial), (a, b))
    for a, b in helpers.values():
        gates = [(a, b)]
        current = _basis([*initial, a & b])
        final = set(_span(_basis([*output_basis, a & b])))
        while not all(_inside(target, current) for target in targets):
            values = _span(current)
            have = set(values)
            forms = (x for x in values if not x & 1)
            for left, right in combinations(forms, 2):
                if checked >= max_pairs or (checked % 256 == 0 and monotonic() >= deadline):
                    return None, checked
                checked += 1
                product = left & right
                if product in final and product not in have:
                    gates.append((left, right))
                    current = _basis([*current, product])
                    break
            else:
                break
        else:
            return gates, checked
    return None, checked

