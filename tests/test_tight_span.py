from time import monotonic

import kickmix as km
from strata.floors import _basis, _row_tables, exact_minimum_on_rows, verify_minimum_witness
from strata.tight_span import construct_one_helper, quadratic_quotient_floor


def test_five_gate_four_bit_inverse_is_proved_and_replayed():
    field = km.GF2Field(4)
    rows = [[(i >> bit) & 1 for bit in range(4)] for i in range(16)]
    outputs = [[((field.invert(i) if i else 0) >> bit) & 1 for bit in range(4)]
               for i in range(16)]
    r = exact_minimum_on_rows(rows, outputs, max_and=5, timeout=4,
                              max_states=100000, max_expansions=100000)
    assert r['backend'] == 'one_helper_span'
    assert r['proven_lower_bound'] == r['value'] == 5
    assert verify_minimum_witness(r, rows, outputs)['and_count'] == 5
    assert 0 < r['pairs_checked'] <= 100000


def test_quadratic_floor_respects_removed_rows_and_constructor_budget():
    # Degree-three product on the cube becomes zero on a strict care set.
    rows = [[(i >> j) & 1 for j in range(3)] for i in range(8)]
    outputs = [[int(i == 7)] for i in range(8)]
    initial, targets = _row_tables(rows, outputs)
    assert quadratic_quotient_floor(_basis(initial), targets) == 2
    short, partial_targets = _row_tables(rows[:-1], outputs[:-1])
    assert quadratic_quotient_floor(_basis(short), partial_targets) == 0
    assert construct_one_helper(_basis(initial), targets, max_pairs=0,
                                deadline=monotonic()+1) == (None, 0)
    assert construct_one_helper(_basis(initial), targets, max_pairs=100,
                                deadline=monotonic()-1) == (None, 0)

