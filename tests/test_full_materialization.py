from itertools import product

import kickmix as km

from strata.measured_boolean_synthesis import measured_xor_and_witness
from strata.measurement_equivalence import equivalent_independent_outputs
from strata.model import digest


def fanout(width, **options):
    # p=x*y, q=p*z. Both dirty outputs consume q, and the first also consumes p.
    rows = [list(x) for x in product((0, 1), repeat=3)]
    values = [[(x*y) ^ (x*y*z), x*y*z] for x, y, z in rows]
    cols = [sum(row[j] << i for i, row in enumerate(rows)) for j in range(3)]
    x, y, z = cols
    p = x & y
    gates = [dict(left=x, right=y, product=p, left_coefficients=2, right_coefficients=4),
             dict(left=p, right=z, product=p & z, left_coefficients=16, right_coefficients=8)]
    w = dict(value=2, witness=gates, output_coefficients=[16 ^ 32, 32],
             contract_hash=digest(dict(inputs=rows, outputs=values)))
    return measured_xor_and_witness(w, rows, values, input_wires=[0, 1, 2], target_wires=[3, 4],
                                    clean_wires=list(range(5, width)), num_qubits=width, **options)


def test_full_materialization_reuses_terminal_products_on_dirty_targets_and_reverses_order():
    result = fanout(7, classical_bit_offset=9)
    full = next(a for a in result['alternatives'] if a['schedule'].get('policy') == 'full_materialization')
    existing = [a for a in result['alternatives'] if 'policy' not in a['schedule']]
    assert any(a['cleanup_strategy'] == 'reverse' for a in existing)
    assert min(a['resources']['toffoli'] for a in existing) == 3
    assert full['resources']['toffoli'] == 2 and full['resources']['Q'] == 7
    assert full['resources_expected']['toffoli'] == 2 and full['ledger_matches_native']
    assert full['proof']['equivalent'] and full['introduced_private_bits'] == [9, 10]
    assert [s.get('node') for s in full['kernel_ledger']] == [0, 1, None, 1, 0]
    assert all(s['cleanup_proof']['status'] == 'proved' for s in full['kernel_ledger'] if s['hmr'])
    assert len(km.Circuit(full['candidate'])) == full['resource_ledger']['instructions']
    # Independent complete basis/sign comparison includes arbitrary prior target values.
    ref = 'CCX q0 q1 q5\nCCX q5 q2 q6\nCX q5 q3\nCX q6 q3\nCX q6 q4\nCCX q5 q2 q6\nCCX q0 q1 q5'
    assert equivalent_independent_outputs(ref, full['candidate'], clean_inputs=[5, 6],
                                          internal_bits=[9, 10])['equivalent']


def test_full_materialization_requires_clean_storage_and_keeps_guarded_fallbacks():
    result = fanout(6)
    assert all(a['schedule'].get('policy') != 'full_materialization' for a in result['alternatives'])
    assert any(a['cleanup_strategy'] == 'guarded_hmr' for a in result['alternatives'])
    limited = fanout(7, max_apply_steps=1)
    assert all(a['schedule'].get('policy') != 'full_materialization' for a in limited['alternatives'])
    assert any(a['cleanup_strategy'] == 'reverse' for a in limited['alternatives'])
