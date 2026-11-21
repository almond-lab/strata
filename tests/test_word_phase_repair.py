import kickmix as km
import pytest

from strata.lab import exhaustive
from strata.lookup_structure import measured_value_surrogate
from strata.model import Contract, Register


LIMITS = dict(max_operations=400, max_variables=64, max_terms=4096, max_product_terms=4096)
# The two measured scratch bits store x0 XOR x1*x2 and x1 XOR x0*x2.
# The intervening affine basis change makes each bilinear term pick up x2.
PREFIX = [
    'CX q0 q3', 'CCX q1 q2 q3', 'CX q1 q4', 'CCX q0 q2 q4',
    'HMR q3 b0', 'HMR q4 b1', 'Z q0 if b0', 'Z q1 if b1',
    'X q0', 'X q1',
]
SUFFIX = [
    'CZ q1 q2 if b0', 'CZ q0 q2 if b1', 'Z q2 if b0', 'Z q2 if b1',
    'X q1', 'X q0',
]


def surrogate(gates, bits=(0, 1)):
    source = km.Circuit('\n'.join(gates))
    return source, *measured_value_surrogate(source, [3, 4], bits, LIMITS)


def test_two_private_word_bits_with_interleaved_and_rebased_repairs():
    source, candidate, evidence = surrogate(PREFIX + SUFFIX)
    assert candidate is not None and evidence['proof']['outcome_independence_proved']
    assert [[a for a, _ in m['phase_repairs']] for m in evidence['matched']] == [[6, 10, 12], [7, 11, 13]]
    contract = Contract('word erase', (Register('x', (0, 1, 2)), Register('scratch', (3, 4), 'clean')), 'identity')
    for native, cases in [(source, 32), (candidate, 8)]:
        result = exhaustive(native, contract, lambda x, _: ({'x': x['x']}, 0), branch_strategy='full')
        assert result.passed and result.tested == cases


def test_mismatched_repair_rejected_by_signed_proof():
    suffix = SUFFIX.copy()
    suffix[2] = 'Z q1 if b0'
    _, candidate, evidence = surrogate(PREFIX + suffix)
    assert candidate is None
    assert evidence['proof']['equivalent'] is not True


@pytest.mark.parametrize(('middle', 'bits'), [(['CX q3 q1'], (0, 1)), (['X q1 if b0'], (0, 1)), ([], (1,))])
def test_intervening_measured_wire_use_or_unsupported_outcome_rejected(middle, bits):
    _, candidate, evidence = surrogate(PREFIX + middle + SUFFIX, bits)
    assert candidate is None and evidence['status'] == 'not_applicable'
    assert 0 not in [int(m['outcome'][1:]) for m in evidence['matched']]


def test_reused_outcome_starts_new_private_epoch():
    gates = PREFIX + SUFFIX + ['CX q0 q3', 'CX q2 q3', 'HMR q3 b0', 'Z q2 if b0', 'Z q0 if b0']
    _, candidate, evidence = surrogate(gates)
    assert candidate is not None
    assert [len(m['phase_repairs']) for m in evidence['matched']] == [3, 3, 2]


def test_projected_word_debts_cover_all_public_inputs_and_check_unchanged_sign():
    from strata.word_debt import prove_word_cleanup

    # q5 is public. The identical unconditional diagonal may depend on it,
    # but no debt or reset does. Both q5 values remain free.
    source, candidate, proof = surrogate(PREFIX + SUFFIX + ['CCZ q0 q1 q5'])
    cert = proof['proof']
    assert cert['kind'] == 'exact_private_word_cleanup'
    assert cert['checked_input_wires'] == [0, 1, 2]
    assert cert['assignments'] == 8 and cert['original_outcomes'] == 2
    a, b = str(source).splitlines(), str(candidate).splitlines()
    assert prove_word_cleanup(a, b[:-1] + ['CZ q0 q1'], [3, 4], [0, 1], LIMITS) is None
    # Without the declared clean value, q3 participates in the first obligation.
    assert prove_word_cleanup(a, b, [4], [0, 1], LIMITS) is None
    assert prove_word_cleanup(a, b, [3, 4], [0], LIMITS) is None
    # Keep the source/candidate edits identical, but inject a formerly unrelated public
    # bit into the value to be erased. Choosing just q5=0 would miss this failure.
    assert prove_word_cleanup(['CX q5 q3', *a], ['CX q5 q3', *b], [3, 4], [0, 1], LIMITS) is None
    # Both programs have an identical reset of a dirty public bit. It must be safe
    # on every assignment even when outside the apparent measured-word dependence.
    assert prove_word_cleanup(['R q5', *a], ['R q5', *b], [3, 4], [0, 1], LIMITS) is None


def test_word_debt_budget_and_forbidden_uses_fail_closed():
    from strata.word_debt import prove_word_cleanup

    source, candidate, _ = surrogate(PREFIX + SUFFIX)
    a, b = str(source).splitlines(), str(candidate).splitlines()
    for key in ('max_terms', 'max_product_terms', 'max_variables', 'max_operations'):
        assert prove_word_cleanup(a, b, [3, 4], [0, 1], {**LIMITS, key: 1}) is None
    # An extra outcome consumer, wire read, or intervening phase on the still-live
    # surrogate scratch cannot be copied through by the exact geometry check.
    for line in ('X q1 if b0', 'CX q3 q1', 'Z q3'):
        assert prove_word_cleanup(a[:7] + [line] + a[7:], b[:5] + [line] + b[5:],
                                  [3, 4], [0, 1], LIMITS) is None


def test_proof_checks_hidden_half_domain_and_conditional_sign():
    from strata.word_debt import prove_word_cleanup

    # On x0=0 the bad repayment would pass. The method receives a full basis
    # contract; a single conditional phase mismatch invalidates the certificate.
    a = ['CX q0 q2', 'CX q1 q2', 'HMR q2 b0', 'Z q1 if b0', 'Z q1 if b0']
    b = ['CX q0 q2', 'CX q1 q2', 'CX q1 q2', 'CX q1 q2']
    assert prove_word_cleanup(a, b, [2], [0], LIMITS) is None
    # An ordinary data-dependent conditioned flip is outside the sign-only theorem.
    assert prove_word_cleanup(['HMR q2 b0', 'X q1 if b0'], ['X q1'], [2], [0], LIMITS) is None
