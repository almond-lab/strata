from fractions import Fraction

import pytest

from strata.accounting import conditional_gate_accounting, measured_resources
from strata.model import StrataError


def counts(text, **kwargs):
    result = conditional_gate_accounting(text, **kwargs)
    assert result['status'] == 'exact', result
    return result['expected_resources']


def test_nested_identical_outcome_is_one_half_not_one_quarter():
    text = 'HMR q3 b0\nPUSH_CONDITION if b0\nCCX q0 q1 q2 if b0\nPOP_CONDITION'
    report = conditional_gate_accounting(text)
    assert report['static_resources']['toffoli'] == 1
    assert report['expected_resources']['toffoli'] == '1/2'
    assert measured_resources(text, accounting='expected')['toffoli'] == 0.5


def test_independent_nested_outcomes_and_opposite_snapshotted_conditions():
    independent = 'HMR q3 b0\nHMR q4 b1\nPUSH_CONDITION if b0\nCCX q0 q1 q2 if b1\nPOP_CONDITION'
    assert counts(independent)['toffoli'] == '1/4'
    opposite = 'HMR q3 b0\nPUSH_CONDITION if b0\nBIT_INVERT b0\nCCX q0 q1 q2 if b0\nPOP_CONDITION'
    assert counts(opposite)['toffoli'] == '0'


def test_nonlinear_derived_condition_and_correlated_conjunction():
    prefix = '''HMR q3 b0
HMR q4 b1
BIT_STORE0 b2
PUSH_CONDITION if b0
BIT_INVERT b2 if b1
POP_CONDITION
'''
    # b2=m0*m1. Conditioning on b0 again does not halve its probability.
    assert counts(prefix + 'PUSH_CONDITION if b0\nCCX q0 q1 q2 if b2\nPOP_CONDITION')['toffoli'] == '1/4'
    assert counts(prefix + 'BIT_INVERT b2\nCCX q0 q1 q2 if b2')['toffoli'] == '3/4'


def test_external_bits_require_declared_joint_distribution():
    text = 'PUSH_CONDITION if b0\nCCX q0 q1 q2 if b1\nPOP_CONDITION'
    report = conditional_gate_accounting(text)
    assert report['status'] == 'unknown'
    assert report['expected_resources']['toffoli'] is None
    assert report['unknown_external_bits'] == ['b0', 'b1']
    with pytest.raises(StrataError, match='external_classical_distribution_required'):
        measured_resources(text, accounting='expected')
    distribution = [dict(values=dict(b0=0, b1=0), probability='1/3'),
                    dict(values=dict(b0=1, b1=1), probability='2/3')]
    assert counts(text, external_distribution=distribution)['toffoli'] == '2/3'
    opposite = [dict(values=dict(b0=0, b1=1), probability='1/2'),
                dict(values=dict(b0=1, b1=0), probability='1/2')]
    assert counts(text, external_distribution=opposite)['toffoli'] == '0'


def test_classical_overwrite_removes_external_uncertainty_and_fractional_rotations_are_priced():
    assert counts('BIT_STORE1 b0\nCCX q0 q1 q2 if b0')['toffoli'] == '1'
    assert counts('HMR q3 b0\nZ_POW q0 0.25 if b0')['t'] == '1/2'
    assert counts('HMR q3 b0\nZ_POW q0 0.125 if b0')['rotations'] == '1/2'
    # A masked fair bit is fair even when its external XOR offset is unknown.
    text = 'HMR q3 b0\nBIT_INVERT b0 if b1\nCCX q0 q1 q2 if b0'
    report = conditional_gate_accounting(text)
    assert report['expected_resources']['toffoli'] == '1/2'
    assert report['expected_resources']['instructions'] is None


def test_register_metadata_keeps_native_instruction_accounting():
    text = 'REGISTER r0 "inputs"\nAPPEND_TO_REGISTER q0 r0\nCCX q0 q1 q2'
    static = measured_resources(text)
    expected = measured_resources(text, accounting='expected')
    assert static['instructions'] == expected['instructions'] == 1


def test_conditional_hmr_retains_old_destination_on_inactive_branches():
    text = 'HMR q3 b0\nBIT_STORE1 b1\nHMR q4 b1 if b0\nCCX q0 q1 q2 if b1'
    assert counts(text)['toffoli'] == '3/4'


def test_joint_distribution_validation_and_budgets_fail_closed():
    with pytest.raises(StrataError, match='sum exactly to one'):
        counts('CCX q0 q1 q2 if b0', external_distribution=[dict(values=dict(b0=1), probability='1/2')])
    report = conditional_gate_accounting('HMR q3 b0\nCCX q0 q1 q2 if b0', max_nodes=2)
    assert report['status'] == 'unknown' and report['reason'] == 'decision_node_budget'


def test_exact_expected_counts_match_enumerated_nonlinear_branches():
    text = '''HMR q3 b0
HMR q4 b1
HMR q5 b2
BIT_STORE0 b3
PUSH_CONDITION if b0
BIT_INVERT b3 if b1
POP_CONDITION
BIT_INVERT b3 if b2
CCX q0 q1 q2 if b3
PUSH_CONDITION if b3
CCX q0 q1 q2 if b0
POP_CONDITION
'''
    expected = sum(((a & b) ^ c) + (((a & b) ^ c) & a)
                   for a in range(2) for b in range(2) for c in range(2))
    assert Fraction(counts(text)['toffoli']) == Fraction(expected, 8)
