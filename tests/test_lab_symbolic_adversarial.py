"""Independent audit of all-outcome claims and their finite-input boundary."""
from dataclasses import asdict
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import product

import kickmix as km
import pytest

from strata.factored import DecisionDAG
from strata.lab import exhaustive, validate_lab_receipt
from strata.model import Contract, Register, StrataError


def test_lazy_classical_and_quantum_phase_roots_survive_mid_input_collection(monkeypatch):
    calls = []
    collect = DecisionDAG.collect
    def observed_collect(self, roots):
        roots = list(roots)
        calls.append(len(roots))
        return collect(self, roots)
    monkeypatch.setattr(DecisionDAG, 'collect', observed_collect)
    rows = [f'HMR q0 b{i}' for i in range(10)] + ['BIT_STORE0 b100']
    rows += [f'BIT_INVERT b100 if b{i}' for i in range(10)]
    rows += ['X q1 if b100', 'NEG if b100', 'BIT_INVERT b101 if b100']
    # Cross the exact periodic collection point while nonconstant quantum and
    # phase roots are live and the dirty classical output remains lazy.
    rows += ['X q0'] * 1002
    rows += ['X q1 if b100', 'NEG if b100', 'BIT_INVERT b101 if b100']
    circuit = km.Circuit('\n'.join(rows))
    contract = Contract('collection', (Register('work', (0,), role='clean'), Register('x', (1,)),
                                      Register('saved', (101,), namespace='classical', role='dirty')), 'id')
    reference = lambda inp, _: ({'x': inp['x']}, 0)
    symbolic = exhaustive(circuit, contract, reference, branch_strategy='symbolic', max_symbolic_nodes=128)
    native = exhaustive(circuit, contract, reference, branch_strategy='full')
    assert symbolic.passed and native.passed and any(calls)
    assert symbolic.proof['proved_cases'] == 4 and symbolic.tested == 4
    assert native.tested == 4 * (1 << 10)
    assert 'not a general quantum-channel' in symbolic.proof['semantic_scope']
    assert 'inactive measurement slots' in symbolic.proof['outcome_semantics']
    validate_lab_receipt(asdict(symbolic))


def test_inactive_measurement_preserves_dirty_classical_bit_and_advances_formal_slot():
    contract = Contract('inactive', (Register('dirty', (0,), role='dirty'),
                                     Register('work', (1, 2), role='clean'),
                                     Register('old', (0,), namespace='classical', role='dirty'),
                                     Register('condition', (1,), namespace='classical')),
                        'id', promise='condition-zero')
    domain = [dict(dirty=a, old=b, condition=0) for a, b in product(range(2), repeat=2)]
    reference = lambda inp, _: ({'condition': 0}, 0)
    prefix = 'HMR q0 b0 if b1\nHMR q1 b2\nX q2 if b2'
    good = exhaustive(km.Circuit(prefix + '\nX q2 if b2'), contract, reference,
                      domain=domain, branch_strategy='symbolic')
    assert good.passed and good.domain_source == 'adapter' and good.promise == 'condition-zero'
    bad = exhaustive(km.Circuit(prefix), contract, reference, domain=domain, branch_strategy='symbolic')
    native = exhaustive(km.Circuit(prefix), contract, reference, domain=domain, branch_strategy='full')
    assert not bad.passed and not native.passed
    assert bad.counterexample['hmr_branch'] == 2
    assert bad.counterexample['values']['work']['actual'] == 2


def test_full_native_precision_phase_carry_wrap_matches_independent_branch_runner():
    first = Fraction(str(km.Circuit('Z_POW q1 0.1')).split()[2])
    with localcontext() as context:
        context.prec = 200
        complement = str(Decimal((2 - first).numerator) / Decimal((2 - first).denominator))
    prefix = ['HMR q0 b0', 'HMR q0 b1', 'BIT_STORE0 b2', 'PUSH_CONDITION if b0',
              'BIT_STORE1 b2 if b1', 'POP_CONDITION', 'PUSH_CONDITION if b2',
              'Z_POW q1 0.1', f'Z_POW q1 {complement}']
    contract = Contract('dyadic-carry', (Register('work', (0,), role='clean'), Register('x', (1,))), 'id')
    reference = lambda inp, _: ({'x': inp['x']}, 0)
    good = km.Circuit('\n'.join(prefix + ['POP_CONDITION']))
    result = exhaustive(good, contract, reference, branch_strategy='symbolic')
    assert result.passed and result.proof['phase_bits'] == first.denominator.bit_length()
    assert result.proof['phase_bits'] >= 128
    assert exhaustive(good, contract, reference, branch_strategy='full').passed
    # A tiny nonzero dyadic phase conditioned on a product of outcomes must not
    # disappear through float rounding or a phase-flip-only native comparison.
    broken = km.Circuit('\n'.join(prefix + ['Z_POW q1 0.000000000000000000000000000001', 'POP_CONDITION']))
    symbolic = exhaustive(broken, contract, reference, branch_strategy='symbolic')
    native = exhaustive(broken, contract, reference, branch_strategy='full')
    assert not symbolic.passed and not native.passed
    assert symbolic.counterexample['hmr_branch'] == 3
    assert Fraction(symbolic.counterexample['phase']) != 0
    assert symbolic.counterexample['phase'] == native.counterexample['phase']


def test_adapter_input_domain_and_shards_cannot_claim_generic_all_inputs():
    circuit = km.Circuit('HMR q1 b0\nX q0 if b0\nX q0 if b0')
    contract = Contract('finite-inputs', (Register('x', (0,)), Register('work', (1,), role='clean')),
                        'id', promise='specified-syndromes')
    reference = lambda inp, _: ({'x': inp['x']}, 0)
    limited = exhaustive(circuit, contract, reference, domain=[dict(x=0)],
                         branch_strategy='symbolic', scale={'syndromes': 1, 'field_bits': 1})
    assert limited.passed and limited.exhaustive and limited.cases == 1
    assert limited.domain_source == 'adapter' and limited.promise == 'specified-syndromes'
    assert limited.evidence_kind == 'exact_small' and limited.scope == {'syndromes': 1, 'field_bits': 1}
    assert 'proved_cases only' in limited.proof['proof_applies_to']
    shard = exhaustive(circuit, contract, reference, domain=[dict(x=0), dict(x=1)],
                       branch_strategy='symbolic', shard_count=2, shard_index=1)
    assert shard.passed and not shard.exhaustive and shard.evidence_kind == 'sample'
    assert shard.proof['all_measurement_outcomes'] and shard.proof['proved_cases'] == 1
    with pytest.raises(StrataError):
        validate_lab_receipt(asdict(shard))
