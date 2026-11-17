"""Independent frozen whole-boundary discovery fixtures, without supplied cuts.

The source emitters below predate the first acceptance run. Discovery receives
only native bytes, the zero-scratch contract, and a cost expression. A bank
retirement must also pass independent native enumeration of every dirty public
input and exact phase; bounded searches remain counted failures to discover.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import kickmix as km
import numpy as np
import pytest

from strata.restructure import _check, restructure_file


@dataclass(frozen=True)
class Fixture:
    name: str
    family: str
    text: str
    clean: tuple

    @property
    def sha256(self):
        return hashlib.sha256(self.text.encode()).hexdigest()


def fixtures():
    bank = ['CCX q0 q1 q6', 'CCX q0 q2 q7', 'CCX q6 q3 q8', 'CCX q7 q3 q9']
    consumer = ['CX q8 q4', 'CX q9 q4', 'CX q6 q5', 'CX q7 q5']
    yield Fixture('distributed_twolevel', 'nested_distributed_products',
                  '\n'.join(bank + consumer + bank[::-1]), (6, 7, 8, 9))
    yield Fixture('shared_factor_outputs', 'linear_consumer_of_nonlinear_vector',
                  '\n'.join(bank + ['CX q8 q4', 'CX q9 q5'] + bank[::-1]), (6, 7, 8, 9))
    basis = ['CX q3 q1', 'CX q1 q2', 'X q0']
    yield Fixture('mixed_basis_twolevel', 'nested_distributed_products',
                  '\n'.join(basis + bank + consumer + ['CX q6 q4', 'CX q7 q4']
                            + bank[::-1] + basis[::-1]), (6, 7, 8, 9))
    reordered = [bank[i] for i in (1, 0, 3, 2)]
    rename = {0: 3, 1: 1, 2: 0, 3: 2, 4: 5, 5: 4, 6: 9, 7: 6, 8: 8, 9: 7}
    lines = [' '.join(f'q{rename[int(word[1:])]}' if word.startswith('q') else word
                      for word in line.split())
             for line in reordered + consumer + reordered[::-1]]
    yield Fixture('permuted_schedule', 'nested_distributed_products', '\n'.join(lines), (6, 7, 8, 9))
    products = ['CCX q0 q2 q5', 'CCX q0 q3 q6', 'CCX q1 q2 q7', 'CCX q1 q3 q8']
    encoded = ['CX q5 q9', 'CX q7 q9', 'CX q6 q10', 'CX q8 q10']
    observe = ['CX q1 q0', 'CX q10 q9', 'CCX q0 q9 q4', 'CX q10 q9', 'CX q1 q0']
    yield Fixture('parity_product', 'nonlinear_consumer_of_encoded_vector',
                  '\n'.join(products + encoded + observe + encoded[::-1] + products[::-1]),
                  (5, 6, 7, 8, 9, 10))
    products = ['CCX q0 q1 q5', 'CCX q2 q3 q6']
    encoded = ['CX q5 q7', 'CX q6 q7', 'CX q5 q8']
    yield Fixture('duplicated_reencoding', 'nonlinear_consumer_of_encoded_vector',
                  '\n'.join(products + encoded + ['CCX q7 q8 q4'] + encoded[::-1] + products[::-1]),
                  (5, 6, 7, 8))
    chain = ['CCX q0 q1 q11'] + [f'CCX q{previous} q{data} q{previous+1}'
                                for previous, data in zip(range(11, 18), range(2, 9))]
    yield Fixture('high_support_bound', 'out_of_default_truth_budget',
                  '\n'.join(chain + ['CCX q18 q9 q10'] + chain[::-1]), tuple(range(11, 19)))


# Filled from emitter output before the first discovery run; the fixed hashes
# prevent adapting a difficult fixture after inspecting its search result.
FIXTURE_HASHES = {'distributed_twolevel': '36da7b6c8300539f6ec48a8053f806595011389761b22da6f17b1288c6441134', 'shared_factor_outputs': 'dce16e06ebf787067436b72c2bb4dd6bf19fa00f0e3471fea34c193e6484ed7f', 'mixed_basis_twolevel': '734a4137b56aee6af651e0e640951fa7bd6e471bc53549384ed475a4ee80a371', 'permuted_schedule': 'c56dcb4f0f062e0f1fac0c1721d60b3bd2c4934c477965afaf3c1e45b98f82fb', 'parity_product': 'f9b13a18d29c7ff633f956535fd7b0f752fd39b462e408e19f9878f613a736d9', 'duplicated_reencoding': '09080673a3ed510aabd99450e269f0ba4702e2900136bfa5701066f01dd1af90', 'high_support_bound': '5513a72ef65ee3cb4e6b558d217ba1136a49938b36439dc09402cd8850e89e34'}

BUDGET = dict(rounds=1, beam=4, max_candidates=12, max_operations=200,
              max_variables=32, max_terms=65536, max_product_terms=65536)
OBJECTIVE = 'Q*toffoli'


def native_boundary(circuit, width, clean):
    public = [q for q in range(width) if q not in clean]
    count = 1 << len(public)
    initial = np.zeros((width, count), dtype=np.bool_)
    for j, q in enumerate(public):
        initial[q] = [(row >> j) & 1 for row in range(count)]
    sim = km.Simulator(batch_size=count, seed=199)
    wires = [km.q(q) for q in range(width)]
    sim.write_across_shots(wires, initial)
    sim.do(circuit)
    return sim.read_across_shots(wires), sim.read_phase_flipped_across_shots()


def run_frozen(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    frozen = list(fixtures())
    assert {f.name: f.sha256 for f in frozen} == FIXTURE_HASHES
    # Record inputs and the equal public search budgets before evaluating any.
    manifest = dict(objective=OBJECTIVE, budgets=BUDGET,
                    inputs=[dict(name=f.name, family=f.family, sha256=f.sha256, clean=list(f.clean))
                            for f in frozen])
    (root / 'frozen-inputs.json').write_text(json.dumps(manifest, indent=2))
    rows = []
    for fixture in frozen:
        source = root / f'{fixture.name}.kmx'
        source.write_text(fixture.text)
        result = restructure_file(source, root / fixture.name, clean_inputs=fixture.clean,
                                  cost=OBJECTIVE, **BUDGET)
        before = km.Circuit(fixture.text)
        after = km.Circuit(Path(result['best']['path']).read_text())
        original = native_boundary(before, before.num_qubits, fixture.clean)
        emitted = native_boundary(after, before.num_qubits, fixture.clean)
        independent_equal = all(np.array_equal(a, b) for a, b in zip(original, emitted))
        # Inspect physical retirement independently of constructor annotations.
        retired = [q for q in fixture.clean if q >= after.num_qubits]
        rows.append(dict(name=fixture.name, family=fixture.family, source_sha256=fixture.sha256,
                         status=result['status'], independent_native_equal=independent_equal,
                         before=result['baseline']['resources'], after=result['best']['resources'],
                         before_cost=result['baseline']['cost'], after_cost=result['best']['cost'],
                         discovery_seconds=result['seconds'], attempt_count=len(result['attempts']),
                         physical_clean_wires_retired=retired,
                         deep_bank_retirement=independent_equal and len(retired) >= 2
                            and result['best']['cost'] < result['baseline']['cost'],
                         mechanisms=[dict(name=a['mechanism'], status=a['status'],
                                          reason=a.get('reason'), delta=a.get('cost_delta'),
                                          structural_witness=a.get('construction', {}).get('structural_witness'))
                                     for a in result['attempts']],
                         receipt=str(root / fixture.name / 'restructure.json')))
    summary = dict(frozen=manifest, results=rows,
                   deep_bank_retirements=sum(row['deep_bank_retirement'] for row in rows),
                   bounded_or_no_gain=sum(not row['deep_bank_retirement'] for row in rows),
                   discovery_seconds=sum(row['discovery_seconds'] for row in rows),
                   independent_native_checks_passed=sum(row['independent_native_equal'] for row in rows))
    (root / 'acceptance-summary.json').write_text(json.dumps(summary, indent=2))
    return summary


@pytest.fixture(scope='module')
def discovery(tmp_path_factory):
    return run_frozen(tmp_path_factory.mktemp('whole-boundary-acceptance'))


def test_frozen_untagged_suite_discovers_deep_bank_retirements(discovery):
    assert all('INFO_TAG' not in fixture.text for fixture in fixtures())
    assert discovery['independent_native_checks_passed'] == len(FIXTURE_HASHES)
    winners = [row for row in discovery['results'] if row['deep_bank_retirement']]
    assert len(winners) >= 3
    assert len({row['family'] for row in winners}) >= 2
    for row in winners:
        assert row['after']['Q'] < row['before']['Q']
        assert row['after_cost'] < row['before_cost']


def test_bounded_searches_remain_visible_in_acceptance_denominator(discovery):
    assert discovery['deep_bank_retirements'] + discovery['bounded_or_no_gain'] == len(FIXTURE_HASHES)
    high = next(row for row in discovery['results'] if row['name'] == 'high_support_bound')
    attempt = next(a for a in high['mechanisms'] if a['name'] == 'joint_boundary_synthesis')
    assert attempt['status'] == 'not_constructed' and 'budget' in attempt['reason']


def test_discovered_replacement_does_not_erase_phase_or_dirty_obligations(discovery):
    winner = next(row for row in discovery['results'] if row['deep_bank_retirement'])
    fixture = next(f for f in fixtures() if f.name == winner['name'])
    receipt = json.loads(Path(winner['receipt']).read_text())
    before = km.Circuit(fixture.text)
    after_text = Path(receipt['best']['path']).read_text()
    private = receipt['best'].get('private_bits', ())
    limits = {key: BUDGET[key] for key in ('max_operations', 'max_variables', 'max_terms', 'max_product_terms')}
    changed_phase = km.Circuit(after_text + '\nZ q0')
    phase = _check(before, changed_phase, fixture.clean, limits, internal_bits=private)
    assert phase['equivalent'] is False
    # Removing the clean promise must block retirement of arbitrary dirty bits.
    dirty = _check(before, km.Circuit(after_text), (), limits, internal_bits=private)
    assert dirty.get('equivalent') is not True
    leaked = _check(before, km.Circuit(after_text + f'\nX q{fixture.clean[-1]}'), fixture.clean, limits,
                    internal_bits=private)
    assert leaked['equivalent'] is False
