"""Independent adversarial audit of native branch execution and certificates.

The oracle rewrites each chosen HMR outcome into one deterministic *whole*
native circuit. Native kickmix itself implements every classical condition and
stack snapshot; this shares neither the lab compiler nor its stack evaluator.
Each measurement target is swapped into a fresh zero wire to avoid native R's
random discarded-state phase. Original R gates have fresh legality markers.
"""
from dataclasses import asdict, replace
from fractions import Fraction
from pathlib import Path
import random
import shlex

import kickmix as km
import numpy as np
import pytest

from strata import open as open_project
from strata.lab import exhaustive, validate_lab_receipt
from strata.lab_program import compile_program
from strata.model import Contract, Register, StrataError


CONTRACT = Contract('audit', (Register('x', (0, 1)),
                             Register('scratch', (2,), role='clean')), 'audit:identity')


def native_branches(text):
    circuit = km.Circuit(text)
    lines = [shlex.split(line) for line in str(circuit).splitlines()]
    count = sum(words[0] == 'HMR' for words in lines)
    outputs = {}
    for branch in range(1 << count):
        forced, measurement, markers = [], 0, []
        fresh = circuit.num_qubits
        for words in lines:
            suffix = ' ' + ' '.join(words[words.index('if'):]) if 'if' in words else ''
            if words[0] == 'HMR':
                outcome = (branch >> measurement) & 1
                if outcome:
                    forced.append(f'Z {words[1]}{suffix}')
                forced.extend([f'SWAP {words[1]} q{fresh}{suffix}',
                               f'BIT_STORE{outcome} {words[2]}{suffix}'])
                fresh += 1
                measurement += 1
            else:
                if words[0] == 'R':
                    marker = fresh
                    fresh += 1
                    markers.append(km.q(marker))
                    forced.append(f'CX {words[1]} q{marker}{suffix}')
                forced.append(shlex.join(words))
        native = km.Circuit('\n'.join(forced))
        sim = km.Simulator(batch_size=4, seed=918)
        initial = np.array([[False, True, False, True],
                            [False, False, True, True], [False] * 4])
        sim.write_across_shots([km.q(q) for q in range(3)], initial)
        sim.do(native)
        actual = sim.read_across_shots([km.q(q) for q in range(3)])
        illegal = (np.any(sim.read_across_shots(markers), axis=0)
                   if markers else np.zeros(4, dtype=bool))
        outputs[branch] = [
            (sum(int(actual[q, x]) << q for q in range(2)), bool(actual[2, x]),
             Fraction(sim.read_shot_phase(x)) % 2, bool(illegal[x]))
            for x in range(4)]
    return outputs


def reference_from_zero(outputs):
    return lambda inp, _: ({'x': outputs[0][inp['x']][0]}, outputs[0][inp['x']][2])


def assert_matches_native(text):
    outputs = native_branches(text)
    expected = outputs[0]
    passed = all(not scratch and not illegal and (value, phase) == (expected[x][0], expected[x][2])
                 for branch in outputs.values()
                 for x, (value, scratch, phase, illegal) in enumerate(branch))
    reference = reference_from_zero(outputs)
    auto = exhaustive(km.Circuit(text), CONTRACT, reference, batch_size=3)
    full = exhaustive(km.Circuit(text), CONTRACT, reference,
                      branch_strategy='full', batch_size=11)
    assert auto.passed is full.passed is passed
    if passed:
        validate_lab_receipt(asdict(auto))
    return auto


def corrected(n=3):
    return '\n'.join(f'CCX q0 q1 q2\nHMR q2 b{i}\nCZ q0 q1 if b{i}' for i in range(n))


@pytest.mark.parametrize('suffix,reduction', [
    # Independent outcomes multiply; zero and every unit branch miss the error.
    ('PUSH_CONDITION if b0\nCZ q0 q1 if b1\nPOP_CONDITION', 'none'),
    ('PUSH_CONDITION if b0\nPUSH_CONDITION if b1\nZ q0 if b2\nPOP_CONDITION\nPOP_CONDITION', 'none'),
    ('BIT_STORE1 b1 if b0\nZ q0 if b1', 'none'),
    ('BIT_STORE0 b1 if b0\nZ q0 if b1', 'none'),
    ('BIT_INVERT b1 if b0\nZ q0 if b1\nZ q0 if b0\nZ q0 if b1', 'affine_hmr'),
    # x * x = x, whereas x * (1 xor x) = 0.
    ('PUSH_CONDITION if b0\nZ q0 if b0\nPOP_CONDITION', 'affine_hmr'),
    ('PUSH_CONDITION if b0\nBIT_INVERT b0\nX q0 if b0\nPOP_CONDITION', 'affine_hmr'),
    # These require full enumeration even when paired operations undo each other.
    ('X q0 if b1\nX q0 if b1', 'none'),
    ('Z_POW q0 0.25 if b1\nZ_POW q0 -0.25 if b1', 'none'),
    ('NEG if b1\nNEG if b1', 'none'),
    ('R q2 if b1', 'none'),
    # Unconditional exact fractional phases coexist with affine integer signs.
    ('Z_POW q0 0.125\nZ_POW q1 -0.25', 'affine_hmr'),
])
def test_branch_reduction_against_whole_native_execution(suffix, reduction):
    result = assert_matches_native(corrected() + '\n' + suffix)
    assert result.reduction == reduction


@pytest.mark.parametrize('text', [
    # A bit is overwritten inside its own pushed condition: the snapshot survives.
    corrected() + '\nPUSH_CONDITION if b0\nBIT_STORE0 b0\nZ q0\nZ q0\nPOP_CONDITION',
    # Push the inner snapshot while the outer condition is inactive.
    corrected() + '\nPUSH_CONDITION if b0\nPUSH_CONDITION if b1\nBIT_INVERT b1\nPOP_CONDITION\nZ q0 if b1\nPOP_CONDITION',
    # HMR both reads and overwrites the same condition bit.
    corrected() + '\nCX q0 q2 if b0\nHMR q2 b0 if b0\nZ q0 if b0',
    # Inactive HMR must preserve the previous destination bit and target wire.
    corrected() + '\nCX q0 q2\nHMR q2 b1 if b0\nZ q0 if b1\nCX q0 q2',
    # Inner disabled reset must not reject a nonzero wire.
    corrected() + '\nBIT_STORE0 b3\nPUSH_CONDITION if b3\nR q0\nPOP_CONDITION',
    corrected() + '\nPUSH_CONDITION if b0\nX q2\nR q2\nPOP_CONDITION',
    # Reuse both HMR destinations and temporary classical state repeatedly.
    '\n'.join('CX q0 q2\nHMR q2 b0\nBIT_INVERT b0\nZ q0 if b0\nZ q0' for _ in range(4)),
])
def test_runtime_condition_snapshot_and_reset_matches_native(text):
    assert_matches_native(text)


def test_seeded_programs_never_get_false_affine_certificates():
    rng = random.Random(76019)
    gates = ['Z q0', 'CZ q0 q1', 'X q0', 'CX q0 q1',
             'BIT_INVERT b0', 'BIT_INVERT b1', 'BIT_STORE0 b0',
             'BIT_STORE1 b1', 'Z_POW q1 0.25', 'R q2', 'NEG']
    for _ in range(40):
        lines = [corrected()]
        depth = 0
        for __ in range(12):
            choice = rng.randrange(6)
            if choice == 0 and depth < 3:
                lines.append(f'PUSH_CONDITION if b{rng.randrange(3)}')
                depth += 1
            elif choice == 1 and depth:
                lines.append('POP_CONDITION')
                depth -= 1
            else:
                gate = rng.choice(gates)
                if rng.randrange(2):
                    gate += f' if b{rng.randrange(3)}'
                lines.append(gate)
        lines.extend(['POP_CONDITION'] * depth)
        assert_matches_native('\n'.join(lines))


def test_every_trajectory_matches_native_for_seeded_conditional_hmr_programs():
    rng = random.Random(2181)
    gates = ['X q0', 'CX q0 q2', 'CCX q0 q1 q2', 'Z q0',
             'CZ q0 q1', 'Z_POW q1 0.125', 'BIT_INVERT b0',
             'BIT_STORE1 b1', 'BIT_STORE0 b2']
    for _ in range(24):
        lines, depth = ['X q2', 'X q2'], 0
        for measurement in range(3):
            for __ in range(5):
                choose = rng.randrange(5)
                if choose == 0 and depth < 3:
                    lines.append(f'PUSH_CONDITION if b{rng.randrange(3)}')
                    depth += 1
                elif choose == 1 and depth:
                    lines.append('POP_CONDITION')
                    depth -= 1
                else:
                    gate = rng.choice(gates)
                    if rng.randrange(2):
                        gate += f' if b{rng.randrange(3)}'
                    lines.append(gate)
            hmr = f'HMR q{rng.randrange(3)} b{rng.randrange(3)}'
            if rng.randrange(2):
                hmr += f' if b{rng.randrange(3)}'
            lines.append(hmr)
        lines.extend(['POP_CONDITION'] * depth)
        circuit = km.Circuit('\n'.join(lines))
        expected = native_branches(str(circuit))
        batch = [(branch, x) for branch in expected for x in range(4)]
        sim = km.Simulator(batch_size=len(batch), seed=155)
        sim.use_same_registers_as(circuit)
        wires = [km.q(q) for q in range(3)]
        initial = np.array([[(x >> q) & 1 for _, x in batch] for q in range(3)], dtype=bool)
        sim.write_across_shots(wires, initial)
        assert compile_program(str(circuit)).execute(sim, batch, lambda: None) is None
        actual = sim.read_across_shots(wires)
        for shot, (branch, x) in enumerate(batch):
            got = (sum(int(actual[q, shot]) << q for q in range(2)),
                   bool(actual[2, shot]), Fraction(sim.read_shot_phase(shot)) % 2, False)
            assert got == expected[branch][x]


def test_combined_branch_error_is_explicitly_found():
    text = corrected() + '\nPUSH_CONDITION if b0\nCZ q0 q1 if b1\nPOP_CONDITION'
    native = native_branches(text)
    assert all(native[b] == native[0] for b in (1, 2, 4))
    assert native[3] != native[0]
    result = exhaustive(km.Circuit(text), CONTRACT, lambda inp, _: ({'x': inp['x']}, 0))
    assert not result.passed and result.counterexample['hmr_branch'] == 3
    assert result.reduction == 'none'


def test_promised_domain_canonicalization_and_shard_partition():
    contract = replace(CONTRACT, promise='x in {0,2,3}')
    domain = [{'x': 3}, {'x': 0}, {'x': 2, 'scratch': 0}, {'x': 3}]
    reference = lambda inp, _: ({'x': inp['x']}, 0)
    circuit = km.Circuit(corrected())
    whole = exhaustive(circuit, contract, reference, domain=domain)
    shards = [exhaustive(circuit, contract, reference, domain=reversed(domain),
                         shard_count=3, shard_index=i, batch_size=i + 1)
              for i in range(3)]
    assert whole.cases == 3 and whole.executed_cases == 3
    assert all(r.domain_hash == whole.domain_hash for r in shards)
    assert sum(r.tested for r in shards) == whole.tested
    for shard in shards:
        assert shard.passed and not shard.exhaustive and shard.evidence_kind == 'sample'
        with pytest.raises(StrataError):
            validate_lab_receipt(asdict(shard))


@pytest.mark.parametrize('field,value', [
    ('hmr_count', True), ('cases', 1.0), ('branches', 7),
    ('executed_cases', 1), ('executed_branches', 8),
    ('represented_trajectories', 12), ('tested', 32),
    ('reduction', 'unproved'), ('promise', 'promised'),
    ('shard', {'index': 0, 'count': 2}), ('evidence_kind', 'sample'),
])
def test_receipt_rejects_inconsistent_or_partial_evidence(field, value):
    receipt = asdict(exhaustive(km.Circuit(corrected()), CONTRACT,
                               lambda inp, _: ({'x': inp['x']}, 0)))
    with pytest.raises(StrataError):
        validate_lab_receipt({**receipt, field: value})


@pytest.fixture
def project_lab(tmp_path):
    example = Path(__file__).parents[1] / 'examples' / 'and_fusion'
    for name in ('strata.toml', 'base.kmx'):
        (tmp_path / name).write_bytes((example / name).read_bytes())
    task = open_project(tmp_path / 'strata.toml').task('and')
    parameters = task.params(True)
    circuit, tree = task.adapter().build(parameters, {})
    result = task.run_lab(circuit, tree, parameters)
    return task, parameters, circuit, tree, result


def test_project_binds_exact_receipt_to_artifact_contract_parameters_and_promise(project_lab):
    task, parameters, circuit, tree, result = project_lab
    record = task.lab_record(result, circuit, tree, parameters)
    task.validate_lab(record)
    assert record['evidence_kind'] == 'exact_small'
    assert record['scope'] == parameters
    for mutation in ({'circuit_hash': '0' * 64}, {'contract_hash': '0' * 64},
                     {'scope': {'bits': 999}}, {'promise': 'weakened promise'},
                     {'evidence_kind': 'sample'}, {'exhaustive': False}):
        with pytest.raises(StrataError):
            task.lab_record(replace(result, **mutation), circuit, tree, parameters)


def test_project_rejects_raw_untyped_adapter_success(project_lab, monkeypatch):
    task, parameters, circuit, tree, _ = project_lab
    adapter = task.adapter()
    monkeypatch.setattr(adapter, 'lab', lambda *args, **kwargs: {'passed': True}, raising=False)
    with pytest.raises(StrataError, match='typed'):
        task.run_lab(circuit, tree, parameters)


def test_project_partial_receipt_cannot_certify(project_lab):
    task, parameters, circuit, tree, result = project_lab
    partial = task.run_lab(circuit, tree, parameters, {'shard_count': 2, 'shard_index': 0})
    assert partial.passed and not partial.exhaustive
    complete = task.lab_record(result, circuit, tree, parameters)
    record = {**complete, **asdict(partial)}
    with pytest.raises(StrataError, match='incomplete'):
        task.validate_lab(record)


def test_public_line_retains_shard_evidence_but_requires_complete_lab_for_build(project_lab):
    task, _, _, _, _ = project_lab
    project = task.project
    project.init()
    line = project.open_line(task.base_design(), 'fuse', ['root'],
                             implementation={'fused': True})
    shard = line.lab(bits=1, limits={'shard_count': 2, 'shard_index': 0})
    assert shard.passed and not shard.exhaustive and shard.evidence_kind == 'sample'
    saved = project.store.events(line.id)[-1]
    assert saved['stage'] == 'lab' and saved['evidence_kind'] == 'sample'
    with pytest.raises(StrataError):
        line.build()
    assert line.lab(bits=1).exhaustive
    assert line.build()['passed']
