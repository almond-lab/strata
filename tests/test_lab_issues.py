"""Promised domains, exact branch reduction, reset legality and native batching."""
from dataclasses import asdict, replace
from fractions import Fraction
import itertools
import shlex

import kickmix as km
import numpy as np
import pytest

from strata.lab import exhaustive, validate_lab_receipt
from strata.lab_program import compile_program
from strata.model import Contract, Register, StrataError


IDENTITY = Contract('identity', (Register('x', (0,)), Register('scratch', (1,), role='clean')), 'identity')
REFERENCE = lambda inp, scale: ({'x': inp['x']}, 0)


def repeated(n, reuse=False):
    return '\n'.join(f'CX q0 q1\nHMR q1 b{0 if reuse else k}\nZ q0 if b{0 if reuse else k}'
                     for k in range(n))


def native_full(text):
    """Independent scalar harness: native gates and exact native phase access.

    Deliberately has no compiler, batching, shared reduction or injected-Z path.
    HMR's Kraus sign is written directly as an exact phase for each assignment.
    """
    rows = [shlex.split(line) for line in str(km.Circuit(text)).splitlines()]
    n = sum(words[0] == 'HMR' for words in rows)
    passed = True
    for branch, x in itertools.product(range(1 << n), range(2)):
        sim = km.Simulator(batch_size=1, seed=0)
        sim.write_across_shots(km.q(0), np.array([bool(x)]))
        sim.write_across_shots(km.q(1), np.array([False]))
        stack, m = [], 0
        for words in rows:
            op = words[0]
            condition = bool(sim.read_across_shots(km.b(int(words[-1][1:])))[0]) if 'if' in words else True
            if op == 'PUSH_CONDITION':
                stack.append(condition)
                continue
            if op == 'POP_CONDITION':
                if stack:
                    stack.pop()
                continue
            active = all(stack) and condition
            if op == 'HMR':
                if active:
                    q, b = km.q(int(words[1][1:])), km.b(int(words[2][1:]))
                    outcome = (branch >> m) & 1
                    kick = outcome * int(sim.read_across_shots(q)[0])
                    sim.write_shot_phase(0, Fraction(sim.read_shot_phase(0)) + kick)
                    sim.write_across_shots(q, np.array([False]))
                    sim.write_across_shots(b, np.array([bool(outcome)]))
                m += 1
            elif active:
                if op == 'R' and sim.read_across_shots(km.q(int(words[1][1:])))[0]:
                    passed = False
                    break
                # The scalar harness handles all stack/inline conditions itself.
                sim.do(km.Circuit(' '.join(words[:words.index('if')] if 'if' in words else words)))
        passed &= (int(sim.read_across_shots(km.q(0))[0]) == x and
                   not sim.read_across_shots(km.q(1))[0] and
                   Fraction(sim.read_shot_phase(0)) % 2 == 0)
    return bool(passed)


CASES = [
    (repeated(2), True, 'affine_hmr'),
    (repeated(3), True, 'affine_hmr'),
    (repeated(11), True, 'affine_hmr'),
    (repeated(4, reuse=True), True, 'affine_hmr'),
    ('CX q0 q1\nHMR q1 b0\nBIT_INVERT b0\nZ q0 if b0\nZ q0\n' + repeated(2), True, 'affine_hmr'),
    ('CX q0 q1\nHMR q1 b0\nPUSH_CONDITION if b0\nBIT_STORE0 b0\nZ q0\nPOP_CONDITION\n' + repeated(2), True, 'affine_hmr'),
    (repeated(1) + '\nCX q0 q1\nHMR q1 b1\nBIT_INVERT b1 if b0\nZ q0 if b1\nZ q0 if b0', True, 'affine_hmr'),
    (repeated(2) + '\nPUSH_CONDITION if b0\nZ q0 if b1\nPOP_CONDITION', False, 'none'),
    (repeated(2) + '\nX q0 if b0\nX q0 if b0', True, 'none'),
    (repeated(1) + '\nCX q0 q1 if b0\nHMR q1 b1 if b0\nZ q0 if b1', True, 'none'),
    ('CX q0 q1\nHMR q1 b0\n' + repeated(1), False, 'affine_hmr'),
    ('\n'.join(repeated(3).splitlines()[:-1]), False, 'affine_hmr'),
    (repeated(2) + '\nZ_POW q0 0.25 if b0', False, 'none'),
    (repeated(2) + '\nBIT_STORE1 b1 if b0\nZ q0 if b1', False, 'none'),
    (repeated(2) + '\nPUSH_CONDITION if b0\nPUSH_CONDITION if b0\nZ q0\nZ q0\nPOP_CONDITION\nPOP_CONDITION', True, 'affine_hmr'),
    (repeated(2) + '\nR q1', True, 'affine_hmr'),
]


@pytest.mark.parametrize('text,passed,reduction', CASES)
def test_reduction_matches_independent_native_full_enumeration(text, passed, reduction):
    assert native_full(text) is passed
    c = km.Circuit(text)
    reduced = exhaustive(c, IDENTITY, REFERENCE, batch_size=7)
    full = exhaustive(c, IDENTITY, REFERENCE, branch_strategy='full', batch_size=19)
    symbolic = exhaustive(c, IDENTITY, REFERENCE, branch_strategy='symbolic')
    assert reduced.passed is full.passed is symbolic.passed is passed
    assert reduced.reduction == reduction
    if passed:
        validate_lab_receipt(asdict(reduced))
        assert reduced.tested == reduced.cases * reduced.executed_branches
        assert reduced.represented_trajectories == full.tested
        assert full.tested == full.cases * full.branches


def test_nonlinear_condition_fails_only_on_combined_branch():
    result = exhaustive(km.Circuit(CASES[7][0]), IDENTITY, REFERENCE)
    assert result.counterexample['hmr_branch'] == 3
    assert result.reduction == 'none'


def test_promised_domain_deduplicates_and_binds_canonical_inputs():
    contract = replace(IDENTITY, promise='x-is-zero')
    circuit = km.Circuit('R q1\nZ q0')
    a = exhaustive(circuit, contract, REFERENCE, domain=[{'x': 0}, {'x': 0, 'scratch': 0}])
    b = exhaustive(circuit, contract, REFERENCE, domain=iter([{'scratch': 0, 'x': 0}]))
    assert a.passed and a.cases == 1 and a.domain_hash == b.domain_hash
    assert a.promise == 'x-is-zero' and a.domain_source == 'adapter'
    validate_lab_receipt(asdict(a))
    different = exhaustive(circuit, contract, REFERENCE, domain=[{'x': 1}])
    assert not different.passed and different.domain_hash != a.domain_hash
    with pytest.raises(StrataError, match='explicit adapter'):
        exhaustive(circuit, contract, REFERENCE)


@pytest.mark.parametrize('domain', [[], [{'x': -1}], [{'x': 2}], [{'x': .0}],
                                     [{'x': 0, 'scratch': 1}], [{'x': 0, 'extra': 0}], [{}]])
def test_invalid_promised_domains_fail_closed(domain):
    with pytest.raises(StrataError):
        exhaustive(km.Circuit('R q1'), replace(IDENTITY, promise='specified'), REFERENCE, domain=domain)


def test_all_input_domain_cannot_be_narrowed_and_hash_is_order_independent():
    circuit = km.Circuit('R q1')
    with pytest.raises(StrataError, match='weaken'):
        exhaustive(circuit, IDENTITY, REFERENCE, domain=[{'x': 0}])
    a = exhaustive(circuit, IDENTITY, REFERENCE)
    b = exhaustive(circuit, IDENTITY, REFERENCE, domain=[{'x': 1}, {'x': 0}, {'x': 1}])
    assert a.domain_hash == b.domain_hash


def test_checked_reset_cannot_erase_nonzero_or_dirty_scratch():
    bad = exhaustive(km.Circuit('CX q0 q1\nR q1'), IDENTITY, REFERENCE)
    assert not bad.passed and bad.counterexample['reason'] == 'reset_nonzero'
    assert bad.counterexample['input']['x'] == 1
    dirty = Contract('dirty', (Register('dirty', (0,), role='dirty'),), 'id')
    assert not exhaustive(km.Circuit('R q0'), dirty, lambda inp, _: ({}, 0)).passed
    # Disabled resets do not impose an obligation on a target they never touch.
    assert exhaustive(km.Circuit('R q0 if b0\nR q1'), IDENTITY, REFERENCE).passed
    branch_bad = exhaustive(km.Circuit(repeated(2) + '\nX q1 if b1\nR q1'), IDENTITY, REFERENCE)
    assert not branch_bad.passed and branch_bad.counterexample['hmr_branch'] == 2


def test_deterministic_shards_and_batches_are_honestly_partial():
    c = km.Circuit(repeated(3))
    small = exhaustive(c, IDENTITY, REFERENCE, batch_size=1)
    large = exhaustive(c, IDENTITY, REFERENCE, batch_size=100)
    assert small.tested == large.tested == 8
    assert small.domain_hash == large.domain_hash
    shards = [exhaustive(c, IDENTITY, REFERENCE, shard_index=i, shard_count=2) for i in range(2)]
    assert sum(s.tested for s in shards) == small.tested
    for shard in shards:
        assert shard.passed and not shard.exhaustive and shard.evidence_kind == 'sample'
        with pytest.raises(StrataError):
            validate_lab_receipt(asdict(shard))


def test_receipt_cannot_disguise_reduced_simulations_or_invent_scale():
    result = exhaustive(km.Circuit(repeated(3)), IDENTITY, REFERENCE)
    assert result.evidence_kind == 'exact_full' and result.scope == {}
    for update in ({'tested': result.represented_trajectories}, {'executed_branches': result.branches},
                   {'domain_hash': ''}, {'promise': ''}, {'evidence_kind': 'exact_small'},
                   {'represented_branches': 0}):
        with pytest.raises(StrataError):
            validate_lab_receipt({**asdict(result), **update})
    scaled = exhaustive(km.Circuit(repeated(3)), IDENTITY, REFERENCE, scale={'bits': 1})
    assert scaled.evidence_kind == 'exact_small'


def test_fractional_phases_remain_exact_with_reduced_hmr():
    c = km.Circuit(repeated(3) + '\nZ_POW q0 0.125')
    good = exhaustive(c, IDENTITY, lambda inp, _: ({'x': inp['x']}, Fraction(inp['x'], 8)))
    assert good.passed and good.reduction == 'affine_hmr'
    bad = exhaustive(c, IDENTITY, REFERENCE)
    assert not bad.passed and bad.counterexample['phase'] == '1/8'


def test_native_segments_are_cached_and_metadata_keeps_quoting():
    text = 'REGISTER r0 "if"\nAPPEND_TO_REGISTER q0 r0\nINFO_TAG "if"\nR q1\n' + 'X q0\nX q0\n' * 2000
    circuit = km.Circuit(text)
    program = compile_program(str(circuit))
    assert compile_program(str(circuit)) is program
    assert len(program.segments) < 10
    assert exhaustive(circuit, IDENTITY, REFERENCE).passed


def test_many_branch_program_uses_bounded_symbolic_proof_without_integer_overflow():
    c = km.Circuit(repeated(70) + '\nX q0 if b0\nX q0 if b0')
    result = exhaustive(c, IDENTITY, REFERENCE)
    assert result.passed and result.reduction == 'symbolic_hmr'
    assert result.branches == 1 << 70 and result.tested == result.cases == 2
    validate_lab_receipt(asdict(result))
    with pytest.raises(StrataError, match='budget'):
        exhaustive(c, IDENTITY, REFERENCE, branch_strategy='full')


def test_scaled_bundle_component_lab_keeps_frozen_bytes_and_exact_coverage():
    import copy
    import hashlib
    from strata.bundle import Bundle, BundleFrontend, Component
    from strata.component_lab import ComponentLab, run_component_labs, validate_component_lab

    fixed = b'fixed numeric table\n'
    frontend = BundleFrontend([Component('model', 'small.kmx', 'kickmix'),
                               Component('table', 'table.dat', frozen=True,
                                         sha256=hashlib.sha256(fixed).hexdigest())])
    artifact = Bundle({'small.kmx': repeated(3).encode(), 'table.dat': fixed})
    tree = frontend.tree(artifact, 'bundle-reference')

    class Adapter:
        def lab_components(self, parameters):
            return {'model': ComponentLab(IDENTITY, REFERENCE)}

    result = run_component_labs(frontend, artifact, tree, {'bits': 1}, Adapter())
    assert result.passed and result.exhaustive and result.evidence_kind == 'component_scoped'
    assert result.components['model']['lab']['evidence_kind'] == 'exact_small'
    assert result.components['model']['lab']['cases'] == 2
    assert result.frozen == {'table': hashlib.sha256(fixed).hexdigest()}
    validate_component_lab(asdict(result), frontend)
    for field, replacement in [('components', {}), ('frozen', {}), ('exhaustive', False)]:
        with pytest.raises(StrataError):
            validate_component_lab({**asdict(result), field: replacement}, frontend)
    bad_hash = copy.deepcopy(asdict(result))
    bad_hash['components']['model']['lab']['circuit_hash'] = 'a' * 64
    with pytest.raises(StrataError):
        validate_component_lab(bad_hash, frontend)
    changed = Bundle({'small.kmx': repeated(3).encode(), 'table.dat': b'changed table'})
    with pytest.raises(StrataError, match='frozen'):
        run_component_labs(frontend, changed, tree, {'bits': 1}, Adapter())


def test_bundle_requires_each_mutable_component_and_rejects_failed_child():
    from strata.bundle import Bundle, BundleFrontend, Component
    from strata.component_lab import ComponentLab, run_component_labs

    frontend = BundleFrontend([Component('model', 'model.kmx', 'kickmix')])
    artifact = Bundle({'model.kmx': b'CX q0 q1'})
    tree = frontend.tree(artifact, 'bundle-reference')

    class Missing:
        def lab_components(self, parameters):
            return {}

    class Failing:
        def lab_components(self, parameters):
            return {'model': ComponentLab(IDENTITY, REFERENCE)}

    with pytest.raises(StrataError, match='every nonfrozen'):
        run_component_labs(frontend, artifact, tree, {}, Missing())
    result = run_component_labs(frontend, artifact, tree, {}, Failing())
    assert not result.passed and not result.exhaustive
    assert result.components['model']['lab']['counterexample']['values']['scratch']['actual'] == 1


def test_numeric_component_accepts_only_byte_bound_typed_external_receipt():
    import hashlib
    from strata.bundle import Bundle, BundleFrontend, Component
    from strata.callbacks import CheckRequest, CheckReceipt, FullArtifactLab
    from strata.component_lab import run_component_labs, validate_component_lab

    frontend = BundleFrontend([Component('table', 'table.json', 'data')])
    artifact = Bundle({'table.json': b'[1, 2, 3]'})
    tree = frontend.tree(artifact, 'numeric-reference')
    parameters, domain = {'precision': 3}, {'rows': [0, 1, 2]}
    raw_hash = hashlib.sha256(artifact.files['table.json']).hexdigest()
    request = CheckRequest('lab', artifact.files['table.json'], raw_hash, parameters, domain)
    receipt = asdict(CheckReceipt.for_request(request, passed=True))
    receipt['verifier_hash'] = 'f' * 64
    numeric = FullArtifactLab(True, False, 'full_artifact', {'parameters': parameters, 'domain': domain},
                              raw_hash, tree.children[0].contract.fingerprint, 'f' * 64, receipt)

    class Adapter:
        def lab_components(self, parameters):
            return {'table': numeric}

    result = run_component_labs(frontend, artifact, tree, parameters, Adapter())
    assert result.passed and not result.exhaustive
    validate_component_lab(asdict(result), frontend)
    different = Bundle({'table.json': b'[1, 2, 4]'})
    with pytest.raises(StrataError):
        run_component_labs(frontend, different, tree, parameters, Adapter())


def test_component_lab_hook_receives_actual_candidate_and_all_frozen_is_not_exhaustive():
    import hashlib
    from strata.bundle import Bundle, BundleFrontend, Component
    from strata.component_lab import run_component_labs, validate_component_lab

    content = b'frozen table'
    frontend = BundleFrontend([Component('fixed', 'fixed.dat', frozen=True,
                                         sha256=hashlib.sha256(content).hexdigest())])
    artifact = Bundle({'fixed.dat': content})
    tree = frontend.tree(artifact, 'frozen-reference')
    seen = []

    class Adapter:
        def lab_components(self, parameters, candidate):
            seen.append(candidate)
            return {}

    result = run_component_labs(frontend, artifact, tree, {}, Adapter())
    assert seen == [artifact]
    assert result.passed and not result.exhaustive
    validate_component_lab(asdict(result), frontend)
