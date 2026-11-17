"""Research preflight validates exact small evidence without promoting its scope."""
from dataclasses import asdict
import json

import kickmix as km
import pytest

from strata.experiments import create_experiment
from strata.lab import exhaustive
from strata.model import Contract, Register, StrataError
from strata.research_ladder import (default_research_ladder, record_tiny_check,
    record_resource_prediction, validate_research_decision)
from strata.store import Store
from test_v041_comparisons import manifest, retained_finish


def setup_tiny(tmp_path, *, exact=True):
    store = Store(tmp_path / 'store')
    path = manifest(tmp_path)
    small = create_experiment(store, path, {'quality': 2})
    target = create_experiment(store, path, {'quality': 8})
    full = retained_finish(store, small, artifact=b'tiny-candidate')
    row = store.get(full)
    row['seconds'] = .02
    if exact:
        row['verifier']['receipt']['exact_check'] = dict(method='exhaustive', passed=True,
            complete=True, checked_cases=4, expected_cases=4,
            artifact_sha256=row['artifacts']['candidate']['sha256'], scale={'quality': 2})
    finish = store.put('experiment_finish', row)
    return store, path, small, target, finish


def prediction(store, target, tiny=None, basis=None, **changes):
    estimates = dict(wall_seconds=2, threads=1, memory_bytes=1024, storage_bytes=1024)
    estimates.update(changes)
    return record_resource_prediction(store, target_experiment=target, tiny_receipt=tiny,
        basis=basis, estimates=estimates, model='Quadratic scaling from measured tiny execution.',
        assumptions=['The same frozen generator and verifier apply.', 'Fixed overhead dominates tiny timing.'])


def budget():
    return dict(wall_seconds=3, threads=1, memory_bytes=2048, storage_bytes=2048)


def test_guidance_is_default_and_legacy_calls_remain_advisory(tmp_path):
    store, _, _, target, _ = setup_tiny(tmp_path)
    assert default_research_ladder()['recommended_stage'] == 'tiny_exact'
    result = validate_research_decision(store, target, {'seconds': 20})
    assert result['status'] == 'advisory_legacy'
    with pytest.raises(StrataError, match='tiny exact'):
        validate_research_decision(store, target, budget(), stage='full')


def test_full_build_requires_matching_exact_prototype_and_resource_model(tmp_path):
    store, _, _, target, finish = setup_tiny(tmp_path)
    tiny = record_tiny_check(store, result=finish, target_experiment=target, scale={'quality': 2})
    model = prediction(store, target, tiny)
    result = validate_research_decision(store, target, budget(), stage='full',
        evidence={'tiny_receipt': tiny, 'resource_prediction': model})
    assert result['status'] == 'ready' and result['full_size_verified'] is False
    assert store.get(tiny)['full_size_verified'] is False
    assert store.get(model)['evidence_kind'] == 'model'
    with pytest.raises(StrataError, match='every predicted resource'):
        validate_research_decision(store, target, {'wall_seconds': 3}, stage='full',
            evidence={'tiny_receipt': tiny, 'resource_prediction': model})


@pytest.mark.parametrize('change', ['bare_pass', 'sample', 'artifact', 'scale'])
def test_tiny_checker_cannot_use_a_boolean_sample_or_unrelated_artifact(tmp_path, change):
    store, _, _, target, finish = setup_tiny(tmp_path)
    row = store.get(finish)
    check = row['verifier']['receipt']['exact_check']
    if change == 'bare_pass':
        row['verifier']['receipt']['exact_check'] = {'passed': True}
    elif change == 'sample':
        check['checked_cases'] = 3
    elif change == 'artifact':
        check['artifact_sha256'] = store.blob(b'unrelated')
    else:
        check['scale'] = {'quality': 1}
    changed = store.put('experiment_finish', row)
    with pytest.raises(StrataError, match='complete exhaustive'):
        record_tiny_check(store, result=changed, target_experiment=target, scale={'quality': 2})


def test_generator_drift_and_non_scale_settings_cannot_reuse_tiny_pass(tmp_path):
    store, path, _, _, finish = setup_tiny(tmp_path)
    target = create_experiment(store, path, {'quality': 8, 'unused_worker_setting': 'different mechanism'})
    with pytest.raises(StrataError, match='same frozen generator'):
        record_tiny_check(store, result=finish, target_experiment=target, scale={'quality': 2})
    worker = path.parent / 'worker.py'
    worker.write_text(worker.read_text() + '\n# changed generator\n')
    target = create_experiment(store, path, {'quality': 8})
    with pytest.raises(StrataError, match='same frozen generator'):
        record_tiny_check(store, result=finish, target_experiment=target, scale={'quality': 2})


def test_exact_tiny_oracle_may_differ_from_full_official_checker(tmp_path):
    store, path, _, _, finish = setup_tiny(tmp_path)
    verifier = path.parent / 'verify.py'
    verifier.write_text(verifier.read_text() + '\n# distinct official full-size checker\n')
    target = create_experiment(store, path, {'quality': 8})
    tiny = record_tiny_check(store, result=finish, target_experiment=target, scale={'quality': 2})
    assert store.get(tiny)['target_experiment'] == target
    spec = json.loads(path.read_text()); spec['contract'] = {'task': 'unrelated boundary'}
    path.write_text(json.dumps(spec))
    other = create_experiment(store, path, {'quality': 8})
    with pytest.raises(StrataError, match='contract family'):
        record_tiny_check(store, result=finish, target_experiment=other, scale={'quality': 2})


def test_native_lab_receipt_must_bind_artifact_scale_and_contract(tmp_path):
    circuit = km.Circuit('X q0\nX q0')
    contract = Contract('identity', (Register('x', (0,)),), 'identity')
    exact = asdict(exhaustive(circuit, contract, lambda inp, _: ({'x': inp['x']}, 0), scale={'quality': 2}))
    store = Store(tmp_path / 'store')
    path = manifest(tmp_path, contract={'tiny_contract_hash': exact['contract_hash']})
    small = create_experiment(store, path, {'quality': 2})
    target = create_experiment(store, path, {'quality': 8})
    finish = store.get(retained_finish(store, small, artifact=str(circuit).encode()))
    finish['seconds'] = .02
    finish = store.put('experiment_finish', finish)
    proof = store.put('exact_lab_receipt', exact)
    tiny = record_tiny_check(store, result=finish, target_experiment=target, scale={'quality': 2}, proof=proof)
    assert store.get(tiny)['exact_basis'] == 'validated_exact_lab'
    forged = store.put('exact_lab_receipt', {**exact, 'exhaustive': False})
    with pytest.raises(StrataError, match='lab certificate'):
        record_tiny_check(store, result=finish, target_experiment=target, scale={'quality': 2}, proof=forged)


def test_slow_tiny_requires_feasibility_reason_and_prediction_respects_reservation(tmp_path):
    store, _, _, target, finish = setup_tiny(tmp_path)
    row = store.get(finish); row['seconds'] = 1.1
    slow = store.put('experiment_finish', row)
    with pytest.raises(StrataError, match='one-second'):
        record_tiny_check(store, result=slow, target_experiment=target, scale={'quality': 2})
    tiny = record_tiny_check(store, result=slow, target_experiment=target, scale={'quality': 2},
                             runtime_justification='Interpreter startup and exact checker setup exceed the target.')
    model = prediction(store, target, tiny, threads=1000)
    with pytest.raises(StrataError, match='frozen experiment reservation'):
        validate_research_decision(store, target, budget(), stage='full',
            evidence={'tiny_receipt': tiny, 'resource_prediction': model})
    with pytest.raises(StrataError, match='one-second'):
        validate_research_decision(store, target, {'wall_seconds': 2}, stage='tiny_exact')


def test_nonparameterizable_exception_requires_retained_basis_and_scaling(tmp_path):
    store, _, _, target, finish = setup_tiny(tmp_path, exact=False)
    exception = dict(kind='no_exact_tiny', reason='Only a fixed scientific dataset is available; the screen is empirical.',
                     alternative_evidence=[finish])
    model = prediction(store, target, basis=[finish])
    result = validate_research_decision(store, target, budget(), stage='full',
        evidence={'resource_prediction': model}, exception=exception)
    assert result['status'] == 'ready' and not result['exception']['full_size_verified']
    with pytest.raises(StrataError, match='retained alternative'):
        validate_research_decision(store, target, budget(), stage='full',
            evidence={'resource_prediction': model}, exception={**exception, 'alternative_evidence': []})


def test_later_rejection_revokes_tiny_preflight(tmp_path):
    store, _, small, target, finish = setup_tiny(tmp_path)
    tiny = record_tiny_check(store, result=finish, target_experiment=target, scale={'quality': 2})
    model = prediction(store, target, tiny)
    retained_finish(store, small, accepted=False, artifact=b'tiny-candidate')
    with pytest.raises(StrataError, match='conflicting scientific'):
        validate_research_decision(store, target, budget(), stage='full',
            evidence={'tiny_receipt': tiny, 'resource_prediction': model})


def test_sibling_experiment_rejection_revokes_tiny_and_final_preflight(tmp_path):
    store, path, _, target, finish = setup_tiny(tmp_path)
    tiny = record_tiny_check(store, result=finish, target_experiment=target, scale={'quality': 2})
    model = prediction(store, target, tiny)
    sibling = create_experiment(store, path, {'quality': 2, 'unused_worker_setting': 'other worker'})
    retained_finish(store, sibling, accepted=False, artifact=b'tiny-candidate')
    with pytest.raises(StrataError, match='conflicting scientific'):
        validate_research_decision(store, target, budget(), stage='full',
            evidence={'tiny_receipt': tiny, 'resource_prediction': model})
    final = retained_finish(store, target, artifact=b'final-candidate')
    sibling = create_experiment(store, path, {'quality': 8, 'unused_worker_setting': 'other worker'})
    retained_finish(store, sibling, accepted=False, artifact=b'final-candidate')
    for stage in ('final', 'polish'):
        with pytest.raises(StrataError, match='conflicting scientific'):
            validate_research_decision(store, target, budget(), stage=stage,
                evidence={'final_receipt': final})


def test_polish_binds_final_candidate_bytes_and_multiseed_requires_draw_policy(tmp_path):
    store, path, _, target, _ = setup_tiny(tmp_path)
    final = retained_finish(store, target, artifact=b'final-candidate')
    assert validate_research_decision(store, target, budget(), stage='polish',
        evidence={'final_receipt': final})['status'] == 'ready'
    wrong = create_experiment(store, path, {'quality': 9})
    with pytest.raises(StrataError, match='frozen input'):
        validate_research_decision(store, wrong, budget(), stage='polish', evidence={'final_receipt': final})
    (path.parent / 'selected.txt').write_bytes(b'final-candidate')
    spec = json.loads(path.read_text()); spec['inputs'] = {'selected': 'selected.txt'}
    path.write_text(json.dumps(spec))
    polish = create_experiment(store, path, {'quality': 9})
    assert validate_research_decision(store, polish, budget(), stage='polish',
        evidence={'final_receipt': final})['status'] == 'ready'
    with pytest.raises(StrataError, match='precommitted draw'):
        validate_research_decision(store, polish, budget(), stage='multi_seed', evidence={'final_receipt': final})


def test_final_selection_requires_all_precommitted_draws(tmp_path):
    store = Store(tmp_path / 'store')
    path = manifest(tmp_path, draw=True, policy={'aggregate': 'mean', 'draws': [{'draw': 0}, {'draw': 2}]})
    first = create_experiment(store, path, {'draw': 0})
    second = create_experiment(store, path, {'draw': 2})
    finish = retained_finish(store, first)
    with pytest.raises(StrataError, match='complete consistent'):
        validate_research_decision(store, first, budget(), stage='final', evidence={'final_receipt': finish})
    retained_finish(store, second)
    assert validate_research_decision(store, first, budget(), stage='final',
        evidence={'final_receipt': finish})['status'] == 'ready'


def test_audited_infrastructure_correction_allows_replacement_tiny_receipt(tmp_path):
    from strata.experiment_observations import invalidate_verification
    store, _, _, target, finish = setup_tiny(tmp_path)
    bad = store.get(finish)
    bad['status'] = 'rejected_by_verifier'
    bad['verifier']['receipt'] = {'accepted': False, 'process': {'returncode': -24}}
    failure = store.put('experiment_finish', bad)
    with pytest.raises(StrataError, match='conflicting scientific'):
        record_tiny_check(store, result=finish, target_experiment=target, scale={'quality': 2})
    invalidate_verification(store, failure, actor='researcher', reason='Wrapper hid the child CPU-limit signal.')
    assert record_tiny_check(store, result=finish, target_experiment=target, scale={'quality': 2})


def test_agent_packet_and_explicit_decision_use_the_preflight(tmp_path):
    from strata.experiment_reports import agent_packet, record_agent_decision
    store, _, _, target, finish = setup_tiny(tmp_path)
    packet = agent_packet(store)
    assert packet['research_ladder']['recommended_stage'] == 'tiny_exact'
    kwargs = dict(packet=packet['record'], hypothesis='Mechanism reduces full work.',
        decisive_experiment=target, expected_outcome='Same contract with fewer operations.',
        budget=budget(), reopening_condition='A tiny counterexample or measured scaling mismatch.')
    with pytest.raises(StrataError, match='tiny exact'):
        record_agent_decision(store, research_stage='full', **kwargs)
    assert not store.all('agent_decision')
    tiny = record_tiny_check(store, result=finish, target_experiment=target, scale={'quality': 2})
    model = prediction(store, target, tiny)
    result = record_agent_decision(store, research_stage='full',
        research_evidence={'tiny_receipt': tiny, 'resource_prediction': model}, **kwargs)
    assert result['research_preflight']['status'] == 'ready'
    assert record_agent_decision(store, **kwargs)['research_preflight']['status'] == 'advisory_legacy'
