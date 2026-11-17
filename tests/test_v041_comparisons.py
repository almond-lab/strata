"""Cross-worker comparison contracts and precommitted verifier draws (#57/#59)."""
import json
import os
import sys

import pytest

from strata.experiment_comparisons import IDENTITY_SCHEMA, verifier_identity
from strata.experiment_reports import agent_packet, experiment_report
from strata.experiments import create_experiment, create_sweep, load_manifest, run_experiment
from strata.model import StrataError
from strata.store import Store
from test_v04_experiments import local_transport


def manifest(tmp_path, name='case', *, draw=False, policy=None, **overrides):
    root = tmp_path / name
    root.mkdir(exist_ok=True)
    (root / 'worker.py').write_text(
        'import pathlib,sys\npathlib.Path(sys.argv[2]).write_text(sys.argv[1])\n# ' + name)
    (root / 'verify.py').write_text('''import json,pathlib,sys
value = int(pathlib.Path(sys.argv[1]).read_text())
draw = int(sys.argv[2]) if len(sys.argv)>2 else 0
print(json.dumps({'accepted': draw >= 0, 'resources': {'work': value + draw}}))
''')
    output = name + '.candidate'
    verifier = dict(version='independent/1', dependencies={'verify': 'verify.py'},
                    argv=[sys.executable, '{dependency:verify}', '{artifact:candidate}'])
    if draw:
        verifier['argv'].append('{setting:draw}')
    body = dict(version=1, settings={'quality': 2, 'unused_worker_setting': name, **({'draw': 0} if draw else {})},
        argv=[sys.executable, '{dependency:worker}', '{setting:quality}', output],
        dependencies={'worker': 'worker.py'}, outputs={'candidate': {'path': output, 'type': 'text'}},
        verifier=verifier, cost={'expression': 'work**2', 'unit': 'work2', 'resources': {'work': 'resources.work'}},
        limits={'timeout': 3, 'memory_bytes': 134217728, 'max_file_bytes': 1048576})
    if policy:
        body.update(draw_settings=['draw'] if draw else [], draw_policy=policy)
    body.update(overrides)
    path = root / 'manifest.json'
    path.write_text(json.dumps(body))
    return path


def retained_finish(store, experiment, *, resource=2, accepted=True, artifact=b'candidate', identity=None):
    exp = store.get(experiment, 'experiment')
    spec = exp['manifest']
    pin = verifier_identity(spec, exp) if identity is None else identity
    declared = spec['cost']
    from strata.cost import Formula
    inputs = {**exp['settings'], 'work': resource}
    value = Formula(declared['expression'])(inputs)
    cost = dict(value=value, expression=declared['expression'], unit=declared['unit'],
        resource_selectors=declared['resources'], inputs={'work': resource},
        verifier_identity=pin if accepted else None, evidence_kind='verified' if accepted else 'model')
    return store.put('experiment_finish', dict(experiment=experiment, attempt='attempt-' + experiment,
        status='accepted_by_verifier' if accepted else 'rejected_by_verifier', seconds=1,
        metrics={}, artifacts={'candidate': {'sha256': store.blob(artifact)}}, cost=cost,
        verified_cost=cost if accepted else None,
        verifier=dict(identity=pin, identity_schema=IDENTITY_SCHEMA, returncode=0,
                      receipt={'accepted': accepted, 'resources': {'work': resource}})))


def test_different_workers_output_paths_and_unused_settings_share_verified_contract(tmp_path, request):
    if os.environ.get('STRATA_REAL_SANDBOX') != '1':
        request.getfixturevalue('local_transport')
    store = Store(tmp_path / 'store')
    baseline = create_experiment(store, manifest(tmp_path, 'baseline'))
    candidate = create_experiment(store, manifest(tmp_path, 'candidate'), settings={'quality': 1})
    before, after = [run_experiment(store, e) for e in (baseline, candidate)]
    assert before['status'] == after['status'] == 'accepted_by_verifier', (before, after)
    assert before['verifier']['identity'] == after['verifier']['identity']
    report = experiment_report(store)
    assert len(report['verified_baselines']) == 1
    assert report['best_independently_accepted']['experiment'] == candidate
    assert report['best_independently_accepted']['value'] == 1
    assert report['best_independently_accepted']['delta_from_reference'] == -3
    assert len(report['artifact_comparisons']) == 2
    assert agent_packet(store)['best_independently_accepted']['value'] == 1
    # Reports are immutable evidence consumers; source removal is harmless.
    for path in tmp_path.glob('*/*.py'):
        path.unlink()
    assert experiment_report(store) == report


@pytest.mark.parametrize('change', ['dependency', 'formula', 'constant', 'unit', 'contract', 'input', 'environment'])
def test_semantic_changes_separate_comparison_contracts(tmp_path, change):
    store = Store(tmp_path / 'store')
    first = manifest(tmp_path, 'first')
    second = manifest(tmp_path, 'second')
    a = create_experiment(store, first)
    body = json.loads(second.read_text())
    if change == 'dependency':
        (second.parent / 'verify.py').write_text('# changed verifier policy\nprint("changed")')
    elif change == 'formula':
        body['cost']['expression'] = 'work**3'
    elif change == 'constant':
        body['cost']['expression'] = 'work*factor'
        body['settings']['factor'] = 7
    elif change == 'unit':
        body['cost']['unit'] = 'different units'
    elif change == 'contract':
        body['contract'] = 'different scientific acceptance contract'
    elif change == 'input':
        (second.parent / 'data.txt').write_text('different verifier reference data')
        body['inputs'] = {'reference': 'data.txt'}
        body['verifier']['argv'].append('{input:reference}')
    elif change == 'environment':
        body['environment'] = {'VERIFICATION_POLICY': 'strict'}
    second.write_text(json.dumps(body))
    b = create_experiment(store, second)
    retained_finish(store, a, artifact=b'base')
    retained_finish(store, b, artifact=b'candidate')
    report = experiment_report(store)
    assert len(report['verified_baselines']) == 2
    assert report['best_independently_accepted'] is None


def test_constants_that_change_in_same_formula_are_not_comparable(tmp_path):
    path = manifest(tmp_path)
    body = json.loads(path.read_text())
    body['settings']['factor'] = 1
    body['cost']['expression'] = 'work*factor'
    path.write_text(json.dumps(body))
    store = Store(tmp_path / 'store')
    for factor in [1, 9]:
        exp = create_experiment(store, path, {'factor': factor})
        retained_finish(store, exp, artifact=str(factor).encode())
    assert len(experiment_report(store)['verified_baselines']) == 2


@pytest.mark.parametrize('aggregate,expected', [('mean', 10), ('worst', 16), ('all', None)])
def test_declared_draws_aggregate_objectives_never_luckiest_draw(tmp_path, aggregate, expected):
    path = manifest(tmp_path, draw=True, policy=dict(aggregate=aggregate, draws=[{'draw': 0}, {'draw': 2}]))
    store = Store(tmp_path / 'store')
    for draw in [0, 2]:
        exp = create_experiment(store, path, {'draw': draw})
        retained_finish(store, exp, resource=2 + draw)
    report = experiment_report(store)
    aggregate_row, = report['artifact_comparisons']
    assert len(aggregate_row['observations']) == 2
    assert aggregate_row['independently_accepted']
    assert aggregate_row['value'] == expected
    assert (report['best_independently_accepted']['value'] if expected is not None
            else report['best_independently_accepted']) == expected
    assert agent_packet(store)['artifact_comparisons'] == report['artifact_comparisons']


def test_worst_draw_respects_maximization_direction(tmp_path):
    path = manifest(tmp_path, draw=True, policy=dict(aggregate='worst', draws=[{'draw': 0}, {'draw': 2}]))
    store = Store(tmp_path / 'store')
    for draw in [0, 2]:
        retained_finish(store, create_experiment(store, path, {'draw': draw}), resource=2 + draw)
    assert experiment_report(store, payload={'direction': 'max'})['best_independently_accepted']['value'] == 4


@pytest.mark.parametrize('failure', ['rejected', 'incomplete', 'duplicate_conflict', 'unexpected', 'different_artifact'])
def test_partial_failed_or_inconsistent_draw_sets_are_never_best(tmp_path, failure):
    path = manifest(tmp_path, draw=True, policy=dict(aggregate='mean', draws=[{'draw': 0}, {'draw': 2}]))
    store = Store(tmp_path / 'store')
    sweep = create_sweep(store, path, {'draw': [0, 2]}, seeds=[19], replicates=1, max_jobs=2)
    a, b = store.get(sweep)['experiments']
    retained_finish(store, a, resource=1)
    if failure != 'incomplete':
        retained_finish(store, b, resource=3, accepted=failure != 'rejected',
                        artifact=b'another candidate' if failure == 'different_artifact' else b'candidate')
    if failure == 'duplicate_conflict':
        retained_finish(store, a, resource=0)
    if failure == 'unexpected':
        retained_finish(store, create_experiment(store, path, {'draw': 9}), resource=0)
    report = experiment_report(store)
    assert report['best_independently_accepted'] is None
    assert not report['verified_baselines']
    assert all(not a['comparison_eligible'] and a['reasons'] for a in report['artifact_comparisons'])


def test_duplicate_artifact_and_duplicate_draw_do_not_increase_weight(tmp_path):
    path = manifest(tmp_path, draw=True, policy=dict(aggregate='mean', draws=[{'draw': 0}, {'draw': 2}]))
    store = Store(tmp_path / 'store')
    for draw, source in [(0, 'a'), (0, 'b'), (2, 'a')]:
        exp = create_experiment(store, path, {'draw': draw, 'unused_worker_setting': source})
        retained_finish(store, exp, resource=draw + 2)
    report = experiment_report(store)
    assert len(report['artifact_comparisons']) == 1
    assert len(report['artifact_comparisons'][0]['observations']) == 3
    assert report['best_independently_accepted']['value'] == 10


def test_different_precommitted_draw_sets_never_compare_best_against_best(tmp_path):
    store = Store(tmp_path / 'store')
    for name, draws in [('base', [0, 2]), ('candidate', [0, 4])]:
        path = manifest(tmp_path, name, draw=True, policy=dict(aggregate='mean', draws=[{'draw': d} for d in draws]))
        for draw in draws:
            retained_finish(store, create_experiment(store, path, {'draw': draw}),
                            resource=2 + draw, artifact=name.encode())
    report = experiment_report(store)
    assert len(report['verified_baselines']) == 2
    assert report['best_independently_accepted'] is None


@pytest.mark.parametrize('pending', [False, True])
def test_undeclared_verifier_setting_sweep_withholds_lucky_or_early_best(tmp_path, pending):
    path = manifest(tmp_path, draw=True)
    store = Store(tmp_path / 'store')
    sweep = create_sweep(store, path, {'draw': [0, 2]}, seeds=[19], replicates=1, max_jobs=2)
    a, b = store.get(sweep)['experiments']
    retained_finish(store, a, resource=1)
    if not pending:
        retained_finish(store, b, resource=3, accepted=False)
    report = experiment_report(store)
    assert report['best_independently_accepted'] is None
    assert not report['verified_baselines']
    assert 'without a declared draw policy' in report['artifact_comparisons'][0]['reasons'][0]
    assert experiment_report(store, experiment=a)['best_independently_accepted'] is None


def test_verifier_seed_and_replicates_are_complete_explicit_slots(tmp_path):
    path = manifest(tmp_path, policy=dict(aggregate='mean', seeds=[5, 7], replicates=2))
    body = json.loads(path.read_text())
    body['verifier']['argv'].append('{seed}')
    path.write_text(json.dumps(body))
    store = Store(tmp_path / 'store')
    with pytest.raises(StrataError, match='explicit seed'):
        create_experiment(store, path)
    for seed, replicate, resource in [(5, 0, 1), (5, 1, 2), (7, 0, 3)]:
        retained_finish(store, create_experiment(store, path, seed=seed, replicate=replicate), resource=resource)
    assert experiment_report(store)['best_independently_accepted'] is None
    retained_finish(store, create_experiment(store, path, seed=7, replicate=1), resource=4)
    assert experiment_report(store)['best_independently_accepted']['value'] == 7.5


@pytest.mark.parametrize('change', [
    {'draw_settings': ['missing']},
    {'draw_settings': ['draw']},
    {'draw_settings': ['draw', 'draw']},
    {'draw_policy': {'aggregate': 'min', 'draws': [{'draw': 1}]}},
    {'draw_policy': {'aggregate': 'mean', 'draws': [{'draw': 1}, {'draw': 1}]}},
    {'draw_policy': {'aggregate': 'mean', 'draws': [{}]}},
    {'draw_policy': {'aggregate': 'mean', 'draws': [{'draw': 1}], 'replicates': 0}},
])
def test_invalid_draw_commitments_are_rejected_before_execution(tmp_path, change):
    path = manifest(tmp_path, draw=True, policy={'aggregate': 'mean', 'draws': [{'draw': 1}]})
    body = json.loads(path.read_text())
    if change == {'draw_settings': ['draw']}:
        body.pop('draw_policy')
    body.update(change)
    path.write_text(json.dumps(body))
    with pytest.raises(StrataError, match='draw'):
        load_manifest(path)


def test_mutated_semantic_identity_cannot_certify_a_result(tmp_path):
    store = Store(tmp_path / 'store')
    exp = create_experiment(store, manifest(tmp_path))
    retained_finish(store, exp, identity='0' * 64)
    assert experiment_report(store)['best_independently_accepted'] is None


@pytest.mark.parametrize('explicit', [False, True])
def test_verifier_cannot_import_unpinned_worker_sibling(tmp_path, local_transport, explicit):
    path = manifest(tmp_path)
    body = json.loads(path.read_text())
    (path.parent / 'helper.py').write_text('OFFSET = 0\n')
    verifier = path.parent / 'verify.py'
    verifier.write_text('import helper\n' + verifier.read_text())
    body['dependencies']['helper'] = 'helper.py'
    if explicit:
        body['verifier']['dependencies']['helper'] = 'helper.py'
    path.write_text(json.dumps(body))
    store = Store(tmp_path / 'store')
    result = run_experiment(store, create_experiment(store, path))
    assert result['status'] == ('accepted_by_verifier' if explicit else 'verification_failed'), result
    if not explicit:
        assert 'ModuleNotFoundError' in result['verifier']['stderr']


def test_explicit_verifier_helper_bytes_define_comparison_identity(tmp_path):
    store = Store(tmp_path / 'store')
    for index in [0, 1]:
        path = manifest(tmp_path, 'case-' + str(index))
        body = json.loads(path.read_text())
        (path.parent / 'helper.py').write_text('OFFSET = ' + str(index))
        body['verifier']['dependencies']['helper'] = 'helper.py'
        path.write_text(json.dumps(body))
        retained_finish(store, create_experiment(store, path), artifact=str(index).encode())
    report = experiment_report(store)
    assert len(report['verified_baselines']) == 2
    assert report['best_independently_accepted'] is None


def test_verifier_dependency_sibling_layout_is_semantic_but_host_prefix_is_not(tmp_path, local_transport):
    store = Store(tmp_path / 'store')
    results = []
    for name, sibling in [('baseline', True), ('relocated', True), ('separated', False)]:
        path = manifest(tmp_path, name)
        body = json.loads(path.read_text())
        helper = path.parent / ('helper.txt' if sibling else 'elsewhere/helper.txt')
        helper.parent.mkdir(exist_ok=True)
        helper.write_text('identical pinned helper bytes')
        (path.parent / 'verify.py').write_text('''import json,pathlib
work = 2 if pathlib.Path(__file__).with_name('helper.txt').exists() else 1
print(json.dumps({'accepted': True, 'resources': {'work': work}}))
''')
        body['verifier']['dependencies']['helper'] = str(helper.relative_to(path.parent))
        path.write_text(json.dumps(body))
        results.append(run_experiment(store, create_experiment(store, path)))
    assert all(r['status'] == 'accepted_by_verifier' for r in results), results
    assert [r['verified_cost']['value'] for r in results] == [4, 4, 1]
    identities = [r['verifier']['identity'] for r in results]
    assert identities[0] == identities[1]
    assert identities[0] != identities[2]
    report = experiment_report(store)
    assert len(report['verified_baselines']) == 2
    assert report['best_independently_accepted'] is None


def test_verifier_source_layout_distinguishes_kind_and_directory_roots():
    from strata.experiment_comparisons import _source_layout
    source = lambda path, directory=False: dict(path=path, directory=directory)
    layout = _source_layout(dict(dependency={'a': source('/root/a.py'), 'b': source('/root/b.py'),
        'package': source('/root/pkg', True)}, input={'data': source('/root/data.txt')}))
    assert layout['file_groups'] == [['dependency:a', 'dependency:b'], ['input:data']]
    assert layout['directories'] == ['dependency:package']
    relocated = _source_layout(dict(dependency={'a': source('/other/a.py'), 'b': source('/other/b.py'),
        'package': source('/arbitrary/pkg', True)}, input={'data': source('/different/data.txt')}))
    assert relocated == layout
