"""Small-first agent preflight, separate from execution and scientific acceptance.

Explicit research stages validate retained prerequisites. Legacy script wrapping
does not infer a task's scale or acquire a new scientific acceptance claim.
"""
from __future__ import annotations

import math

from .model import StrataError, digest

_STAGES = ('tiny_exact', 'scaling', 'full', 'final', 'polish', 'multi_seed')
_RESOURCES = ('wall_seconds', 'threads', 'memory_bytes', 'storage_bytes')


def default_research_ladder():
    return dict(version=1, recommended_stage='tiny_exact', target_tiny_seconds=1,
        stages=[
            dict(stage='tiny_exact', action='Run the same parameterized mechanism on a tiny exact case; aim below one second when feasible.'),
            dict(stage='scaling', action='Use retained tiny measurements to predict full cost, time, memory and storage; state assumptions.'),
            dict(stage='full', action='Build one bounded full candidate after checking the prediction against its declared resources.'),
            dict(stage='final', action='Select independently accepted candidate bytes before expensive refinement.'),
            dict(stage='polish', action='Spend slow refinement only on the selected final candidate.'),
            dict(stage='multi_seed', action='Run the precommitted final verification draws; preserve all outcomes.')],
        decision_fields=['research_stage', 'research_evidence', 'research_exception'],
        exception='Nonparameterizable or non-exact tasks require a reason and retained alternative evidence; scaling and full verification still apply.',
        scope='Tiny exactness covers only the checked instance. Scaling is a model. Neither proves full-size correctness.',
        legacy='Omitted stages remain advisory for existing callers; agents should declare a stage explicitly.')


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise StrataError(f'{label} must be a nonempty explanation')
    return value


def _manifest(store, experiment):
    value = experiment['manifest']
    return store.get(value) if isinstance(value, str) else value


def _family(store, experiment):
    manifest = _manifest(store, experiment)
    def sources(items):
        return {name: dict(files=item['files'], directory=item.get('directory', False))
                for name, item in items.items()}
    executables = []
    for stage in manifest['stages']:
        executable = stage['argv'][0]
        if executable.startswith('{dependency:') and executable.endswith('}'):
            executable = manifest['dependencies'][executable[len('{dependency:'):-1]]['path']
        executables.append(executable)
    return digest(dict(
        stages=[dict(argv=s['argv'], outputs=s.get('outputs')) for s in manifest['stages']],
        dependencies=sources(manifest.get('dependencies', {})), inputs=sources(manifest.get('inputs', {})),
        environment=manifest.get('environment', {}),
        runtimes={name: manifest.get('runtimes', {}).get(name) for name in executables},
        contract=manifest.get('contract'), compatibility=manifest.get('compatibility')))


def _artifact(store, finish, selected=None):
    artifacts = finish.get('artifacts', {})
    if selected is None:
        if len(artifacts) != 1:
            raise StrataError('select exactly one research candidate artifact')
        selected = next(iter(artifacts))
    matches = [value['sha256'] for name, value in artifacts.items()
               if selected in (name, value.get('sha256'))]
    if len(set(matches)) != 1:
        raise StrataError('research artifact is absent or ambiguous')
    store.verify_blob(matches[0])
    return matches[0]


def _successful_finish(store, handle, *, accepted=False):
    finish = store.get(handle, 'experiment_finish')
    if finish.get('status') not in {'accepted_by_verifier', 'artifact_validated', 'execution_finished'}:
        raise StrataError('research evidence requires a completed successful receipt')
    if accepted:
        from .experiment_reports import _independently_accepted
        if not _independently_accepted(store, finish):
            raise StrataError('final candidate requires independent acceptance')
    # A later contradictory verdict cannot be hidden behind an older receipt.
    from .experiment_comparisons import artifact_identity, verifier_identity
    from .experiment_observations import infrastructure_evidence
    manifest = _manifest(store, store.get(finish['experiment'], 'experiment'))
    artifact_key = artifact_identity(manifest, finish.get('artifacts', {}))
    def acceptance_contract(experiment, spec):
        if not spec.get('verifier'):
            return None
        return digest(dict(verifier=verifier_identity(spec, experiment),
            contract=spec.get('contract') if spec.get('contract_explicit', True) else None,
            compatibility=spec.get('compatibility'), draw_policy=spec.get('draw_policy')))
    contract_key = acceptance_contract(store.get(finish['experiment'], 'experiment'), manifest)
    corrections = {r['result']: r for _, r in store.all('experiment_verification_invalidation')}
    for handle, other in store.all('experiment_finish'):
        if other.get('status') == 'rejected_by_verifier':
            other_experiment = store.get(other['experiment'], 'experiment')
            other_manifest = _manifest(store, other_experiment)
            same_contract = (contract_key is not None and
                acceptance_contract(other_experiment, other_manifest) == contract_key)
            if other['experiment'] != finish['experiment'] and not same_contract:
                continue
            correction = corrections.get(handle)
            if (correction and correction.get('experiment') == other['experiment'] and
                infrastructure_evidence(store, other) == correction.get('evidence') and correction.get('evidence')):
                continue
            if artifact_key is not None and artifact_key == artifact_identity(other_manifest, other.get('artifacts', {})):
                raise StrataError('research receipt has a conflicting scientific rejection')
    return finish


def record_tiny_check(store, *, result, target_experiment, scale, artifact=None,
                      proof=None, runtime_justification=None):
    """Bind a complete exact check to a smaller instance of the frozen generator."""
    finish = _successful_finish(store, result)
    small = store.get(finish['experiment'], 'experiment')
    target = store.get(target_experiment, 'experiment')
    if not isinstance(scale, dict) or not scale or any(type(v) is not int or v < 1 for v in scale.values()):
        raise StrataError('tiny scale must name positive integer generator settings')
    if any(small.get('settings', {}).get(k) != v for k, v in scale.items()):
        raise StrataError('tiny scale differs from its frozen experiment settings')
    sizes = {k: target.get('settings', {}).get(k) for k in scale}
    if any(type(v) is not int or v < scale[k] for k, v in sizes.items()) or not any(sizes[k] > v for k, v in scale.items()):
        raise StrataError('target must be a larger instance of the declared scale settings')
    if (_family(store, small) != _family(store, target) or
        {k: v for k, v in small['settings'].items() if k not in scale} !=
        {k: v for k, v in target['settings'].items() if k not in scale}):
        raise StrataError('tiny and full experiments require the same frozen generator and contract family')
    sha = _artifact(store, finish, artifact)
    if proof is not None:
        from .lab import validate_lab_receipt
        exact = store.get(proof)
        validate_lab_receipt(exact)
        if exact['circuit_hash'] != sha or any(exact['scope'].get(k) != v for k, v in scale.items()):
            raise StrataError('exact lab receipt differs from the tiny artifact or scale')
        contract = _manifest(store, small).get('contract')
        if not isinstance(contract, dict) or contract.get('tiny_contract_hash') != exact['contract_hash']:
            raise StrataError('exact lab contract must be pinned as manifest contract.tiny_contract_hash')
        basis = 'validated_exact_lab'
    else:
        from .experiment_reports import _independently_accepted
        from .screening_evidence import _verified_artifacts
        if not _independently_accepted(store, finish) or sha not in _verified_artifacts(store, finish):
            raise StrataError('tiny evidence needs an artifact-bound independent exact checker or exact lab receipt')
        verifier = finish['verifier']
        verifier = store.get(verifier) if isinstance(verifier, str) else verifier
        exact = verifier.get('receipt', {}).get('exact_check', {})
        if (not isinstance(exact, dict) or exact.get('method') != 'exhaustive' or
            exact.get('passed') is not True or exact.get('complete') is not True or
            type(exact.get('expected_cases')) is not int or exact['expected_cases'] < 1 or
            type(exact.get('checked_cases')) is not int or exact['checked_cases'] != exact['expected_cases'] or
            exact.get('artifact_sha256') != sha or exact.get('scale') != scale):
            raise StrataError('tiny exact checker must bind complete exhaustive case counts, artifact and scale')
        basis = 'independent_exhaustive_checker_receipt'
    seconds = finish.get('seconds')
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
        raise StrataError('tiny receipt requires measured finite wall seconds')
    if seconds >= 1:
        _text(runtime_justification, 'tiny runtime above the one-second target')
    value = dict(version=1, result=result, proof=proof, artifact_sha256=sha,
        source_experiment=finish['experiment'], target_experiment=target_experiment,
        generator_family=_family(store, target), scale=scale, target_scale=sizes,
        seconds=seconds, runtime_justification=runtime_justification, exact_basis=basis,
        full_size_verified=False, scope='Exact evidence on this tiny instance only; no full-size correctness claim.')
    return store.put('research_tiny_check', value)


def _tiny(store, handle, target):
    row = store.get(handle, 'research_tiny_check')
    if row['target_experiment'] != target:
        raise StrataError('tiny evidence belongs to another full experiment')
    # Revalidate the underlying receipt and current contradictory evidence.
    record_tiny_check(store, result=row['result'], target_experiment=target,
        scale=row['scale'], artifact=row['artifact_sha256'], proof=row['proof'],
        runtime_justification=row.get('runtime_justification'))
    return row


def record_resource_prediction(store, *, target_experiment, estimates, model,
                               assumptions, tiny_receipt=None, basis=None):
    """Record a resource model, not measured full-size cost or correctness."""
    target = store.get(target_experiment, 'experiment')
    if not isinstance(estimates, dict) or set(estimates) != set(_RESOURCES):
        raise StrataError('prediction needs wall_seconds, threads, memory_bytes and storage_bytes')
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in estimates.values()) or estimates['wall_seconds'] <= 0 or estimates['threads'] < 1:
        raise StrataError('resource predictions must be finite and nonnegative with positive time and threads')
    _text(model, 'scaling model')
    if not isinstance(assumptions, list) or not assumptions:
        raise StrataError('scaling prediction requires explicit assumptions')
    for assumption in assumptions:
        _text(assumption, 'scaling assumption')
    if tiny_receipt is not None:
        _tiny(store, tiny_receipt, target_experiment)
    if basis is None:
        basis = []
    if not isinstance(basis, list) or not tiny_receipt and not basis:
        raise StrataError('prediction needs tiny evidence or retained alternative measurements')
    for handle in basis:
        store.get(handle)
    return store.put('research_resource_prediction', dict(version=1,
        target_experiment=target_experiment, generator_family=_family(store, target),
        tiny_receipt=tiny_receipt, basis=basis, estimates=estimates, model=model,
        assumptions=assumptions, evidence_kind='model', full_size_verified=False,
        scope='Declared extrapolation; measure the full run before claiming its cost or correctness.'))


def _exception(store, value):
    if not isinstance(value, dict) or value.get('kind') not in {'nonparameterizable', 'no_exact_tiny'}:
        raise StrataError('research exception must identify a nonparameterizable task or unavailable exact tiny check')
    _text(value.get('reason'), 'research exception reason')
    refs = value.get('alternative_evidence')
    if not isinstance(refs, list) or not refs:
        raise StrataError('research exception needs retained alternative evidence')
    for handle in refs:
        store.get(handle)
    return dict(kind=value['kind'], reason=value['reason'], alternative_evidence=refs,
                full_size_verified=False)


def validate_research_decision(store, decisive_experiment, budget, *, stage=None,
                               evidence=None, exception=None):
    """Validate an explicitly declared agent stage without inferring script scale."""
    target = store.get(decisive_experiment, 'experiment')
    if stage is None:
        return dict(status='advisory_legacy', recommended_stage='tiny_exact',
                    scope='No scale inferred; new agent workflows should declare research_stage.')
    if stage not in _STAGES:
        raise StrataError('unknown research stage: ' + str(stage))
    if not isinstance(budget, dict) or not budget or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in budget.values()):
        raise StrataError('research budgets must be positive finite named resources')
    evidence = {} if evidence is None else evidence
    if not isinstance(evidence, dict):
        raise StrataError('research evidence must be an object of retained handles')
    waived = _exception(store, exception) if exception is not None else None
    manifest = _manifest(store, target)
    if stage == 'tiny_exact':
        wall = budget.get('wall_seconds')
        if type(wall) not in (int, float) or not math.isfinite(wall) or wall <= 0:
            raise StrataError('tiny research decisions require an explicit wall_seconds budget')
        if wall >= 1:
            _text(evidence.get('runtime_justification'), 'tiny budget above the one-second target')
    if stage in {'scaling', 'full'}:
        tiny_handle = evidence.get('tiny_receipt')
        if tiny_handle:
            _tiny(store, tiny_handle, decisive_experiment)
        elif waived is None:
            raise StrataError('research stage requires a retained tiny exact receipt or justified exception')
        if stage == 'full':
            prediction = store.get(evidence.get('resource_prediction'), 'research_resource_prediction')
            if prediction['target_experiment'] != decisive_experiment or prediction['generator_family'] != _family(store, target):
                raise StrataError('resource prediction belongs to another frozen experiment')
            if tiny_handle != prediction.get('tiny_receipt'):
                raise StrataError('resource prediction must use the declared tiny evidence')
            if not tiny_handle and not set(waived['alternative_evidence']) <= set(prediction['basis']):
                raise StrataError('resource prediction must use the exception alternative evidence')
            estimates = prediction['estimates']
            limits = dict(manifest['resources'], wall_seconds=manifest['max_wall_seconds'])
            if any(estimates[k] > limits[k] for k in _RESOURCES):
                raise StrataError('predicted full resources exceed the frozen experiment reservation')
            if any(k not in budget or budget[k] < estimates[k] for k in _RESOURCES):
                raise StrataError('full decision budget must cover every predicted resource')
    if stage in {'final', 'polish', 'multi_seed'}:
        finish = _successful_finish(store, evidence.get('final_receipt'), accepted=True)
        sha = _artifact(store, finish, evidence.get('artifact'))
        from .screening_evidence import _verified_artifacts
        if sha not in _verified_artifacts(store, finish):
            raise StrataError('selected final artifact was not passed to its independent verifier')
        source = store.get(finish['experiment'], 'experiment')
        source_manifest = _manifest(store, source)
        if source_manifest.get('draw_policy'):
            from .experiment_comparisons import artifact_identity
            from .experiment_reports import experiment_report
            report = experiment_report(store, experiment=finish['experiment'])
            if report['metric'] is None:
                raise StrataError('final draw selection requires a reportable verifier-bound objective; acceptance-only draw reports are unsupported')
            artifact_key = artifact_identity(source_manifest, finish['artifacts'])
            if not any(row['artifact_identity'] == artifact_key and
                       finish['experiment'] in row['experiments'] and row['independently_accepted']
                       for row in report['artifact_comparisons']):
                raise StrataError('final selection requires a complete consistent precommitted draw set')
        inputs = {value for item in manifest.get('inputs', {}).values() for value in item['files'].values()}
        if finish['experiment'] != decisive_experiment and sha not in inputs:
            raise StrataError('final candidate bytes must be a frozen input of the proposed experiment')
        if stage == 'multi_seed' and not manifest.get('draw_policy'):
            raise StrataError('final multi-seed verification requires a precommitted draw policy')
    return dict(status='ready', stage=stage, evidence=evidence, exception=waived,
        full_size_verified=False, scope='Research scheduling prerequisites only; full execution and independent verification remain required.')
