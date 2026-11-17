"""Semantic objective contracts and precommitted, artifact-bound draw sets.

Execution identities retain the whole worker. Comparison identities deliberately
describe only the independent verifier and the objective it certifies.
"""
from __future__ import annotations

import ast
import itertools
import math
from pathlib import Path
import re

from .cost import Formula
from .model import StrataError, canonical, digest

IDENTITY_SCHEMA = 'strata-verifier/2'
_ARG = re.compile(r'^\{(setting|input|dependency|artifact):([^{}]+)\}$')


def _names(expression):
    formula = Formula(expression, functions={'ceil': math.ceil, 'floor': math.floor})
    return {n.id for n in ast.walk(formula.tree) if isinstance(n, ast.Name)} - set(formula.functions)


def validate_draw_policy(manifest):
    """Normalize a finite draw commitment before any experiment is launched."""
    names = manifest.get('draw_settings', [])
    if (not isinstance(names, list) or any(not isinstance(n, str) for n in names) or
        len(set(names)) != len(names) or not set(names) <= set(manifest.get('settings', {}))):
        raise StrataError('draw_settings must list distinct declared setting names')
    policy = manifest.get('draw_policy')
    if policy is None:
        if names:
            raise StrataError('draw_settings requires a precommitted draw_policy')
        return
    if (not isinstance(policy, dict) or set(policy) - {'aggregate', 'draws', 'seeds', 'replicates'} or
        policy.get('aggregate') not in {'mean', 'worst', 'all'}):
        raise StrataError('draw_policy requires aggregate mean, worst or all')
    if not manifest.get('verifier'):
        raise StrataError('draw_policy requires an independent verifier')
    draws = policy.get('draws', [{}] if not names else None)
    if (not isinstance(draws, list) or not draws or any(not isinstance(d, dict) or set(d) != set(names)
            or any(type(v) not in (str, int, float, bool) or type(v) is float and not math.isfinite(v)
                   for v in d.values()) for d in draws)):
        raise StrataError('draw_policy.draws must precommit every draw setting in each row')
    if len({canonical(d) for d in draws}) != len(draws):
        raise StrataError('duplicate draw rows require explicit draw_policy.replicates')
    argv = manifest['verifier'].get('argv', [])
    referenced = {m[2] for arg in argv if (m := _ARG.fullmatch(arg)) and m[1] == 'setting'}
    if not set(names) <= referenced:
        raise StrataError('draw_settings must be passed explicitly to the independent verifier')
    seeds = policy.get('seeds')
    if '{seed}' in argv and seeds is None:
        raise StrataError('verifier seed draws require precommitted draw_policy.seeds')
    if seeds is not None and (not isinstance(seeds, list) or not seeds or
            any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds)):
        raise StrataError('draw_policy.seeds must list distinct integer seeds')
    replicates = policy.get('replicates', 1)
    if type(replicates) is not int or replicates < 1:
        raise StrataError('draw_policy.replicates must be a positive integer')
    if len(draws) * len(seeds or [None]) * replicates > 100000:
        raise StrataError('draw_policy exceeds 100000 precommitted slots')
    manifest['draw_settings'] = sorted(names)
    manifest['draw_policy'] = dict(aggregate=policy['aggregate'],
        draws=sorted(draws, key=canonical), replicates=replicates,
        **({'seeds': sorted(seeds)} if seeds is not None else {}))


def _source(item):
    if not isinstance(item, dict) or 'files' not in item:
        return item
    return dict(files=item['files'], directory=item.get('directory', False),
        **({'basename': Path(item['path']).name} if item.get('path') else {}))


def verifier_sources(manifest):
    """The explicitly pinned source closure mounted for a verifier invocation.

    Worker-only files must not be visible as accidental sibling imports. Shared
    files enter this closure only through an argv reference or an explicit
    verifier dependency declaration (directories include their complete trees).
    """
    verifier = manifest.get('verifier') or {}
    dependencies = {**manifest.get('dependencies', {}), **verifier.get('dependencies', {})}
    inputs = manifest.get('inputs', {})
    used = {'dependency': set(verifier.get('dependencies', {})), 'input': set()}
    mappings = {'dependency': dependencies, 'input': inputs}
    for arg in verifier.get('argv', []):
        match = _ARG.fullmatch(arg)
        if match and match[1] in mappings:
            used[match[1]].add(match[2])
        elif not match:
            for kind, mapping in mappings.items():
                for name, item in mapping.items():
                    path = item.get('path') if isinstance(item, dict) else None
                    if path and (arg == path or item.get('directory') and Path(arg).is_absolute()
                                 and Path(arg).is_relative_to(path)):
                        used[kind].add(name)
    return {kind: {name: mappings[kind][name] for name in sorted(names)} for kind, names in used.items()}


def _source_layout(sources):
    """Pin observable sibling layout without pinning host directory prefixes."""
    files, directories, legacy = {}, [], []
    for kind, mapping in sources.items():
        for name, item in mapping.items():
            label = kind + ':' + name
            if item.get('directory'):
                # Directory declarations occupy independent named staging roots;
                # their internal relative paths are already pinned in files.
                directories.append(label)
            elif item.get('path'):
                # Inputs and dependencies use separate staging namespaces even
                # when their original source files share one host directory.
                files.setdefault((kind, str(Path(item['path']).parent)), []).append(label)
            else:
                legacy.append(label)
    return dict(file_groups=sorted(sorted(names) for names in files.values()),
                directories=sorted(directories), unknown_layout=sorted(legacy))


def verifier_contract(manifest, experiment, *, family=False):
    """Pin verifier code, arguments, environment and its own setting values.

    ``family`` omits setting values solely to detect undeclared changing verifier
    conditions. It is never itself authority to compare those conditions.
    """
    verifier = manifest.get('verifier') or {}
    source_closure = verifier_sources(manifest)
    dependencies, inputs = source_closure['dependency'], source_closure['input']
    used_dependencies = set(verifier.get('dependencies', {}))
    used_inputs = set()
    artifacts = {}
    draw_names = set(manifest.get('draw_settings', [])) if manifest.get('draw_policy') else set()
    argv = []
    for index, arg in enumerate(verifier.get('argv', [])):
        match = _ARG.fullmatch(arg)
        if match:
            kind, name = match.groups()
            if kind == 'setting':
                value = {'draw_setting': name} if name in draw_names else (
                    {'setting': name} if family else {'setting': name, 'value': experiment.get('settings', {}).get(name)})
            elif kind == 'artifact':
                value = {'artifact': artifacts.setdefault(name, len(artifacts))}
            else:
                (used_dependencies if kind == 'dependency' else used_inputs).add(name)
                value = {kind: name}
        elif arg in {'{seed}', '{replicate}'}:
            name = arg[1:-1]
            drawn = bool(manifest.get('draw_policy')) and (name == 'replicate' or 'seeds' in manifest['draw_policy'])
            value = {('draw_' if drawn else '') + name: True if drawn or family else experiment.get(name, 0 if name == 'replicate' else None)}
        else:
            value = arg
            for kind, sources in (('dependency', dependencies), ('input', inputs)):
                for name, item in sources.items():
                    path = item.get('path') if isinstance(item, dict) else None
                    if path and (arg == path or item.get('directory') and Path(arg).is_absolute() and Path(arg).is_relative_to(path)):
                        (used_dependencies if kind == 'dependency' else used_inputs).add(name)
                        value = {kind: name, 'relative': str(Path(arg).relative_to(path)) if arg != path else ''}
        if index == 0:
            runtime_path = dependencies[match[2]]['path'] if match and match[1] == 'dependency' else arg
            runtime = manifest.get('runtimes', {}).get(runtime_path)
            if runtime:
                value = dict(executable={k: v for k, v in runtime.items() if k != 'invocation'})
        argv.append(value)
    declaration = {k: v for k, v in verifier.items() if k not in {'argv', 'dependencies', 'source', 'limits'}}
    source = verifier.get('source')
    if isinstance(source, dict):
        source = {name: _source(row) for name, row in source.items()}
    elif source == next(iter(verifier.get('argv', [])), None):
        source = argv[0] if argv else source
    return dict(schema=IDENTITY_SCHEMA, declaration=declaration, argv=argv, source=source,
        dependencies={n: _source(dependencies[n]) for n in sorted(used_dependencies)},
        inputs={n: _source(inputs[n]) for n in sorted(used_inputs)},
        source_layout=_source_layout(source_closure),
        environment=manifest.get('environment', {}))


def verifier_identity(manifest, experiment):
    return digest(verifier_contract(manifest, experiment))


def comparison_identity(manifest, experiment, metric, unit):
    draws = set(manifest.get('draw_settings', [])) if manifest.get('draw_policy') else set()
    if metric == 'cost':
        cost = manifest.get('cost') or {}
        constants = _names(cost['expression']) - set(cost.get('resources', {}))
        objective = dict(declaration=cost, constants={n: {'draw_setting': n} if n in draws
            else experiment.get('settings', {}).get(n) for n in sorted(constants)})
    else:
        objective = dict(metric=metric, selector=(manifest.get('verifier') or {}).get('metrics', {}).get(metric))
    # Auto-generated display names are not scientific contracts. Old manifests
    # without the explicitness marker retain their existing contract conservatively.
    contract = manifest.get('contract') if manifest.get('contract_explicit', True) else None
    return digest(dict(verifier=verifier_identity(manifest, experiment), objective=objective,
        unit=unit, contract=contract, compatibility=manifest.get('compatibility'),
        draw_settings=sorted(draws), draw_policy=manifest.get('draw_policy')))


def artifact_identity(manifest, artifacts):
    names = []
    for arg in (manifest.get('verifier') or {}).get('argv', []):
        match = _ARG.fullmatch(arg)
        if match and match[1] == 'artifact' and match[2] not in names:
            names.append(match[2])
    if not names:  # Retained pre-v0.4.1 synthetic/legacy receipts.
        names = sorted(artifacts)
    if not names or any(name not in artifacts for name in names):
        return None
    return digest([artifacts[name]['sha256'] for name in names])


def _slot(manifest, experiment):
    policy = manifest['draw_policy']
    return dict(settings={n: experiment.get('settings', {}).get(n) for n in manifest.get('draw_settings', [])},
        seed=experiment.get('seed') if 'seeds' in policy else None, replicate=experiment.get('replicate', 0))


def _expected(manifest):
    policy = manifest['draw_policy']
    return [dict(settings=d, seed=s, replicate=r) for d, s, r in itertools.product(
        policy['draws'], policy.get('seeds', [None]), range(policy['replicates']))]


def verified_comparisons(store, experiments, comparisons, finishes, metric, direction, verified_objective):
    """Retain all evidence; rank only complete, matched artifact observations."""
    groups = {}
    invalidations = {r['result']: dict(record=h, **r)
                     for h, r in store.all('experiment_verification_invalidation')}
    planned_families = {}
    # A scoped report must not hide other precommitted verifier configurations.
    for _, experiment in store.all('experiment'):
        manifest = experiment['manifest']
        if isinstance(manifest, str):
            manifest = store.get(manifest)
        if manifest.get('verifier') and not manifest.get('draw_policy'):
            family = digest(verifier_contract(manifest, experiment, family=True))
            planned_families.setdefault(family, set()).add(verifier_identity(manifest, experiment))
    for row, finish in zip(comparisons, finishes):
        experiment = experiments[row['experiment']]
        manifest = experiment['manifest']
        if isinstance(manifest, str):
            manifest = store.get(manifest)
        if not manifest.get('verifier'):
            continue
        if not manifest.get('draw_policy') and not finish.get('verifier'):
            # A worker-only measurement is not a repeated independent check.
            continue
        evidence = verified_objective(store, finish, metric) if row['independently_accepted'] else None
        binding = manifest.get('verifier', {}).get('metrics', {}).get(metric)
        declared_unit = ((manifest.get('cost') or {}).get('unit') if metric == 'cost'
                         else manifest.get('metrics', {}).get(metric, {}).get('unit') or
                         (binding.get('unit') if isinstance(binding, dict) else None))
        unit = evidence.get('unit') if evidence else declared_unit
        if unit is None:
            # Historical receipts may omit metric declarations. Still use the
            # retained measured unit to prevent a failed draw disappearing.
            unit = (finish.get('metrics', {}).get(metric) or {}).get('unit')
        if metric == 'cost' and not manifest.get('cost'):
            continue
        identity = comparison_identity(manifest, experiment, metric, unit)
        artifact = artifact_identity(manifest, finish.get('artifacts', {}))
        row.update(comparison_identity=identity, artifact_identity=artifact,
                   verifier_identity=verifier_identity(manifest, experiment))
        if artifact is None:
            continue
        key = (identity, artifact)
        group = groups.setdefault(key, dict(comparison_identity=identity, artifact_identity=artifact,
            verifier_identity=row['verifier_identity'], metric=metric, unit=unit,
            policy=manifest.get('draw_policy'), manifest=manifest, observations=[], experiments=[]))
        group['experiments'].append(row['experiment'])
        value = evidence.get('value') if isinstance(evidence, dict) else None
        valid = (isinstance(evidence, dict) and evidence.get('evidence_kind') == 'verified' and
                 type(value) in (int, float) and math.isfinite(value))
        correction = invalidations.get(row['result'])
        no_verdict = row['status'] not in {'accepted_by_verifier', 'rejected_by_verifier'}
        group['observations'].append(dict(experiment=row['experiment'], result=row['result'],
            status=row['status'], accepted=row['independently_accepted'], verified=valid,
            value=value if valid else None, draw=_slot(manifest, experiment) if group['policy'] else None,
            disposition='invalidated_infrastructure' if correction else 'no_verdict' if no_verdict else 'observation',
            **({'invalidation': correction} if correction else {})))
        if not group['policy']:
            family = digest(verifier_contract(manifest, experiment, family=True))
            group['family'] = family
    aggregates = []
    baselines = {}
    references = {}
    for (identity, artifact), group in groups.items():
        observations = group['observations']
        policy = group['policy']
        reasons = []
        values = []
        expected = _expected(group['manifest']) if policy else []
        observed = {}
        for observation in observations:
            if observation['disposition'] != 'observation':
                continue
            observed.setdefault(canonical(observation['draw']), []).append(observation)
        missing = []
        if policy:
            expected_keys = {canonical(d) for d in expected}
            missing = [d for d in expected if canonical(d) not in observed]
            if missing:
                reasons.append('incomplete precommitted draw set')
            if set(observed) - expected_keys:
                reasons.append('observed draws outside precommitted set')
        elif len(planned_families[group['family']]) > 1:
            reasons.append('changing verifier settings or seeds without a declared draw policy')
        if not observed:
            reasons.append('no completed independent verdict; replacement verification required')
        for draws in observed.values():
            if any(not d['accepted'] or not d['verified'] for d in draws):
                reasons.append('one or more observations failed independent acceptance or objective verification')
            valid_values = {d['value'] for d in draws if d['verified']}
            if len(valid_values) > 1:
                reasons.append('conflicting repeated observations; declare independent draw replicates')
            if len(valid_values) == 1:
                values.append(next(iter(valid_values)))
        mode = policy['aggregate'] if policy else 'deterministic'
        eligible = not reasons and bool(values)
        value = None
        if eligible and mode != 'all':
            scale = max(abs(v) for v in values)
            value = ((math.fsum(v / scale for v in values) / len(values) * scale if scale else 0.) if mode == 'mean' else
                     (max(values) if direction == 'min' else min(values)) if mode == 'worst' else values[0])
        item = dict(comparison_identity=identity, artifact_identity=artifact,
            verifier_identity=group['verifier_identity'], metric=metric, unit=group['unit'],
            aggregate=mode, value=value, independently_accepted=eligible,
            comparison_eligible=eligible and value is not None,
            evidence_kind='verified' if eligible else 'incomplete_or_conflicting',
            reasons=sorted(set(reasons)), expected_draws=expected, missing_draws=missing,
            per_draw=[dict(draw=draw, value=next(iter({o['value'] for o in observed.get(canonical(draw), [])}))
                if observed.get(canonical(draw)) and all(o['accepted'] and o['verified'] for o in observed[canonical(draw)])
                and len({o['value'] for o in observed[canonical(draw)]}) == 1 else None) for draw in expected],
            observations=observations, experiments=sorted(set(group['experiments'])),
            scope='complete precommitted finite draw set; no population-expectation claim' if policy
                  else 'consistent independently verified observations of these artifact bytes')
        aggregates.append(item)
        if not item['comparison_eligible']:
            continue
        first = next(o for o in observations if o['disposition'] == 'observation')
        reference = references.setdefault(identity, dict(experiment=first['experiment'], result=first['result'],
            artifact_identity=artifact, value=value, unit=group['unit']))
        if identity not in baselines or ((value < baselines[identity]['value']) if direction == 'min' else value > baselines[identity]['value']):
            baselines[identity] = dict(experiment=first['experiment'], result=first['result'],
                results=[o['result'] for o in observations], comparison_identity=identity,
                artifact_identity=artifact, metric=metric, value=value, unit=group['unit'],
                evidence_kind='verified', aggregate=mode, verifier_identity=group['verifier_identity'],
                expected_draws=expected, reference=reference, delta_from_reference=value-reference['value'],
                scope='best complete independently verified artifact in this verifier/objective/draw comparison group',
                evidence_scope=item['scope'])
    return sorted(baselines.values(), key=lambda x: x['comparison_identity']), aggregates
