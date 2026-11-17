"""Frozen manifests, owned script attempts, and bounded local experiment sweeps.

Commands are argv arrays and substitution replaces entire arguments. A successful
process is execution evidence; acceptance requires the separately pinned verifier.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
import ast
import codecs
import csv
import fcntl
import hashlib
import io
import itertools
import json
import math
import os
import re
import shutil
import time
import tomllib
import uuid

from .model import StrataError, canonical, digest
from .store import Store
from .numeric_jobs import _runtime
from .cost import Formula

READER_VERSION = 'strata-output-readers/1'
_NAME = re.compile(r'^[A-Za-z_][A-Za-z0-9_.-]*$')
_PLACEHOLDER = re.compile(r'^\{(setting|input|dependency|artifact):([^{}]+)\}$')
_SHELLS = {'sh', 'bash', 'dash', 'zsh', 'fish', 'ksh', 'csh', 'pwsh', 'powershell'}
_SUCCESS = {'accepted_by_verifier', 'artifact_validated', 'execution_finished'}


def _finite(value, name, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or positive and value <= 0:
        raise StrataError(f'{name} must be finite' + (' and positive' if positive else ''))
    return value


def _relative(value):
    if not isinstance(value, str) or not value or '\\' in value:
        raise StrataError('artifact paths must be nonempty relative POSIX paths')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or value == '.' or any(p.startswith('.strata') for p in path.parts):
        raise StrataError('artifact path escapes the isolated working directory or uses reserved .strata paths')
    return str(path)


def _hash(path):
    with Path(path).open('rb') as stream:
        value = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
        return value.hexdigest()


def _files(path):
    path = Path(path)
    if path.is_symlink() or not path.exists():
        raise StrataError(f'declared file must exist and cannot be a symlink: {path}')
    if path.is_file():
        return {'': _hash(path)}
    if not path.is_dir():
        raise StrataError('only regular files and directories may be staged')
    result = {}
    for child in sorted(path.rglob('*')):
        if child.is_symlink() or not (child.is_file() or child.is_dir()):
            raise StrataError(f'dependency tree contains a symlink or special file: {child}')
        if child.is_file():
            result[str(child.relative_to(path))] = _hash(child)
    if not result:
        raise StrataError('declared dependency directory is empty')
    return result


def _sources(value, root, label):
    if isinstance(value, list):
        value = {str(i): path for i, path in enumerate(value)}
    if not isinstance(value, dict):
        raise StrataError(f'{label} must be a named path mapping or path list')
    result = {}
    for name, item in value.items():
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or name in {'.', '..'}:
            raise StrataError('input/dependency names must be simple path components')
        spec = item if isinstance(item, dict) else {'path': item}
        if not isinstance(spec.get('path'), str):
            raise StrataError('input/dependency requires a path')
        path = Path(spec['path'])
        path = path if path.is_absolute() else root / path
        # Check before resolve, which would hide a final symlink.
        if path.is_symlink():
            raise StrataError('declared inputs and dependencies cannot be symlinks')
        path = path.resolve()
        files = _files(path)
        if spec.get('sha256') and files != {'': spec['sha256']}:
            raise StrataError('declared dependency sha256 does not match')
        modes = {relative: ((path / relative) if path.is_dir() else path).stat().st_mode & 0o777
                 for relative in files}
        result[name] = dict(path=str(path), files=files, directory=path.is_dir(), modes=modes)
    return result


def _command(argv, root, sources, settings, outputs):
    if not isinstance(argv, list) or not argv or any(not isinstance(arg, str) or '\x00' in arg for arg in argv):
        raise StrataError('argv must be a nonempty argument array, never a shell string')
    result = list(argv)
    for i, arg in enumerate(result):
        match = _PLACEHOLDER.fullmatch(arg)
        if match:
            kind, name = match.groups()
            available = settings if kind == 'setting' else outputs if kind == 'artifact' else sources[kind]
            if name not in available:
                raise StrataError(f'unknown argv placeholder {arg}')
            continue
        if arg in {'{seed}', '{replicate}', '{workdir}', '{checkpoint}'}:
            continue
        if '{' in arg or '}' in arg:
            raise StrataError('placeholders must occupy an entire argument')
        if i == 0:
            executable = Path(arg) if Path(arg).is_absolute() else root / arg if '/' in arg else Path(shutil.which(arg) or '')
            if not executable.is_file():
                raise StrataError('command executable must exist')
            result[i] = str(executable.absolute())
        elif Path(arg).is_absolute() and Path(arg).is_file():
            # Relative strings are command arguments, often output filenames.
            # Their meaning cannot depend on files beside the manifest.
            source = Path(arg).resolve()
            found = None
            for kind, mapping in sources.items():
                for name, spec in mapping.items():
                    path = Path(spec['path'])
                    if source == path:
                        found = '{' + kind + ':' + name + '}'
                    elif spec['directory'] and source.is_relative_to(path):
                        # Retain a frozen source argument, replaced by its staged path at launch.
                        found = str(source)
            if found is None:
                raise StrataError(f'command file argument must be a declared input or dependency: {arg}')
            result[i] = found
    if result[0].startswith('{'):
        match = _PLACEHOLDER.fullmatch(result[0])
        if not match or match[1] != 'dependency':
            raise StrataError('executable placeholders must name a dependency')
        executable = Path(sources['dependency'][match[2]]['path'])
        if not executable.is_file():
            raise StrataError('executable dependency must be a file')
    else:
        executable = Path(result[0])
    if executable.name in _SHELLS or executable.resolve().name in _SHELLS:
        shell = executable.resolve().name if executable.resolve().name in _SHELLS else executable.name
        script = None
        for arg in result[1:]:
            if shell in {'pwsh', 'powershell'}:
                option = arg.lower() in {'-noprofile', '-noninteractive', '-file'}
            else:
                option = arg in {'--', '--noprofile', '--norc', '--posix', '--no-config'} or re.fullmatch(r'-[euxnvf]+', arg)
            if option:
                continue
            script = _PLACEHOLDER.fullmatch(arg)
            break
        if not script or script[1] != 'dependency' or sources['dependency'][script[2]]['directory']:
            raise StrataError('shell argv requires a pinned dependency script; shell command strings and -c are unsupported')
    return result


def _outputs(value):
    if isinstance(value, dict):
        value = [dict(spec, name=name) if isinstance(spec, dict) else dict(name=name, path=spec) for name, spec in value.items()]
    if not isinstance(value, list):
        raise StrataError('outputs must be a list or named mapping')
    result = []
    for item in value:
        if not isinstance(item, dict) or not _NAME.fullmatch(item.get('name', '')):
            raise StrataError('every output needs a simple name')
        spec = dict(item)
        spec['path'] = _relative(spec.get('path'))
        spec.setdefault('type', 'binary')
        if spec['type'] not in {'json', 'jsonl', 'csv', 'text', 'binary'}:
            raise StrataError('unsupported output type')
        if 'nonempty' in spec and type(spec['nonempty']) is not bool:
            raise StrataError('output nonempty must be boolean')
        spec.setdefault('nonempty', spec['type'] not in {'text', 'jsonl'})
        if 'schema' in spec and not isinstance(spec['schema'], dict):
            raise StrataError('output schema must be a JSON-schema subset object')
        if type(spec.get('max_parse_bytes', 16 * 1024**2)) is not int or spec.get('max_parse_bytes', 16 * 1024**2) <= 0:
            raise StrataError('max_parse_bytes must be a positive integer')
        result.append(spec)
    if len({s['name'] for s in result}) != len(result) or len({s['path'] for s in result}) != len(result):
        raise StrataError('output names and paths must be unique')
    return result


def load_manifest(path):
    """Read JSON/TOML, validate commands, and freeze inputs/runtime/reader identities."""
    path = Path(path).resolve()
    try:
        raw = tomllib.loads(path.read_text()) if path.suffix.lower() == '.toml' else json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise StrataError(f'cannot read experiment manifest: {exc}') from exc
    if not isinstance(raw, dict) or raw.get('version', 1) != 1:
        raise StrataError('experiment manifest version must be 1')
    spec = json.loads(canonical(raw))
    spec['contract_explicit'] = 'contract' in raw
    root = (path.parent / spec.get('working_directory', '.')).resolve()
    if not root.is_dir():
        raise StrataError('manifest working_directory must exist')
    settings = spec.setdefault('settings', {})
    if not isinstance(settings, dict) or any(not isinstance(k, str) or not _NAME.fullmatch(k) or type(v) not in (str, int, float, bool) or type(v) is float and not math.isfinite(v) for k, v in settings.items()):
        raise StrataError('settings must map simple names to finite scalar values')
    spec.update(version=1, source_manifest=str(path), source_root=str(root))
    spec['dependencies'] = _sources(spec.get('dependencies', {}), root, 'dependencies')
    spec['inputs'] = _sources(spec.get('inputs', {}), root, 'inputs')
    sources = {'dependency': spec['dependencies'], 'input': spec['inputs']}
    spec['outputs'] = _outputs(spec.get('outputs', []))
    stages = spec.get('stages')
    if stages is None:
        stages = [dict(name='run', argv=spec.get('argv'), outputs=spec['outputs'])]
        for key in ('checkpoint', 'resume_argv'):
            if key in spec:
                stages[0][key] = spec[key]
    if not isinstance(stages, list) or not stages:
        raise StrataError('manifest needs argv or a nonempty stages list')
    all_outputs = {o['name'] for o in spec['outputs']}
    for stage in stages:
        if not isinstance(stage, dict):
            raise StrataError('stages must be objects')
        all_outputs.update(o['name'] for o in _outputs(stage.get('outputs', [])))
    runtimes = {}
    for i, stage in enumerate(stages):
        stage.setdefault('name', f'stage-{i + 1}')
        if not _NAME.fullmatch(stage['name']):
            raise StrataError('stage name must be a simple name')
        stage['outputs'] = _outputs(stage.get('outputs', spec['outputs'] if i == len(stages) - 1 else []))
        stage['argv'] = _command(stage.get('argv'), root, sources, settings, all_outputs)
        if stage.get('resume_argv'):
            if not stage.get('checkpoint'):
                raise StrataError('resume_argv requires an explicit checkpoint')
            stage['resume_argv'] = _command(stage['resume_argv'], root, sources, settings, all_outputs)
        if stage.get('checkpoint'):
            cp = stage['checkpoint']
            cp = {'path': cp, 'type': 'binary'} if isinstance(cp, str) else dict(cp)
            cp['name'] = 'checkpoint'
            stage['checkpoint'] = _outputs([cp])[0]
            if cp.get('commit_marker'):
                stage['checkpoint']['commit_marker'] = _relative(cp['commit_marker'])
        for command in [stage['argv'], stage.get('resume_argv')]:
            if command:
                match = _PLACEHOLDER.fullmatch(command[0])
                exe = sources['dependency'][match[2]]['path'] if match else command[0]
                runtimes[exe] = _runtime(exe)
    if len({stage['name'] for stage in stages}) != len(stages):
        raise StrataError('stage names must be unique')
    spec['stages'] = stages
    metrics = spec.setdefault('metrics', {})
    if not isinstance(metrics, dict):
        raise StrataError('metrics must be a named mapping')
    for name, metric in metrics.items():
        if name in settings:
            raise StrataError('metric names cannot shadow frozen settings used by cost formulas')
        if not _NAME.fullmatch(name) or not isinstance(metric, dict) or metric.get('output') not in all_outputs:
            raise StrataError('metric must name a declared output')
        if not isinstance(metric.get('unit'), str) or not metric['unit'].strip():
            raise StrataError('metric needs an explicit unit')
        metric.setdefault('evidence_kind', 'measured')
        if metric['evidence_kind'] not in {'measured', 'model', 'bound'}:
            raise StrataError('script metrics may declare measured, model or bound; verification is separate')
        if not isinstance(metric.get('selector', ''), (str, list)):
            raise StrataError('metric selector must be a dotted path, JSON pointer or path list')
    from .sandbox import effective_limits
    limit_keys = {'timeout', 'max_output', 'memory_bytes', 'max_file_bytes', 'cpu_seconds'}
    def limits(overrides, defaults=None):
        if not isinstance(overrides, dict) or set(overrides) - limit_keys:
            raise StrataError('unsupported command limit; use timeout, max_output, memory_bytes, max_file_bytes or cpu_seconds')
        return effective_limits(**{**(defaults or {}), **overrides})
    declared_limits = spec.get('limits', {})
    spec['limits'] = limits(declared_limits)
    for stage in stages:
        stage['limits'] = limits(stage.get('limits', {}), declared_limits)
    resources = spec.setdefault('resources', {})
    explicit_memory = 'memory_bytes' in resources
    stage_memory = max(stage['limits']['memory_bytes'] for stage in stages)
    for key, default in (('threads', 1), ('memory_bytes', stage_memory), ('storage_bytes', max(stage['limits']['max_file_bytes'] for stage in stages))):
        resources.setdefault(key, default)
        if type(resources[key]) is not int or resources[key] < (1 if key == 'threads' else 0):
            raise StrataError('resource reservations must be nonnegative integers, with at least one thread')
    if set(resources) - {'threads', 'memory_bytes', 'storage_bytes'}:
        raise StrataError('unsupported resource reservation')
    if stage_memory > resources['memory_bytes']:
        raise StrataError('stage memory limit exceeds the declared aggregate memory reservation')
    spec.setdefault('enforcement', 'cooperative')
    if spec['enforcement'] not in {'cooperative', 'hard'}:
        raise StrataError('enforcement must explicitly be cooperative or hard')
    if spec['enforcement'] == 'hard':
        raise StrataError('hard aggregate enforcement is unavailable; explicitly select cooperative reservations')
    environment = spec.setdefault('environment', {})
    if not isinstance(environment, dict) or any(not isinstance(k, str) or not re.fullmatch('[A-Za-z_][A-Za-z0-9_]*', k) or not isinstance(v, str) or '\x00' in v for k, v in environment.items()):
        raise StrataError('environment must contain explicit nonsecret string values')
    spec['environment'] = {'PATH': os.defpath, 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8', 'OMP_NUM_THREADS': str(resources['threads']), 'OPENBLAS_NUM_THREADS': str(resources['threads']), 'MKL_NUM_THREADS': str(resources['threads']), **environment}
    from .sandbox import declared_environment
    spec['environment'] = declared_environment(spec['environment'])
    verifier = spec.get('verifier')
    if verifier:
        if not isinstance(verifier, dict) or not isinstance(verifier.get('version'), str) or not verifier['version']:
            raise StrataError('independent verifier needs an explicit version')
        verifier['dependencies'] = _sources(verifier.get('dependencies', {}), root, 'verifier dependencies')
        if any(spec['dependencies'][key] != verifier['dependencies'][key] for key in set(spec['dependencies']) & set(verifier['dependencies'])):
            raise StrataError('verifier cannot redefine a worker dependency name')
        vsources = {'dependency': {**spec['dependencies'], **verifier['dependencies']}, 'input': spec['inputs']}
        verifier['argv'] = _command(verifier.get('argv'), root, vsources, settings, all_outputs)
        if not any(arg.startswith('{artifact:') for arg in verifier['argv']):
            raise StrataError('independent verifier must receive a declared {artifact:name}')
        match = _PLACEHOLDER.fullmatch(verifier['argv'][0])
        exe = vsources['dependency'][match[2]]['path'] if match else verifier['argv'][0]
        runtimes[exe] = _runtime(exe)
        if verifier.get('sha256') and verifier['sha256'] != runtimes[exe]['executable_sha256']:
            raise StrataError('verifier executable sha256 does not match')
        verifier.setdefault('source', verifier['dependencies'] or verifier['argv'][0])
        verifier.setdefault('accepted_selector', 'accepted')
        if not isinstance(verifier.get('metrics', {}), dict):
            raise StrataError('verifier metrics must be a named mapping')
        for name, binding in verifier.get('metrics', {}).items():
            if not _NAME.fullmatch(name) or name in settings:
                raise StrataError('verifier metric names must be simple names and cannot shadow settings')
            if isinstance(binding, dict):
                if set(binding) - {'selector', 'unit'} or 'selector' not in binding:
                    raise StrataError('verifier metric binding requires selector and unit')
                if not isinstance(binding.get('unit'), str) or not binding['unit'].strip():
                    raise StrataError('verifier metric needs an explicit unit')
                if name in metrics and binding['unit'] != metrics[name]['unit']:
                    raise StrataError('verifier metric unit must match the worker metric unit')
                selector = binding['selector']
            else:
                if name not in metrics:
                    raise StrataError('verifier-only metric requires a selector and explicit unit')
                selector = binding
            if not isinstance(selector, (str, list)):
                raise StrataError('verifier metric selector must be a dotted path, JSON pointer or path list')
        verifier['limits'] = limits(verifier.get('limits', {}), declared_limits)
        if not explicit_memory:
            resources['memory_bytes'] = max(resources['memory_bytes'], verifier['limits']['memory_bytes'])
        elif verifier['limits']['memory_bytes'] > resources['memory_bytes']:
            raise StrataError('verifier memory limit exceeds the declared aggregate memory reservation')
    spec['runtimes'] = runtimes
    spec.setdefault('max_attempts', 3)
    spec.setdefault('retention', 'compact')
    if spec['retention'] not in {'compact', 'keep'}:
        raise StrataError('retention must be compact or keep')
    if type(spec['max_attempts']) is not int or spec['max_attempts'] < 1:
        raise StrataError('max_attempts must be positive')
    spec.setdefault('max_wall_seconds', (sum(stage['limits']['timeout'] for stage in stages) + (verifier['limits']['timeout'] if verifier else 0)) * spec['max_attempts'])
    _finite(spec['max_wall_seconds'], 'max_wall_seconds', True)
    if spec.get('cost'):
        if not isinstance(spec['cost'], dict) or not spec['cost'].get('unit'):
            raise StrataError('cost requires expression and unit')
        Formula(spec['cost'].get('expression'), functions={'ceil': math.ceil, 'floor': math.floor})
        if not isinstance(spec['cost'].get('resources', {}), dict):
            raise StrataError('cost resources must map formula variables to independent verifier selectors')
    if spec.get('decisive'):
        decisive = spec['decisive']
        required = {'question', 'frozen', 'relaxations', 'mapping', 'threshold', 'interpretation'}
        if not isinstance(decisive, dict) or not required <= set(decisive) or not isinstance(decisive['mapping'], dict) or not isinstance(decisive['relaxations'], dict) or not isinstance(decisive['frozen'], dict):
            raise StrataError('decisive experiment needs question, frozen, relaxations, mapping, threshold and interpretation')
        if set(decisive['relaxations']) != set(decisive['mapping']) or any(not v for v in decisive['mapping'].values()):
            raise StrataError('each relaxation needs an explicit mapping to the real contract')
        decisive['evidence_scope'] = 'relaxed success motivates a full candidate; incomplete optimization failure is not an impossibility proof'
    for key in ('screening', 'continuation'):
        if spec.get(key):
            policy = spec[key]
            if not isinstance(policy, dict) or not {'policy', 'identity', 'metric'} <= set(policy) or policy['metric'] not in metrics:
                raise StrataError(f'{key} requires a policy, identity and declared metric')
            if key == 'screening' and policy.get('after_stage') not in {s['name'] for s in stages}:
                raise StrataError('screening must identify the cheap stage by after_stage')
            if key == 'screening':
                full_stage = policy.get('full_stage', stages[-1]['name'])
                full_metric = policy.get('full_metric', policy['metric'])
                stage_names = [s['name'] for s in stages]
                if full_stage not in stage_names or full_metric not in metrics or stage_names.index(full_stage) <= stage_names.index(policy['after_stage']):
                    raise StrataError('screening requires a later full_stage and a declared full_metric')
                full_outputs = {o['name'] for o in stages[stage_names.index(full_stage)]['outputs']}
                if metrics[full_metric]['output'] not in full_outputs:
                    raise StrataError('full_stage does not produce the declared full_metric')
            if key == 'continuation' and not any(s.get('checkpoint') and metrics[policy['metric']]['output'] in {o['name'] for o in s['outputs']} for s in stages):
                raise StrataError('continuation policy needs a checkpoint stage producing its measured metric')
    from .experiment_comparisons import validate_draw_policy
    validate_draw_policy(spec)
    spec['reader'] = dict(version=READER_VERSION, source_sha256=_hash(__file__))
    spec.setdefault('name', path.stem)
    spec.setdefault('contract', spec['name'])
    canonical(spec)
    return spec


def _staged_runtime_paths(spec):
    dependencies = {**spec['dependencies'], **(spec.get('verifier') or {}).get('dependencies', {})}
    commands = [stage[key] for stage in spec['stages'] for key in ('argv', 'resume_argv') if stage.get(key)]
    if spec.get('verifier'):
        commands.append(spec['verifier']['argv'])
    paths, live = set(), set()
    for command in commands:
        executable = command[0]
        match = _PLACEHOLDER.fullmatch(executable)
        if match and match[1] == 'dependency':
            paths.add(dependencies[match[2]]['path'])
        elif Path(executable).is_absolute() and any(item['directory'] and Path(executable).is_relative_to(item['path'])
                                                    for item in dependencies.values()):
            paths.add(executable)
        else:
            live.add(executable)
    return paths - live


def _assert_identity(spec, *, staged=False):
    groups = (spec['inputs'], spec['dependencies'], (spec.get('verifier') or {}).get('dependencies', {}))
    if not staged:
        for group in groups:
            for item in group.values():
                if _files(item['path']) != item['files']:
                    raise StrataError('dependency drift blocks resume; create a descendant experiment with new dependencies')
    staged_executables = _staged_runtime_paths(spec)
    if any(_runtime(path) != identity for path, identity in spec['runtimes'].items()
           if not staged or path not in staged_executables):
        raise StrataError('runtime drift blocks resume')
    if spec['reader'] != dict(version=READER_VERSION, source_sha256=_hash(__file__)):
        raise StrataError('output reader drift blocks resume')


def _freeze_files(store, spec):
    for group in (spec['inputs'], spec['dependencies'], (spec.get('verifier') or {}).get('dependencies', {})):
        for item in group.values():
            for relative, expected in item['files'].items():
                source = Path(item['path']) / relative if item['directory'] else Path(item['path'])
                if store.blob_file(source) != expected:
                    raise StrataError('dependency changed during experiment creation')


def create_experiment(store, manifest_path, settings=None, seed=None, replicate=0, *, parent=None):
    """Create a configuration identity; retries retain it, explicit replicates differ."""
    spec = load_manifest(manifest_path)
    return _create_frozen(store, spec, settings, seed, replicate, parent=parent)


def _create_frozen(store, spec, settings, seed, replicate, parent=None, sources_frozen=False):
    settings = {**spec['settings'], **(settings or {})}
    if set(settings) != set(spec['settings']) or any(type(v) not in (str, int, float, bool) for v in settings.values()):
        raise StrataError('settings overrides must name declared scalar settings')
    if any(settings.get(key) != value for key, value in (spec.get('decisive') or {}).get('frozen', {}).items()):
        raise StrataError('decisive experiment overrides a declared frozen setting')
    if seed is not None and type(seed) is not int or type(replicate) is not int or replicate < 0:
        raise StrataError('seed must be an explicit integer and replicate a nonnegative integer')
    if seed is None and (any('{seed}' in s['argv'] or '{seed}' in s.get('resume_argv', []) for s in spec['stages']) or
                         '{seed}' in (spec.get('verifier') or {}).get('argv', [])):
        raise StrataError('this manifest requires an explicit seed')
    if parent:
        store.get(parent, 'experiment')
    canonical(settings)
    if not sources_frozen:
        _assert_identity(spec)
        _freeze_files(store, spec)
    identity = digest(dict(manifest=spec, settings=settings, seed=seed, replicate=replicate, parent=parent))
    # Lock the lookup and insert together: simultaneous creators must not mint
    # distinct timestamp-bearing records for the same configuration identity.
    store.db.execute("CREATE INDEX IF NOT EXISTS records_identity ON records(kind,json_extract(body,'$.identity')) WHERE kind='experiment'")
    store.db.execute('BEGIN IMMEDIATE')
    try:
        row = store.db.execute("SELECT id FROM records WHERE kind='experiment' AND json_extract(body,'$.identity')=?", (identity,)).fetchone()
        if row:
            store.get(row[0], 'experiment')
            store.db.commit()
            return row[0]
        body = dict(version=1, identity=identity, manifest=spec, settings=settings, seed=seed, replicate=replicate, parent=parent, created_at=time.time())
        handle = digest(dict(kind='experiment', body=body))
        store.db.execute('INSERT INTO records(id,kind,body) VALUES (?,?,?)', (handle, 'experiment', canonical(body)))
        store.db.commit()
        return handle
    except BaseException:
        store.db.rollback()
        raise


def _records(store, kind, experiment):
    store.db.execute("CREATE INDEX IF NOT EXISTS records_experiment ON records(kind,json_extract(body,'$.experiment'))")
    rows = store.db.execute("SELECT id FROM records WHERE kind=? AND json_extract(body,'$.experiment')=? ORDER BY rowid", (kind, experiment)).fetchall()
    return [(handle, store.get(handle, kind)) for (handle,) in rows]


def experiment_status(store, handle):
    experiment = store.get(handle, 'experiment')
    starts = _records(store, 'experiment_attempt', handle)
    finishes = _records(store, 'experiment_finish', handle)
    completed = {row['attempt']: row for _, row in finishes}
    charges = _records(store, 'experiment_charge', handle)
    settled = {row['charge']: row['seconds'] for _, row in _records(store, 'experiment_charge_finish', handle)}
    spent = sum(completed[row['attempt']]['seconds'] if row['attempt'] in completed else _attempt_charge(row, charges, settled) for _, row in starts)
    latest = dict(handle=finishes[-1][0], **finishes[-1][1]) if finishes else None
    checkpoints = _records(store, 'experiment_checkpoint', handle)
    stages = _records(store, 'experiment_stage', handle)
    state = latest['status'] if latest else 'planned'
    if not starts:
        gates = _records(store, 'experiment_gate', handle)
        if gates and gates[-1][1]['decision'].get('action') == 'prune':
            state = 'screened_out'
    if any(row['attempt'] not in completed for _, row in starts):
        state = 'reconciliation_required'
    elif state not in _SUCCESS and spent >= experiment['manifest']['max_wall_seconds']:
        state = 'budget_exhausted'
    elif state not in _SUCCESS and len(starts) >= experiment['manifest']['max_attempts']:
        state = 'attempt_budget_exhausted'
    return dict(experiment=handle, identity=experiment['identity'], status=state, attempts=len(starts), attempt_budget_exhausted=len(starts) >= experiment['manifest']['max_attempts'], wall_budget_exhausted=spent >= experiment['manifest']['max_wall_seconds'], completed_stages=len({r['index'] for _, r in stages}), spent_wall_seconds=spent, remaining_wall_seconds=max(0, experiment['manifest']['max_wall_seconds'] - spent), latest=latest, checkpoint=dict(handle=checkpoints[-1][0], **checkpoints[-1][1]) if checkpoints else None, accounting='unfinished stages conservatively charge their reserved timeout and require board reconciliation')


def _attempt_charge(attempt, charges, settled):
    rows = [(handle, row) for handle, row in charges if row['attempt'] == attempt['attempt']]
    return sum(settled.get(h, row['reserved_seconds']) for h, row in rows) if rows else attempt['reserved_seconds']


def _reconcile_lost(store, handle, board):
    finished = {row['attempt'] for _, row in _records(store, 'experiment_finish', handle)}
    jobs = {row['job_id']: row for row in board.snapshot()['jobs']}
    charges = _records(store, 'experiment_charge', handle)
    settled = {row['charge']: row['seconds'] for _, row in _records(store, 'experiment_charge_finish', handle)}
    for start_handle, row in _records(store, 'experiment_attempt', handle):
        if row['attempt'] in finished or jobs.get(row['job'], {}).get('state') != 'lost':
            continue
        store.put('experiment_finish', dict(experiment=handle, attempt=row['attempt'], start_record=start_handle, status='failed', seconds=_attempt_charge(row, charges, settled), metrics={}, artifacts={}, stages=[], checkpoint=None, verifier=None, cost=None, verified_cost=None, policy_decisions=[], error=dict(kind='reconciled_lost_supervisor', job=row['job']), finished_at=time.time()))


def _run_charged(store, handle, attempt, operation, board, token, argv, work, **limits):
    from .supervision import run_owned
    charge = store.put('experiment_charge', dict(experiment=handle, attempt=attempt, operation=operation, reserved_seconds=limits['timeout'], started_at=time.time()))
    receipt = run_owned(board, token, argv, work, **limits)
    store.put('experiment_charge_finish', dict(experiment=handle, attempt=attempt, charge=charge, seconds=receipt['seconds']))
    return receipt


@contextmanager
def _lock(store, handle):
    with (store.path / f'experiment-{handle}.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise StrataError('experiment already has an active coordinator') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _stage_sources(store, work, spec, verifier=False):
    result = {'dependency': {}, 'input': {}}
    if verifier:
        from .experiment_comparisons import verifier_sources
        sources = verifier_sources(spec)
    else:
        sources = {'dependency': spec['dependencies'], 'input': spec['inputs']}
    for kind, mapping in sources.items():
        for name, item in mapping.items():
            # Keep source names and sibling module layout without collisions
            # between equally named files from unrelated source directories.
            source_path = Path(item['path'])
            target = (work / '.strata' / kind / 'directories' / name if item['directory']
                      else work / '.strata' / kind / 'files' / digest(str(source_path.parent))[:20] / source_path.name)
            for relative, blob in item['files'].items():
                destination = target / relative if item['directory'] else target
                store.copy_blob(blob, destination)
                # Modes are frozen along with bytes; ongoing attempts never need
                # the live source again. The fallback supports older manifests.
                mode = item.get('modes', {}).get(relative, 0o755)
                destination.chmod(mode)
            result[kind][name] = str(target)
    return result


def _verify_staged(spec, sources, *, verifier=False, artifacts=None):
    if verifier:
        from .experiment_comparisons import verifier_sources
        declarations = verifier_sources(spec)
    else:
        declarations = {'dependency': spec['dependencies'], 'input': spec['inputs']}
    for kind, mapping in declarations.items():
        for name, item in mapping.items():
            if _files(sources[kind][name]) != item['files']:
                raise StrataError('command modified a staged input or pinned dependency')
    for path, expected in (artifacts or {}).items():
        if Path(path).is_symlink() or _hash(path) != expected:
            raise StrataError('command modified a previously validated artifact')


def _substitute(argv, experiment, work, sources, artifacts, checkpoint):
    values = {'{seed}': experiment['seed'], '{replicate}': experiment['replicate'], '{workdir}': str(work), '{checkpoint}': checkpoint}
    spec = experiment['manifest']
    output = []
    for arg in argv:
        match = _PLACEHOLDER.fullmatch(arg)
        if match:
            kind, name = match.groups()
            mapping = experiment['settings'] if kind == 'setting' else artifacts if kind == 'artifact' else sources[kind]
            if name not in mapping:
                raise StrataError(f'argument references an unavailable {kind}: {name}')
            value = mapping[name]
        elif arg in values:
            value = values[arg]
            if value is None:
                raise StrataError(f'argument {arg} has no available value')
        else:
            value = arg
            for kind in ('input', 'dependency'):
                mapping = spec['inputs'] if kind == 'input' else {**spec['dependencies'], **((spec.get('verifier') or {}).get('dependencies', {}))}
                for name, item in mapping.items():
                    source = Path(item['path'])
                    if item['directory'] and Path(arg).is_absolute() and Path(arg).is_relative_to(source) and name in sources[kind]:
                        value = str(Path(sources[kind][name]) / Path(arg).relative_to(source))
        output.append(str(value).lower() if type(value) is bool else str(value))
    return output


def _schema(value, schema, where='output'):
    supported = {'type', 'required', 'properties', 'items', 'minItems', 'maxItems', 'minimum', 'maximum', 'enum', 'additionalProperties'}
    if set(schema) - supported:
        raise StrataError('unsupported output schema keyword: ' + ', '.join(sorted(set(schema) - supported)))
    types = {'object': lambda v: isinstance(v, dict), 'array': lambda v: isinstance(v, list), 'string': lambda v: isinstance(v, str), 'number': lambda v: type(v) in (int, float) and math.isfinite(v), 'integer': lambda v: type(v) is int, 'boolean': lambda v: type(v) is bool, 'null': lambda v: v is None}
    if 'type' in schema and (schema['type'] not in types or not types[schema['type']](value)):
        raise StrataError(f'{where} does not match schema type')
    if 'enum' in schema and value not in schema['enum']:
        raise StrataError(f'{where} not in schema enum')
    if isinstance(value, dict):
        if not set(schema.get('required', [])) <= set(value):
            raise StrataError(f'{where} missing required schema fields')
        properties = schema.get('properties', {})
        if schema.get('additionalProperties') is False and set(value) - set(properties):
            raise StrataError(f'{where} contains undeclared fields')
        for key in set(value) & set(properties):
            _schema(value[key], properties[key], f'{where}.{key}')
    if isinstance(value, list):
        if len(value) < schema.get('minItems', 0) or len(value) > schema.get('maxItems', math.inf):
            raise StrataError(f'{where} has invalid array length')
        for child in value:
            _schema(child, schema.get('items', {}), where + '[]')
    if type(value) in (int, float) and (not math.isfinite(value) or value < schema.get('minimum', -math.inf) or value > schema.get('maximum', math.inf)):
        raise StrataError(f'{where} outside finite schema bounds')


def _decode(data, kind):
    def reject_constant(value):
        raise ValueError('nonfinite JSON number: ' + value)
    try:
        if kind == 'json':
            return json.loads(data, parse_constant=reject_constant)
        if kind == 'jsonl':
            return [json.loads(line, parse_constant=reject_constant) for line in data.decode().splitlines() if line.strip()]
        if kind == 'csv':
            reader = csv.DictReader(io.StringIO(data.decode()))
            fields = reader.fieldnames
            if not fields or len(set(fields)) != len(fields):
                raise StrataError('CSV needs a unique header row')
            rows = list(reader)
            if any(None in row or any(v is None for v in row.values()) for row in rows):
                raise StrataError('CSV rows do not match the header')
            return rows
        if kind == 'text':
            return data.decode()
        return data
    except (ValueError, UnicodeError, csv.Error) as exc:
        raise StrataError(f'invalid {kind} output: {exc}') from exc


def _selector(value, selector):
    if isinstance(selector, str):
        parts = [x.replace('~1', '/').replace('~0', '~') for x in selector[1:].split('/')] if selector.startswith('/') else selector.split('.') if selector else []
    else:
        parts = selector
    try:
        for key in parts:
            if isinstance(value, list):
                value = value[-1 if key == 'last' else int(key)]
            else:
                value = value[str(key)]
        return value
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        raise StrataError(f'output selector does not exist: {selector}') from exc


def _safe_output(work, relative):
    path = work / _relative(relative)
    if any(parent.is_symlink() for parent in [path, *path.parents] if parent != work.parent and parent.is_relative_to(work)):
        raise StrataError('output is a symlink or traverses a symlink')
    if not path.resolve().is_relative_to(work.resolve()) or not path.is_file() or path.stat().st_nlink != 1:
        raise StrataError('expected output is missing, escaped, nonregular or hard-linked')
    return path


def _collect(store, work, declarations, decode_names=()):
    artifacts, decoded = {}, {}
    for spec in declarations:
        path = _safe_output(work, spec['path'])
        size = path.stat().st_size
        if not size and spec.get('nonempty', True):
            raise StrataError(f'expected output is empty: {spec["name"]}')
        needs_decode = spec['type'] in {'json', 'jsonl', 'csv'} or spec.get('schema') or spec['name'] in decode_names
        if needs_decode and size > spec.get('max_parse_bytes', 16 * 1024**2):
            raise StrataError('structured output exceeds its explicit parsing byte budget')
        data = path.read_bytes() if needs_decode else None
        if spec['type'] == 'text' and data is None:
            decoder = codecs.getincrementaldecoder('utf-8')()
            try:
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024**2), b''):
                        decoder.decode(block)
                    decoder.decode(b'', final=True)
            except UnicodeError as exc:
                raise StrataError('text output is not UTF-8') from exc
        value = _decode(data, spec['type']) if data is not None else None
        if spec.get('schema'):
            _schema(value, spec['schema'], spec['name'])
        blob = store.blob_file(path)
        if data is not None and hashlib.sha256(data).hexdigest() != blob:
            raise StrataError('output changed while validating its schema')
        artifacts[spec['name']] = dict(sha256=blob, path=spec['path'], type=spec['type'], bytes=size, schema=spec.get('schema'), validation='fresh_isolated_output_schema_checked')
        decoded[spec['name']] = value
    return artifacts, decoded


def _metrics(spec, decoded):
    result = {}
    for name, metric in spec['metrics'].items():
        if metric['output'] not in decoded:
            continue
        value = _selector(decoded[metric['output']], metric.get('selector', ''))
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError as exc:
                raise StrataError('metric selector must contain a finite number') from exc
        _finite(value, f'metric {name}')
        result[name] = dict(value=value, unit=metric['unit'], evidence_kind=metric['evidence_kind'], source=dict(output=metric['output'], selector=metric.get('selector', '')))
    return result


def _restore_artifacts(store, work, artifacts):
    restored = {}
    for name, item in artifacts.items():
        target = work / '.strata' / 'artifacts' / name / PurePosixPath(item['path']).name
        store.copy_blob(item['sha256'], target)
        restored[name] = str(target)
    return restored


def _compact_attempt(store, attempt_dir, experiment, attempt, policy):
    """Drop only private attempt copies whose identical bytes remain in CAS.

    Never hard-link a writable worker output into the object store. Incomplete,
    changed and unvalidated outputs stay available for diagnosis.
    """
    result = dict(policy=policy, removed_files=0, removed_logical_bytes=0, errors=[])
    if policy == 'keep':
        return result
    candidates = {}
    for work in attempt_dir.iterdir():
        if not work.is_dir() or work.is_symlink():
            continue
        private = work / '.strata'
        if private.is_dir() and not private.is_symlink():
            for path in private.rglob('*'):
                if path.is_file() and not path.is_symlink():
                    candidates[path] = None
    for _, row in _records(store, 'experiment_stage', experiment):
        if row['attempt'] != attempt:
            continue
        work = attempt_dir / f'stage-{row["index"]:04d}'
        for item in row['artifacts'].values():
            candidates[work / item['path']] = item['sha256']
    for _, row in _records(store, 'experiment_checkpoint', experiment):
        if row['attempt'] == attempt:
            item = row['artifact']
            candidates[attempt_dir / f'stage-{row["index"]:04d}' / item['path']] = item['sha256']
    for path, expected in candidates.items():
        try:
            if (not path.is_file() or any(p.is_symlink() for p in [path, *path.parents])
                    or not path.resolve().is_relative_to(attempt_dir.resolve())):
                continue
            actual = _hash(path)
            if expected is not None and expected != actual:
                continue
            if not store._blob_path(actual).is_file():
                continue
            store.verify_blob(actual)
            size = path.stat().st_size
            path.unlink()
            result['removed_files'] += 1
            result['removed_logical_bytes'] += size
        except (OSError, StrataError) as exc:
            result['errors'].append(dict(path=str(path.relative_to(attempt_dir)), error=str(exc)))
    return result


def _save_logs(store, receipt):
    receipt = dict(receipt)
    receipt['logs'] = {key: store.blob(str(receipt.get(key, '')).encode()) for key in ('stdout', 'stderr', 'output')}
    return receipt


def _readonly_runtimes(spec):
    paths = set()
    staged = _staged_runtime_paths(spec)
    for executable in spec['runtimes']:
        if executable in staged:
            continue
        invocation = Path(executable)
        root = invocation.parent.parent
        # bwrap cannot mount onto a symlink. Preserve the invoked interpreter
        # name (important for virtualenvs), but bind its containing directory
        # when that name is a link, and bind its real executable separately.
        runtime_path = root if (root / 'pyvenv.cfg').is_file() else invocation.parent.resolve() if invocation.is_symlink() else invocation
        paths.add(str(runtime_path))
        paths.add(str(invocation.resolve()))
    # Bubblewrap hides /tmp; preserve explicitly declared interpreters/venvs.
    return sorted(paths)


def _checkpoint_identity(checkpoint):
    if checkpoint is None:
        return None
    return dict(sha256=checkpoint['artifact']['sha256'], index=checkpoint['index'], completed=checkpoint['completed'])


def _committed_checkpoint(store, experiment, attempt, index, stage, work, *, completed, input_artifacts, input_checkpoint):
    declaration = stage.get('checkpoint')
    if not declaration:
        return None
    if not declaration.get('commit_marker') and not completed:
        # A failed process may have left a truncated checkpoint: only a declared
        # commit marker allows recovery of its checkpoint before stage completion.
        return None
    artifacts, _ = _collect(store, work, [declaration])
    checkpoint = artifacts['checkpoint']
    if declaration.get('commit_marker'):
        marker = _safe_output(work, declaration['commit_marker']).read_text().strip()
        if marker != checkpoint['sha256']:
            raise StrataError('checkpoint commit marker must contain the exact current checkpoint sha256')
    body = dict(experiment=experiment, attempt=attempt, index=index, completed=completed, artifact=checkpoint, input_artifacts=input_artifacts, input_checkpoint=input_checkpoint, created_at=time.time())
    handle = store.put('experiment_checkpoint', body)
    return dict(handle=handle, **body)


def _resumable_stages(store, handle, spec):
    rows = _records(store, 'experiment_stage', handle)
    complete, upstream, checkpoint = {}, {}, None
    for index in range(len(spec['stages'])):
        choices = [(h, r) for h, r in reversed(rows) if r['index'] == index]
        if not choices:
            break
        # Never fall back across a newer restart: a new stage invalidates its
        # downstream cache unless the exact upstream bytes still match.
        stage_handle, row = choices[0]
        if row.get('input_artifacts') != upstream or row.get('input_checkpoint') != _checkpoint_identity(checkpoint):
            break
        for artifact in row['artifacts'].values():
            store.verify_blob(artifact['sha256'])
        complete[index] = (stage_handle, row)
        upstream.update({name: item['sha256'] for name, item in row['artifacts'].items()})
        checkpoint = row.get('checkpoint')
    return complete, upstream, checkpoint


def evaluator_identity(manifest, settings=None, metric=None):
    """Identity to use when collecting/calibrating policies for a frozen evaluator.

    Policy declarations themselves are omitted, allowing a manifest to acquire a
    calibrated gate without creating a circular identity dependency.
    """
    spec = load_manifest(manifest) if not isinstance(manifest, dict) else manifest
    if metric is None:
        if len(spec['metrics']) != 1:
            raise StrataError('policy evaluator identity requires a metric selector')
        metric = next(iter(spec['metrics']))
    if metric not in spec['metrics']:
        raise StrataError('unknown policy metric')
    evaluator = {key: spec.get(key) for key in ('stages', 'metrics', 'runtimes', 'reader', 'environment', 'verifier', 'cost')}
    dependencies = {f'{kind}:{name}': item['files'] for kind in ('inputs', 'dependencies') for name, item in spec[kind].items()}
    dependencies.update({f'verifier:{name}': item['files'] for name, item in (spec.get('verifier') or {}).get('dependencies', {}).items()})
    dependencies.update(runtime=digest(spec['runtimes']), reader=digest(spec['reader']), environment=digest(spec['environment']))
    return dict(scorer=digest(evaluator), dependencies=dependencies, domain=spec['contract'], settings={**spec['settings'], **(settings or {})}, unit=spec['metrics'][metric]['unit'])


def _policy_after_stage(store, experiment, stage, stage_handle, metrics, points, spent):
    spec = experiment['manifest']
    decisions = []
    action = None
    if spec.get('screening') and spec['screening']['after_stage'] == stage['name']:
        from .experiment_policy import ScreeningPolicy, record_screening_candidate
        policy = spec['screening']
        identity = evaluator_identity(spec, experiment['settings'], policy['metric'])
        if policy['metric'] not in metrics:
            raise StrataError('cheap stage did not produce the declared screening metric')
        candidate = record_screening_candidate(store, identity=identity, proxy=metrics[policy['metric']]['value'], source_record=stage_handle, candidate_key=experiment['identity'], metric=policy['metric'])
        decision = ScreeningPolicy(store, policy['policy']).decide(candidate, calibration=policy.get('calibration'), identity=identity)
        decisions.append(decision)
        if decision['action'] == 'prune':
            action = 'screened_out'
    if spec.get('continuation') and stage.get('checkpoint'):
        from .experiment_policy import ContinuationPolicy, record_trajectory
        policy = spec['continuation']
        identity = evaluator_identity(spec, experiment['settings'], policy['metric'])
        if policy['metric'] in metrics:
            points.append(dict(step=spec['stages'].index(stage) + 1, value=metrics[policy['metric']]['value'], compute_seconds=spent, source_record=stage_handle, metric=policy['metric']))
            trajectory = record_trajectory(store, identity=identity, run_id=experiment['identity'], points=points)
            decision = ContinuationPolicy(store, policy['policy']).decide(trajectory, calibration=policy.get('calibration'), identity=identity)
            decisions.append(decision)
            if decision['action'] == 'paused_by_forecast_policy':
                action = 'paused_by_forecast_policy'
    return action, decisions


def _screening_pairs(store, spec, decisions, stage, stage_handle, metrics):
    policy = spec.get('screening')
    if not policy or stage['name'] != policy.get('full_stage', spec['stages'][-1]['name']):
        return []
    from .experiment_policy import ScreeningPolicy, record_screening_pair
    metric = policy.get('full_metric', policy['metric'])
    if metric not in metrics:
        raise StrataError('full evaluator did not produce the declared audit metric')
    audits = []
    for decision in decisions:
        if decision.get('policy') != policy['policy'] or not decision.get('candidate'):
            continue
        candidate = store.get(decision['candidate'], 'screening_candidate')
        if candidate['source_record'] == stage_handle:
            continue
        pair = record_screening_pair(store, candidate=decision['candidate'], full=metrics[metric]['value'], full_record=stage_handle, metric=metric)
        if decision['action'] in {'audit_full', 'prune'}:
            audits.append(ScreeningPolicy(store, policy['policy']).audit(decision=decision['record'], pair=pair))
    return audits


def run_experiment(store, handle, owner='local', board=None, *, resume=True, override_policy=False, claim_timeout=0, deadline=None):
    """Run eligible stages with one owned claim; keep logs and checked artifacts."""
    from .experiment_board import ExperimentBoard
    experiment = store.get(handle, 'experiment')
    spec = experiment['manifest']
    _finite(claim_timeout, 'claim timeout')
    if claim_timeout < 0:
        raise StrataError('claim timeout cannot be negative')
    if deadline is not None:
        _finite(deadline, 'execution deadline')
    _assert_identity(spec)
    with _lock(store, handle):
        board = board or ExperimentBoard(store)
        _reconcile_lost(store, handle, board)
        state = experiment_status(store, handle)
        if state['status'] in _SUCCESS:
            for artifact in state['latest']['artifacts'].values():
                store.verify_blob(artifact['sha256'])
            return dict(state['latest'], reused=True)
        if state['status'] in {'reconciliation_required', 'budget_exhausted', 'attempt_budget_exhausted'}:
            return state
        if state['status'] in {'paused_by_forecast_policy', 'screened_out'} and not override_policy:
            return state
        if deadline is not None and time.monotonic() >= deadline:
            return dict(experiment=handle, status='budget_exhausted')
        completed, upstream, predecessor_checkpoint = {}, {}, None
        if resume and state['status'] not in {'artifact_validated', 'execution_finished', 'rejected_by_verifier', 'verification_failed'}:
            completed, upstream, predecessor_checkpoint = _resumable_stages(store, handle, spec)
        job = board.enqueue(handle, spec['resources'])
        claim_deadline = time.monotonic() + claim_timeout
        if deadline is not None:
            claim_deadline = min(claim_deadline, deadline)
        token = None
        try:
            token = board.claim(job, owner)
            while token is None and time.monotonic() < claim_deadline:
                if board.job(job)['state'] == 'cancelled':
                    return dict(experiment=handle, status='cancelled', job=job, withdrawn=True)
                time.sleep(min(.1, max(0, claim_deadline - time.monotonic())))
                token = board.claim(job, owner)
        except BaseException:
            if token is None:
                board.request_cancel(job)
            raise
        if token is None:
            cancelled = board.job(job)['state'] == 'cancelled'
            board.request_cancel(job)
            status = ('cancelled' if cancelled else 'budget_exhausted' if deadline is not None and time.monotonic() >= deadline
                      else 'waiting_for_resources')
            return dict(experiment=handle, status=status, job=job, withdrawn=True)
        if deadline is not None and time.monotonic() >= deadline:
            board.finish(token, 'budget_exhausted')
            return dict(experiment=handle, status='budget_exhausted', job=job)
        attempt = uuid.uuid4().hex
        attempt_dir = store.path / 'attempts' / handle / attempt
        started = time.monotonic()
        remaining = state['remaining_wall_seconds']
        if deadline is not None:
            remaining = min(remaining, max(0, deadline - started))
        try:
            _assert_identity(spec)
            attempt_dir.mkdir(parents=True)
            start_handle = store.put('experiment_attempt', dict(experiment=handle, attempt=attempt, job=job, owner=owner, reserved_seconds=min(remaining, spec['limits']['timeout']), started_at=time.time(), workdir=str(attempt_dir), policy_override=override_policy))
        except BaseException:
            board.finish(token, 'failed')
            raise
        artifacts, metrics, stage_receipts, points = {}, {}, [], []
        reused_stage_handles = {h for h, _ in completed.values()}
        decisions = []
        if completed and state.get('latest'):
            for decision in state['latest'].get('policy_decisions', []):
                if decision.get('candidate') and decision.get('action'):
                    candidate = store.get(decision['candidate'], 'screening_candidate')
                    if candidate['source_record'] in reused_stage_handles:
                        decisions.append(decision)
        checkpoint = predecessor_checkpoint
        latest_checkpoint = state['checkpoint'] if resume else None
        if (latest_checkpoint and not latest_checkpoint['completed']
                and latest_checkpoint['index'] == len(completed)
                and latest_checkpoint.get('input_artifacts') == upstream
                and latest_checkpoint.get('input_checkpoint') == _checkpoint_identity(predecessor_checkpoint)):
            checkpoint = latest_checkpoint
        status, error, verification, cost = 'failed', None, None, None
        current_index, current_stage, current_work = None, None, None
        verifying = False
        try:
            for index, stage in enumerate(spec['stages']):
                if index in completed:
                    old_handle, old = completed[index]
                    artifacts.update(old['artifacts'])
                    metrics.update(old['metrics'])
                    stage_receipts.append(dict(stage=stage['name'], reused_stage=old_handle, status='artifact_validated'))
                    if spec.get('continuation') and stage.get('checkpoint') and spec['continuation']['metric'] in old['metrics']:
                        points.append(dict(step=index + 1, value=old['metrics'][spec['continuation']['metric']]['value'], compute_seconds=old.get('cumulative_seconds', 0), source_record=old_handle, metric=spec['continuation']['metric']))
                    continue
                _assert_identity(spec, staged=True)
                current_index, current_stage = index, stage
                work = attempt_dir / f'stage-{index:04d}'
                work.mkdir()
                current_work = work
                sources = _stage_sources(store, work, spec)
                artifact_paths = _restore_artifacts(store, work, artifacts)
                input_artifacts = {name: item['sha256'] for name, item in artifacts.items()}
                input_checkpoint = _checkpoint_identity(checkpoint)
                checkpoint_path = None
                if checkpoint:
                    target = work / '.strata' / 'checkpoint' / PurePosixPath(checkpoint['artifact']['path']).name
                    store.copy_blob(checkpoint['artifact']['sha256'], target)
                    checkpoint_path = str(target)
                command = stage.get('resume_argv') if checkpoint and checkpoint['index'] == index and not checkpoint['completed'] else None
                argv = _substitute(command or stage['argv'], experiment, work, sources, artifact_paths, checkpoint_path)
                limits = dict(stage['limits'])
                limits['timeout'] = min(limits['timeout'], remaining - (time.monotonic() - started))
                if limits['timeout'] <= 0:
                    status = 'budget_exhausted'
                    break
                readonly = _readonly_runtimes(spec) + ([str(work / '.strata')] if (work / '.strata').exists() else [])
                receipt = _save_logs(store, _run_charged(store, handle, attempt, stage['name'], board, token, argv, work, environment=spec['environment'], readonly_paths=readonly, **limits))
                stage_receipts.append(dict(stage=stage['name'], **receipt))
                _verify_staged(spec, sources, artifacts={path: artifacts[name]['sha256'] for name, path in artifact_paths.items()})
                if receipt.get('returncode') != 0 or receipt.get('status') != 'execution_finished':
                    status = 'cancelled' if receipt.get('status') == 'cancelled' else 'failed'
                    error = dict(stage=stage['name'], kind=receipt.get('status'), diagnosis=receipt.get('diagnosis'))
                    try:
                        recovered = _committed_checkpoint(store, handle, attempt, index, stage, work, completed=False, input_artifacts=input_artifacts, input_checkpoint=input_checkpoint)
                        checkpoint = recovered or checkpoint
                    except (OSError, StrataError):
                        pass
                    break
                output, decoded = _collect(store, work, stage['outputs'], decode_names={m['output'] for m in spec['metrics'].values()})
                measured = _metrics(spec, decoded)
                artifacts.update(output)
                metrics.update(measured)
                checkpoint = _committed_checkpoint(store, handle, attempt, index, stage, work, completed=True, input_artifacts=input_artifacts, input_checkpoint=input_checkpoint) or checkpoint
                stage_handle = store.put('experiment_stage', dict(experiment=handle, attempt=attempt, index=index, name=stage['name'], artifacts=output, metrics=measured, input_artifacts=input_artifacts, input_checkpoint=input_checkpoint, checkpoint=checkpoint, receipt=receipt, cumulative_seconds=state['spent_wall_seconds'] + time.monotonic() - started))
                action, policy_decisions = _policy_after_stage(store, experiment, stage, stage_handle, measured, points, state['spent_wall_seconds'] + time.monotonic() - started)
                decisions.extend(policy_decisions)
                decisions.extend(_screening_pairs(store, spec, decisions, stage, stage_handle, measured))
                pending_audit = any(d.get('action') == 'audit_full' and not any(a.get('decision') == d.get('record') and 'false_negative' in a for a in decisions) for d in decisions)
                if action == 'paused_by_forecast_policy' and pending_audit:
                    override = dict(experiment=handle, attempt=attempt, action='continue_for_audit', reason='predeclared full-evaluator audit overrides forecast pause', stage=stage['name'])
                    decisions.append(dict(record=store.put('experiment_policy_override', override), **override))
                    action = None
                if action and not override_policy:
                    status = action
                    break
            else:
                missing = {o['name'] for o in spec['outputs']} - set(artifacts)
                if missing:
                    raise StrataError('required outputs missing from pipeline: ' + ', '.join(sorted(missing)))
                if set(spec['metrics']) - set(metrics):
                    raise StrataError('declared metric was not produced')
                status = 'artifact_validated' if artifacts else 'execution_finished'
                if spec.get('verifier'):
                    verifying = True
                    work = attempt_dir / 'verifier'
                    work.mkdir()
                    sources = _stage_sources(store, work, spec, verifier=True)
                    artifact_paths = _restore_artifacts(store, work, artifacts)
                    argv = _substitute(spec['verifier']['argv'], experiment, work, sources, artifact_paths, None)
                    limits = dict(spec['verifier']['limits'])
                    limits['timeout'] = min(limits['timeout'], remaining - (time.monotonic() - started))
                    if limits['timeout'] <= 0:
                        status = 'budget_exhausted'
                    else:
                        verification = _save_logs(store, _run_charged(store, handle, attempt, 'verifier', board, token, argv, work, environment=spec['environment'], readonly_paths=_readonly_runtimes(spec) + [str(work / '.strata')], **limits))
                        _verify_staged(spec, sources, verifier=True)
                        for name, path in artifact_paths.items():
                            if _hash(path) != artifacts[name]['sha256']:
                                raise StrataError('independent verifier modified the candidate artifact')
                        if (verification.get('returncode') != 0 or verification.get('status') != 'execution_finished'
                                or verification.get('diagnosis', {}).get('kind') == 'descendant_process_failed'):
                            status = 'verification_failed'
                            error = dict(kind='verifier_execution_failed', diagnosis=verification.get('diagnosis'),
                                         execution_status=verification.get('status'), returncode=verification.get('returncode'))
                            try:
                                verification['receipt'] = _decode(verification.get('stdout', verification.get('output', '')).encode(), 'json')
                            except StrataError:
                                pass
                        else:
                            payload = _decode(verification.get('stdout', verification.get('output', '')).encode(), 'json')
                            accepted = _selector(payload, spec['verifier']['accepted_selector'])
                            if type(accepted) is not bool:
                                raise StrataError('verifier acceptance selector must be a boolean')
                            from .experiment_comparisons import IDENTITY_SCHEMA, verifier_identity
                            verification.update(receipt=payload, identity=verifier_identity(spec, experiment),
                                identity_schema=IDENTITY_SCHEMA, version=spec['verifier']['version'])
                            status = 'accepted_by_verifier' if accepted else 'rejected_by_verifier'
                            if accepted:
                                for name, binding in spec['verifier'].get('metrics', {}).items():
                                    selector = binding['selector'] if isinstance(binding, dict) else binding
                                    measured = _selector(payload, selector)
                                    _finite(measured, 'independently verified metric')
                                    worker_metric = metrics.get(name)
                                    if worker_metric and not math.isclose(worker_metric['value'], measured, rel_tol=1e-12, abs_tol=1e-12):
                                        raise StrataError('worker metric disagrees with the independent verifier')
                                    unit = worker_metric['unit'] if worker_metric else binding['unit']
                                    source = dict(worker_metric['source']) if worker_metric else {}
                                    source.update(verifier_identity=verification['identity'], verifier_selector=selector)
                                    metrics[name] = dict(value=measured, unit=unit, evidence_kind='verified', source=source)
                if spec.get('cost') and status in {'accepted_by_verifier', 'artifact_validated', 'execution_finished'}:
                    formula = Formula(spec['cost']['expression'], functions={'ceil': math.ceil, 'floor': math.floor})
                    names = {node.id for node in ast.walk(formula.tree) if isinstance(node, ast.Name)} - set(formula.functions)
                    constants = {k: v for k, v in experiment['settings'].items() if type(v) in (int, float)}
                    bindings = spec['cost'].get('resources', {})
                    values = {**constants, **{k: m['value'] for k, m in metrics.items()}}
                    trusted = status == 'accepted_by_verifier' and names <= set(constants) | set(bindings)
                    if verification and verification.get('receipt'):
                        for name, selector in bindings.items():
                            values[name] = _finite(_selector(verification['receipt'], selector), 'verified cost resource')
                    values = {name: values[name] for name in names if name in values}
                    value = formula(values)
                    cost = dict(value=value, expression=spec['cost']['expression'], unit=spec['cost']['unit'], evidence_kind='verified' if trusted else 'model', inputs=values, resource_selectors=bindings, verifier_identity=verification.get('identity') if trusted else None, verification='independent_verifier_resources' if trusted else 'worker_reported')
            _assert_identity(spec, staged=True)
        except Exception as exc:
            status = ('verification_failed' if verifying else 'artifact_validation_failed'
                      if stage_receipts and stage_receipts[-1].get('returncode') == 0 else 'failed')
            error = dict(kind=type(exc).__name__, message=str(exc), stage=current_stage['name'] if current_stage else None)
        except (KeyboardInterrupt, SystemExit) as exc:
            status = 'cancelled'
            error = dict(kind=type(exc).__name__, message='coordinator interrupted; completed checkpoints retained')
        try:
            retention = _compact_attempt(store, attempt_dir, handle, attempt, spec.get('retention', 'compact'))
        except (OSError, StrataError) as exc:
            retention = dict(policy=spec.get('retention', 'compact'), errors=[str(exc)], status='cleanup_incomplete')
        finish = dict(experiment=handle, attempt=attempt, start_record=start_handle, status=status, seconds=time.monotonic() - started, metrics=metrics, artifacts=artifacts, stages=stage_receipts, error=error, verifier=verification, cost=cost, verified_cost=cost if cost and cost['evidence_kind'] == 'verified' and status == 'accepted_by_verifier' else None, checkpoint=checkpoint, policy_decisions=decisions, numeric_analysis='available' if metrics else 'unavailable: no metric selectors declared', finished_at=time.time(), interpretation=(spec.get('decisive') or {}).get('evidence_scope'))
        finish['retention'] = retention
        try:
            result_handle = store.put('experiment_finish', finish)
        except BaseException:
            board.finish(token, 'failed')
            raise
        board.finish(token, status, result_handle)
        return dict(handle=result_handle, **finish)


def _parameter_sets(parameters, max_count):
    if isinstance(parameters, list):
        if not all(isinstance(row, dict) for row in parameters):
            raise StrataError('parameter sets must be objects')
        if len(parameters) > max_count:
            raise StrataError('declared sweep exceeds max_jobs')
        return parameters
    if not isinstance(parameters, dict):
        raise StrataError('parameters must be explicit sets or a named grid')
    axes = {}
    for key, values in parameters.items():
        if isinstance(values, dict) and set(values) <= {'start', 'stop', 'step'} and {'start', 'stop'} <= set(values):
            if any(type(v) is not int for v in values.values()) or values.get('step', 1) == 0:
                raise StrataError('parameter ranges require integers and a nonzero step; stop is exclusive')
            values = range(values['start'], values['stop'], values.get('step', 1))
        if not isinstance(values, (list, range)) or not values:
            raise StrataError('parameter grid axes must be nonempty value lists')
        axes[key] = values
        if math.prod(len(v) for v in axes.values()) > max_count:
            raise StrataError('declared sweep exceeds max_jobs')
    return [dict(zip(axes, row)) for row in itertools.product(*axes.values())]


def create_sweep(store, manifest_path, parameters, seeds, replicates, max_jobs, *, workers=1, capacity=None, max_wall_seconds=None, screening=None):
    """Freeze a finite Cartesian experiment plan, including intentional replicates."""
    if type(max_jobs) is not int or max_jobs < 1 or type(workers) is not int or workers < 1:
        raise StrataError('max_jobs and workers must be positive integers')
    if not isinstance(seeds, list) or not seeds or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise StrataError('sweeps require a nonempty explicit list of distinct integer seeds')
    if type(replicates) is not int or replicates < 1:
        raise StrataError('replicates must be positive')
    parameter_sets = _parameter_sets(parameters, max_jobs // (len(seeds) * replicates))
    if not parameter_sets or len(parameter_sets) * len(seeds) * replicates > max_jobs:
        raise StrataError('declared sweep exceeds max_jobs or has no jobs')
    if len({canonical(s) for s in parameter_sets}) != len(parameter_sets):
        raise StrataError('duplicate parameter sets require the explicit replicates field')
    spec = load_manifest(manifest_path)
    _assert_identity(spec)
    _freeze_files(store, spec)
    handles = [_create_frozen(store, spec, settings, seed, replicate, sources_frozen=True) for settings in parameter_sets for seed in seeds for replicate in range(replicates)]
    requested = max_wall_seconds if max_wall_seconds is not None else len(handles) * spec['max_wall_seconds']
    _finite(requested, 'sweep max_wall_seconds', True)
    if requested < sum(store.get(h, 'experiment')['manifest']['max_wall_seconds'] for h in handles):
        raise StrataError('sweep wall budget must cover all frozen per-experiment reservations; lower manifest budgets first')
    if screening:
        if not isinstance(screening, dict) or not {'policy', 'identity', 'candidates', 'metric'} <= set(screening) or screening['metric'] not in spec['metrics']:
            raise StrataError('sweep screening needs policy, identity, metric and candidate mapping')
        store.get(screening['policy'])
    return store.put('experiment_sweep', dict(version=1, manifest=spec, experiments=handles, parameters=parameter_sets, seeds=seeds, replicates=replicates, max_jobs=max_jobs, workers=workers, capacity=capacity, max_wall_seconds=requested, screening=screening))


def sweep_status(store, handle):
    sweep = store.get(handle, 'experiment_sweep')
    experiments = [experiment_status(store, item) for item in sweep['experiments']]
    counts = {}
    for row in experiments:
        counts[row['status']] = counts.get(row['status'], 0) + 1
    return dict(sweep=handle, planned=len(experiments), counts=counts, spent_wall_seconds=sum(row['spent_wall_seconds'] for row in experiments), max_wall_seconds=sweep['max_wall_seconds'], experiments=experiments)


def run_sweep(store, handle, workers=None, *, owner='sweep', override_policy=False):
    """Wait fairly for capacity within one shared sweep deadline."""
    from .experiment_board import ExperimentBoard
    sweep = store.get(handle, 'experiment_sweep')
    workers = sweep['workers'] if workers is None else workers
    if type(workers) is not int or workers < 1 or workers > sweep['workers']:
        raise StrataError('workers must be positive and no greater than the frozen sweep concurrency')
    # Each run_experiment checks the frozen identity before executing or reusing
    # work, and reports per-experiment source drift without dropping other rows.
    started = time.monotonic()
    remaining = max(0, sweep['max_wall_seconds'] - sweep_status(store, handle)['spent_wall_seconds'])
    deadline = started + remaining
    # Construct before threads so initial schema/capacity has one writer.
    board = ExperimentBoard(store, capacity=sweep['capacity'])
    capacity = board.snapshot()['capacity']
    usable_workers = min([workers, *[capacity[key] // requested for key, requested in sweep['manifest']['resources'].items() if requested]])
    if usable_workers < 1:
        raise StrataError('one experiment exceeds the aggregate cooperative capacity')
    def run_one(experiment_id):
        local = Store(store.path)
        try:
            board = ExperimentBoard(local)
            decision = None
            if sweep.get('screening') and not override_policy:
                from .experiment_policy import ScreeningPolicy
                policy = sweep['screening']
                experiment = local.get(experiment_id, 'experiment')
                candidate = policy['candidates'].get(experiment_id, policy['candidates'].get(experiment['identity']))
                if candidate:
                    identity = evaluator_identity(sweep['manifest'], experiment['settings'], policy['metric'])
                    decision = ScreeningPolicy(local, policy['policy']).decide(candidate, calibration=policy.get('calibration'), identity=identity)
                    local.put('experiment_gate', dict(experiment=experiment_id, sweep=handle, decision=decision))
                    if decision['action'] == 'prune':
                        return dict(experiment=experiment_id, status='screened_out', decision=decision)
            result = run_experiment(local, experiment_id, owner=f'{owner}:{experiment_id[:12]}', board=board,
                                    override_policy=override_policy, claim_timeout=max(0, deadline - time.monotonic()),
                                    deadline=deadline)
            if decision and decision['action'] == 'audit_full' and result.get('handle') and result.get('metrics'):
                from .experiment_policy import record_screening_pair
                metric = policy.get('full_metric', policy['metric'])
                if metric in result['metrics']:
                    pair = record_screening_pair(local, candidate=decision['candidate'], full=result['metrics'][metric]['value'], full_record=result['handle'], metric=metric)
                    result['screening_audit'] = ScreeningPolicy(local, policy['policy']).audit(decision['record'], pair)
            return result
        finally:
            local.db.close()
    results = []
    with ThreadPoolExecutor(max_workers=usable_workers) as pool:
        futures = {pool.submit(run_one, item): item for item in sweep['experiments']}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except (StrataError, OSError) as exc:
                results.append(dict(experiment=futures[future], status='blocked', error=str(exc)))
    results.sort(key=lambda row: row['experiment'])
    receipt = dict(sweep=handle, results=results, status=sweep_status(store, handle), finished_at=time.time(),
                   elapsed_wall_seconds=time.monotonic() - started, scheduling='fifo_until_sweep_deadline')
    result_handle = store.put('experiment_sweep_run', receipt)
    return dict(handle=result_handle, **receipt)
