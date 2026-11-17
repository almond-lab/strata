"""Human and agent entry points for the retained experiment workflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from .model import StrataError


COMMANDS = {'wrap', 'experiment', 'sweep', 'board', 'experiment-report',
            'agent-packet', 'agent-decision', 'experiment-policy'}


def add_parsers(sub):
    p = sub.add_parser('wrap', help='write a manifest for an existing command; no adapter')
    p.add_argument('--out', required=True)
    p.add_argument('--output', action='append', default=[], metavar='NAME=PATH:TYPE')
    p.add_argument('--metric', action='append', default=[], metavar='NAME=OUTPUT:SELECTOR:UNIT')
    p.add_argument('--setting', action='append', default=[], metavar='NAME=JSON')
    p.add_argument('--timeout', type=float, default=60)
    p.add_argument('--max-output', type=int, default=1_000_000, help='captured stdout and stderr limit in bytes')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--dependency',action='append',default=[],metavar='FILE_OR_DIR',
                   help='also pin a local import, data file, or complete source directory')
    p.add_argument('--memory-mb', type=int, default=2048)
    p.add_argument('--storage-mb', type=int, default=64,
                   help='cooperative temporary-storage reservation, not an enforced total quota')
    p.add_argument('--max-file-mb', type=int, default=64)
    p.add_argument('argv', nargs=argparse.REMAINDER)
    p = sub.add_parser('experiment', help='create, run or inspect an existing pipeline')
    p.add_argument('action', choices=('create', 'run', 'status', 'invalidate'))
    p.add_argument('input', help='manifest path for create/run, or retained experiment handle')
    p.add_argument('--store', required=True)
    p.add_argument('--setting', action='append', default=[], metavar='NAME=JSON')
    p.add_argument('--seed', type=int)
    p.add_argument('--replicate', type=int, default=0)
    p.add_argument('--owner', default='cli')
    p.add_argument('--override-policy', action='store_true', help='resume a policy pause, retaining charged budgets and an override receipt')
    p.add_argument('--parent', help='explicit retained parent when creating a descendant experiment')
    p.add_argument('--actor', help='attribution for an infrastructure correction')
    p.add_argument('--reason', help='reason for invalidating a machine-backed infrastructure receipt')
    p = sub.add_parser('sweep', help='declare and execute a bounded reproducible sweep')
    p.add_argument('action', choices=('create', 'run', 'status'))
    p.add_argument('input', help='manifest for create, retained sweep handle otherwise')
    p.add_argument('--store', required=True)
    p.add_argument('--parameters', help='JSON mapping of parameter lists, or explicit settings list')
    p.add_argument('--seeds', default='0')
    p.add_argument('--replicates', type=int, default=1)
    p.add_argument('--max-jobs', type=int, default=100)
    p.add_argument('--workers', type=int)
    p.add_argument('--capacity', help='JSON file with cooperative host capacity')
    p.add_argument('--override-policy', action='store_true')
    p = sub.add_parser('board', help='view claims/events, cancel or reconcile owned work')
    p.add_argument('--store', required=True)
    g = p.add_mutually_exclusive_group()
    g.add_argument('--events-after', type=int)
    g.add_argument('--cancel', metavar='JOB')
    g.add_argument('--reconcile', metavar='JOB')
    g.add_argument('--init-capacity', metavar='JSON', help='configure cooperative capacity; changes require an idle board')
    p.add_argument('--type', dest='event_types', action='append', help='include only these event types')
    p.add_argument('--exclude', dest='event_excludes', action='append', help='exclude event types, e.g. heartbeat')
    for name in ('experiment-report', 'agent-packet'):
        p = sub.add_parser(name)
        p.add_argument('--store', required=True)
        p.add_argument('--experiment')
        p.add_argument('--payload', help='JSON evidence/report configuration')
        p.add_argument('--comparison-identity', help='select one verifier/objective comparison group across experiments')
        p.add_argument('--out')
        if name == 'experiment-report':
            p.add_argument('--format', choices=('json', 'markdown'), default='json')
    for name in ('agent-decision', 'experiment-policy'):
        p = sub.add_parser(name)
        p.add_argument('input', help='JSON request')
        p.add_argument('--store', required=True)


def _read(path):
    try:
        return json.loads(path if isinstance(path, str) and path.lstrip().startswith(('{', '['))
                          else Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise StrataError(f'cannot read JSON request {path}: {exc}') from exc


def _settings(values):
    result = {}
    for item in values:
        key, sep, raw = item.partition('=')
        if not sep or not key or key in result:
            raise StrataError('settings must be distinct NAME=JSON values')
        try:
            result[key] = json.loads(raw)
        except ValueError as exc:
            raise StrataError('setting values must be JSON, including quoted strings') from exc
    return result


def wrap_command(args):
    argv = args.argv[1:] if args.argv[:1] == ['--'] else list(args.argv)
    if not argv:
        raise StrataError('wrap requires an argv command after --')
    executable = shutil.which(argv[0])
    if executable is None:
        raise StrataError('command executable was not found')
    argv[0] = str(Path(executable).absolute())  # preserve virtualenv invocation
    output_paths = {Path(v.partition('=')[2].rpartition(':')[0]) for v in args.output}
    dependencies = {f'extra{i}':str(Path(value).absolute()) for i,value in enumerate(args.dependency)}
    trees = [Path(path).resolve() for path in dependencies.values() if Path(path).is_dir()]
    python_files = []
    for i, value in enumerate(argv[1:], 1):
        path = Path(value)
        if path.is_file() and path not in output_paths:
            if any(path.resolve().is_relative_to(tree) for tree in trees):
                argv[i] = str(path.resolve())
                continue
            if path.suffix == '.py':
                python_files.append(path.name)
            name = f'arg{i}'
            dependencies[name] = str(path.resolve())
            argv[i] = '{dependency:' + name + '}'
    outputs = []
    for declaration in args.output:
        name, sep, value = declaration.partition('=')
        path, colon, kind = value.rpartition(':')
        if not sep or not colon or not name or not path:
            raise StrataError('output must be NAME=RELATIVE_PATH:TYPE')
        outputs.append(dict(name=name, path=path, type=kind))
    metrics = {}
    for declaration in args.metric:
        name, sep, value = declaration.partition('=')
        parts = value.split(':', 2)
        if not sep or not name or name in metrics or len(parts) != 3:
            raise StrataError('metric must be NAME=OUTPUT:SELECTOR:UNIT')
        output, selector, unit = parts
        metrics[name] = dict(output=output, selector=selector, unit=unit,
                             evidence_kind='measured')
    spec = dict(version=1, argv=argv, dependencies=dependencies,
                settings=_settings(args.setting), outputs=outputs, metrics=metrics,
                resources=dict(threads=args.threads, memory_bytes=args.memory_mb * 1024**2,
                               storage_bytes=args.storage_mb * 1024**2),
                limits=dict(timeout=args.timeout, max_output=args.max_output, memory_bytes=args.memory_mb * 1024**2,
                            max_file_bytes=args.max_file_mb * 1024**2))
    # Python imports must leave immutable staged source trees byte-identical.
    if trees and Path(argv[0]).name.startswith(('python', 'pypy')):
        spec['environment'] = {'PYTHONDONTWRITEBYTECODE': '1'}
    destination = Path(args.out).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise StrataError('manifest destination already exists')
    # Validate the proposed manifest through the same parser used by execution.
    from .experiments import load_manifest
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile(mode='w', suffix='.json', dir=destination.parent) as temporary:
        json.dump(spec, temporary, allow_nan=False)
        temporary.flush()
        load_manifest(temporary.name)
    destination.write_text(json.dumps(spec, indent=2, allow_nan=False) + '\n')
    return dict(manifest=str(destination), status='wrapped',
                dependency_notice=(f'Pinned Python files {", ".join(python_files)}. Sibling imports are staged only if declared. '
                                   'Use --dependency FILE_OR_DIR to include helpers or the source directory.' if python_files else None),
                next_command=['strata', 'experiment', 'run', str(destination),
                              '--store', 'work/experiments'],
                scope='Existing command recorded; no run or acceptance is claimed')


def dispatch(args):
    if args.cmd == 'wrap':
        return wrap_command(args)
    from .store import Store
    store = Store(args.store)
    try:
        if args.cmd == 'experiment':
            from .experiments import create_experiment, run_experiment, experiment_status
            if args.action == 'invalidate':
                from .experiment_observations import invalidate_verification
                return invalidate_verification(store, args.input, actor=args.actor, reason=args.reason)
            supplied_path = Path(args.input).is_file()
            if args.action == 'create' or args.action == 'run' and supplied_path:
                handle = create_experiment(store, args.input, settings=_settings(args.setting) or None,
                                           seed=args.seed, replicate=args.replicate, parent=args.parent)
            else:
                if args.setting or args.seed is not None or args.replicate != 0 or args.parent:
                    raise StrataError('retained experiments are immutable; create a new experiment for changed settings')
                handle = args.input
            if args.action == 'create':
                return dict(experiment=handle, status='created')
            if args.action == 'run':
                return run_experiment(store, handle, owner=args.owner, override_policy=args.override_policy)
            return experiment_status(store, handle)
        if args.cmd == 'sweep':
            from .experiments import create_sweep, run_sweep, experiment_status
            if args.action == 'create':
                if not args.parameters:
                    raise StrataError('sweep create needs --parameters')
                try:
                    seeds = [int(value) for value in args.seeds.split(',')]
                except ValueError as exc:
                    raise StrataError('seeds must be comma-separated integers') from exc
                handle = create_sweep(store, args.input, parameters=_read(args.parameters),
                                      seeds=seeds, replicates=args.replicates,
                                      max_jobs=args.max_jobs, workers=args.workers or 1,
                                      capacity=_read(args.capacity) if args.capacity else None)
                return dict(sweep=handle, status='created')
            if args.action == 'run':
                return run_sweep(store, args.input, workers=args.workers, override_policy=args.override_policy)
            policy = store.get(args.input, 'experiment_sweep')
            return dict(sweep=args.input, policy=policy,
                        experiments=[experiment_status(store, h) for h in policy['experiments']])
        if args.cmd == 'board':
            from .experiment_board import ExperimentBoard
            board = ExperimentBoard(store)
            if args.init_capacity:
                return board.configure_capacity(_read(args.init_capacity))
            if args.cancel:
                return board.request_cancel(args.cancel)
            if args.reconcile:
                return board.reconcile(args.reconcile)
            if args.events_after is not None or args.event_types or args.event_excludes:
                return board.event_page(after=args.events_after or 0, types=args.event_types,
                                        exclude_types=args.event_excludes)
            return board.snapshot()
        if args.cmd == 'experiment-policy':
            from .experiment_policy import policy_request
            return policy_request(store, _read(args.input))
        from .experiment_reports import (experiment_report, render_report_markdown,
                                         agent_packet, record_agent_decision)
        if args.cmd == 'agent-decision':
            return record_agent_decision(store, **_read(args.input))
        payload = _read(args.payload) if args.payload else None
        if args.comparison_identity:
            payload = dict(payload or {}, comparison_identity=args.comparison_identity)
        if args.cmd == 'agent-packet':
            return agent_packet(store, experiment=args.experiment, payload=payload)
        result = experiment_report(store, experiment=args.experiment, payload=payload)
        return render_report_markdown(result) if args.format == 'markdown' else result
    finally:
        store.db.close()


def execute(args):
    result = dispatch(args)
    text = result if isinstance(result, str) and args.cmd == 'experiment-report' and args.format == 'markdown' else json.dumps(result, indent=2, allow_nan=False)
    if getattr(args, 'out', None) and args.cmd != 'wrap':
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + '\n')
    print(text)
    if args.cmd == 'experiment' and args.action == 'run' and isinstance(result, dict):
        if result.get('status') not in {'execution_finished', 'artifact_validated', 'accepted_by_verifier'}:
            return 1
    return 0
