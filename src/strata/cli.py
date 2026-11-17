import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from . import open as open_project
from .model import StrataError, digest
from .project import Design


def _wire_list(value):
    wires = tuple(int(item.strip()) for item in value.split(',')) if value.strip() else ()
    if any(wire < 0 for wire in wires):
        raise StrataError('wire indices must be nonnegative')
    return wires


def _memory_status(store, contract, scale):
    from .memory import apply_memory
    if contract:
        return apply_memory(store, contract=contract, scale=scale)
    contracts = sorted({record['contract'] for _, record in store.all('negative')})
    items = [item for fingerprint in contracts
             for item in apply_memory(store, contract=fingerprint, scale=scale)['items']
             if item['contract'] == fingerprint]
    return dict(contracts=contracts, scale=scale, items=items,
                unbound=[dict(handle=handle, identifier=record.get('identifier'), binding_status='unbound')
                         for handle, record in store.all('unbound_attempt')],
                counts={name: sum(item['disposition'] == name for item in items)
                        for name in ('priced', 'closed', 'closed_at_size', 'reopen')})


def main(argv=None):
    parser = argparse.ArgumentParser(description='Strata contract restructuring and evidence')
    parser.add_argument('--project', default='strata.toml')
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('report');p.add_argument('--store', required=True);p.add_argument('--handle');p.add_argument('--format', choices=('json', 'markdown'), default='json')
    p = sub.add_parser('memory-import');p.add_argument('input');p.add_argument('--store', required=True)
    p = sub.add_parser('memory-status');p.add_argument('--store', required=True);p.add_argument('--contract');p.add_argument('--scale')
    for name in ('function-of', 'slack', 'floors'):
        p = sub.add_parser(name);p.add_argument('input')
        p.add_argument('--at', type=int, required=name == 'function-of')
        p.add_argument('--clean-inputs', default='')
        p.add_argument('--from',dest='from_at',type=int,default=0)
        p.add_argument('--domain')
        p.add_argument('--max-operations',type=int,default=10000)
        p.add_argument('--max-variables',type=int,default=128)
        p.add_argument('--max-terms',type=int,default=65536)
        p.add_argument('--max-product-terms',type=int,default=1048576)
        p.add_argument('--max-assignments',type=int,default=65536)
        if name == 'function-of':
            p.add_argument('--wire', type=int, required=True);p.add_argument('--support', required=True)
        elif name == 'slack':
            p.add_argument('--wires')
    for name in ('inspect-file', 'optimize', 'discover', 'evaluate-file', 'note'):
        p=sub.add_parser(name);p.add_argument('input');p.add_argument('--out',required=True)
        if name=='note':p.add_argument('--note',required=True)
        if name=='evaluate-file':
            p.add_argument('candidate');p.add_argument('--hypothesis');p.add_argument('--regions')
            p.add_argument('--domain')
        if name in {'inspect-file','discover','optimize'}:p.add_argument('--clean-inputs',default='')
        if name in {'discover','inspect-file'}:
            p.add_argument('--range',dest='scan_range')
            p.add_argument('--growth-window',type=int,default=8192)
            p.add_argument('--quiet-progress',action='store_true')
        p.add_argument('--max-window',type=int,default=256)
        p.add_argument('--max-variables',type=int,default=64)
        p.add_argument('--max-terms',type=int,default=4096)
        p.add_argument('--cost',default='Q*(toffoli+t/4)')
        if name=='optimize':p.add_argument('--max-candidates',type=int,default=64)
    p=sub.add_parser('restructure');p.add_argument('input');p.add_argument('--out',required=True)
    p.add_argument('--clean-inputs',default='');p.add_argument('--cost',default='Q*(toffoli+t/4)')
    p.add_argument('--rounds',type=int,default=2);p.add_argument('--beam',type=int,default=4)
    p.add_argument('--max-candidates',type=int,default=24);p.add_argument('--archive')
    p.add_argument('--max-operations',type=int,default=10000);p.add_argument('--max-variables',type=int,default=128)
    p.add_argument('--max-terms',type=int,default=65536)
    for name in ('numeric-budget','numeric-forecast','numeric-floor','numeric-bets'):
        p=sub.add_parser(name);p.add_argument('input');p.add_argument('--out',required=True);p.add_argument('--store')
    p=sub.add_parser('numeric-job');p.add_argument('job');p.add_argument('--store',required=True)
    p.add_argument('--run-stages',type=int,default=0)
    sub.add_parser('init'); sub.add_parser('best')
    for name in ['analyze', 'lift', 'gaps']:
        p = sub.add_parser(name); p.add_argument('task')
        if name == 'lift': p.add_argument('--confirm', action='store_true')
    p = sub.add_parser('research');p.add_argument('task');p.add_argument('--run',action='store_true');p.add_argument('--max-lines',type=int,default=6);p.add_argument('--hypotheses');p.add_argument('--parent')
    p = sub.add_parser('price');p.add_argument('task');p.add_argument('--region', required=True);p.add_argument('--resource', required=True);p.add_argument('--delta', required=True, type=int)
    p = sub.add_parser('replay');p.add_argument('design')
    p = sub.add_parser('diff');p.add_argument('a');p.add_argument('b')
    p = sub.add_parser('lab');p.add_argument('line');p.add_argument('--scale', default='')
    p = sub.add_parser('verify');p.add_argument('line')
    p = sub.add_parser('line');ss=p.add_subparsers(dest='action', required=True)
    o=ss.add_parser('open');o.add_argument('--parent',required=True);o.add_argument('--move',required=True);o.add_argument('--regions',required=True);o.add_argument('--implementation',default='{}');o.add_argument('--note',default='');o.add_argument('--replacement-regions',default='')
    for name in ['run','status','park']:
        q=ss.add_parser(name);q.add_argument('line')
        if name=='park':q.add_argument('--reason',required=True);q.add_argument('--reopen',required=True)
    args = parser.parse_args(argv)
    try:
        if args.cmd in {'numeric-budget','numeric-forecast','numeric-floor','numeric-bets'}:
            from .store import Store
            source=Path(args.input).resolve();destination=Path(args.out).resolve()
            if source==destination:raise StrataError('numeric report must not overwrite its input')
            specification=json.loads(source.read_text())
            if not isinstance(specification,dict):raise StrataError('numeric request must be a JSON object')
            store=Store(args.store) if args.store else None
            try:
                if args.cmd=='numeric-budget':
                    from .numeric import error_budget
                    result=error_budget(store=store,**specification)
                elif args.cmd=='numeric-forecast':
                    from .numeric_models import forecast_convergence
                    result=forecast_convergence(**specification)
                elif args.cmd=='numeric-floor':
                    from .numeric_models import exact_grid_floor
                    result=exact_grid_floor(**specification)
                else:
                    from .numeric_jobs import score_bets
                    result=score_bets(store=store,**specification)
                if store:
                    result=dict(result,record=store.put(args.cmd.replace('-','_'),result))
            finally:
                if store:store.db.close()
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
            print(json.dumps(result,indent=2));return 0
        if args.cmd=='numeric-job':
            from .store import Store
            from .numeric_jobs import NumericJob
            store=Store(args.store)
            try:
                job=NumericJob(store,args.job)
                if args.run_stages<0:raise StrataError('run stages must be nonnegative')
                result=job.run(stages=args.run_stages) if args.run_stages else job.status()
            finally:store.db.close()
            print(json.dumps(result,indent=2));return 0
        if args.cmd in {'report', 'memory-import', 'memory-status'}:
            from .store import Store
            store = Store(args.store)
            try:
                if args.cmd == 'report':
                    from .reports import report, markdown_report
                    result = report(store, handle=args.handle)
                    if args.format == 'markdown':
                        print(markdown_report(result))
                        return 0
                elif args.cmd == 'memory-import':
                    from .memory import import_attempts
                    result = import_attempts(store, args.input)
                else:
                    scale = json.loads(args.scale) if args.scale is not None else None
                    if scale is not None and not isinstance(scale, dict):
                        raise StrataError('memory scale must be a JSON object')
                    result = _memory_status(store, args.contract, scale)
            finally:
                store.db.close()
            print(json.dumps(result, indent=2, default=str))
            return 0
        if args.cmd in {'function-of', 'slack', 'floors'}:
            from .frontends import load_circuit
            from .inference import functions_at, is_function_of, information_slack
            circuit = load_circuit(args.input)
            options=dict(from_at=args.from_at,clean_inputs=_wire_list(args.clean_inputs),domain=args.domain,
                         max_operations=args.max_operations,max_variables=args.max_variables,max_terms=args.max_terms,
                         max_product_terms=args.max_product_terms,max_assignments=args.max_assignments)
            if args.cmd == 'function-of':
                result = is_function_of(circuit, args.wire, args.at, _wire_list(args.support),
                                        **options)
            elif args.cmd == 'slack':
                result = information_slack(circuit, args.at,
                    wires=None if args.wires is None else _wire_list(args.wires),
                    **options)
            else:
                from .floors import circuit_floors
                result=circuit_floors(circuit,at=args.at,**options)
            print(json.dumps(result, indent=2, default=str))
            return 0
        if args.cmd=='restructure':
            from .restructure import restructure_file
            result=restructure_file(args.input,args.out,clean_inputs=_wire_list(args.clean_inputs),cost=args.cost,
                rounds=args.rounds,beam=args.beam,max_candidates=args.max_candidates,archive=args.archive,
                max_operations=args.max_operations,max_variables=args.max_variables,max_terms=args.max_terms)
            print(json.dumps(result,indent=2,default=str));return 0
        if args.cmd in {'inspect-file','optimize','discover','evaluate-file','note'}:
            from .file_workflow import inspect_file, optimize_file
            options=dict(cost=args.cost,max_window=args.max_window,max_variables=args.max_variables,max_terms=args.max_terms)
            if args.cmd in {'inspect-file','discover','optimize'}:options['clean_inputs']=_wire_list(args.clean_inputs)
            if args.cmd in {'discover','inspect-file'}:
                scan_range=None
                if args.scan_range:
                    parts=args.scan_range.split(':')
                    if len(parts)!=2:raise StrataError('range must be START:STOP')
                    scan_range=(int(parts[0]),int(parts[1]))
                options.update(scan_range=scan_range,growth_window=args.growth_window,
                    progress=None if args.quiet_progress else lambda row:print(json.dumps(row),file=sys.stderr,flush=True))
            if args.cmd=='optimize':result=optimize_file(args.input,args.out,max_candidates=args.max_candidates,**options)
            elif args.cmd=='discover':
                from .discovery import discovery_brief
                result=discovery_brief(args.input,args.out,**options)
            elif args.cmd=='evaluate-file':
                from .discovery import evaluate_candidate
                hypothesis=json.loads(Path(args.hypothesis).read_text()) if args.hypothesis else None
                regions=json.loads(Path(args.regions).read_text()) if args.regions else None
                result=evaluate_candidate(args.input,args.candidate,args.out,hypothesis=hypothesis,regions=regions,domain=args.domain,**options)
            elif args.cmd=='note':
                from .discovery import record_research_note
                result=record_research_note(args.input,args.out,json.loads(Path(args.note).read_text()))
            else:result=inspect_file(args.input,args.out,**options)
            print(json.dumps(result,indent=2,default=str))
            return 0
        project = open_project(args.project)
        if args.cmd == 'init': result = project.init()
        elif args.cmd == 'best': result=project.best()
        elif args.cmd == 'research':
            from .research import ResearchRound, ResearchPolicy, Hypothesis
            hypotheses=None if not args.hypotheses else [Hypothesis(**x) for x in json.loads(Path(args.hypotheses).read_text())]
            parent=Design(project,args.parent or project.best()['tasks'][args.task]['design'])
            if parent.record['task']!=args.task:raise StrataError('research parent belongs to a different task')
            round_=ResearchRound(project,parent,ResearchPolicy(max_lines=args.max_lines))
            plan=round_.plan(hypotheses)
            result=round_.run(plan['handle']) if args.run else plan
        elif args.cmd == 'replay': result=project.replay(args.design)
        elif args.cmd == 'diff':
            a,b=Design(project,args.a),Design(project,args.b)
            result=dict(architecture_changed=a.tree.architecture!=b.tree.architecture, before=asdict(a.tree), after=asdict(b.tree), cost_delta=b.record['cost']-a.record['cost'])
        elif args.cmd in ['analyze','lift','gaps','price']:
            task=project.task(args.task)
            frontend=task.frontend
            circuit=frontend.load(project.root/task.config['base'])
            if args.cmd=='analyze':result=frontend.analyze(circuit,project.store.path/'analysis'/args.task)
            elif args.cmd=='lift':
                tree,coverage=frontend.lift(circuit)
                h=project.store.put('lift',dict(tree=asdict(tree),coverage=coverage,confirmed=args.confirm))
                result=dict(handle=h,coverage=coverage,confirmed=args.confirm,
                            note='Confirmation records a proposal; only verified adapters create certified designs.')
            elif args.cmd=='gaps':result=task.base_design().gaps()
            else:result=task.base_design().price(args.region,args.resource,args.delta)
        elif args.cmd=='lab':
            scale=dict((k,int(v)) for k,v in (pair.split('=') for pair in args.scale.split(',') if pair))
            result=asdict(project.line(args.line).lab(**scale))
        elif args.cmd=='verify':result=dict(design=project.line(args.line).verify().id)
        elif args.action=='open':
            result=dict(line=project.open_line(args.parent,args.move,args.regions.split(','),args.note,json.loads(args.implementation),replacement_regions=tuple(filter(None,args.replacement_regions.split(',')))).id)
        elif args.action=='status':result=project.line(args.line).status()
        elif args.action=='park':
            project.line(args.line).park(args.reason,args.reopen);result=dict(status='parked',evidence='estimate',scientific_status='priced')
        else:
            line=project.line(args.line);lab=line.lab()
            if not lab.passed:raise StrataError('lab failed: '+json.dumps(asdict(lab)))
            line.build();result=dict(design=line.verify().id)
        print(json.dumps(result,indent=2,default=str))
        return 0
    except (StrataError, KeyError, OSError, ValueError, TypeError) as exc:
        print(json.dumps(dict(error=str(exc))))
        return 2


if __name__=='__main__':
    raise SystemExit(main())
