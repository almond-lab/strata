import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from time import monotonic
from . import open as open_project
from .model import StrataError, digest
from .project import Design


def _wire_list(value):
    if not value.strip():
        return ()
    wires = []
    for item in value.split(','):
        match = re.fullmatch(r'\s*(\d+)\s*(?:(?:\.\.|-)\s*(\d+)\s*)?', item)
        if not match:
            raise StrataError('wire indices must be nonnegative integers, comma lists, or inclusive ranges such as 16..27 or 16-27')
        start = int(match[1]); stop = int(match[2]) if match[2] is not None else start
        if stop < start:
            raise StrataError('wire range end must be at least its start')
        if len(wires) + stop - start + 1 > 1_000_000:
            raise StrataError('wire list expansion exceeds the 1000000-index limit')
        wires.extend(range(start, stop + 1))
    return tuple(wires)

def _bit_list(value):
    try:
        return _wire_list(re.sub(r'\bb(?=\d)', '', value))
    except StrataError as exc:
        raise StrataError('classical bit indices must be nonnegative integers or bN names, optionally in inclusive ranges') from exc


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
    from .experiment_cli import add_parsers, COMMANDS as EXPERIMENT_COMMANDS
    add_parsers(sub)
    p = sub.add_parser('report');p.add_argument('--store', required=True);p.add_argument('--handle');p.add_argument('--format', choices=('json', 'markdown'), default='json')
    p = sub.add_parser('memory-import');p.add_argument('input');p.add_argument('--store', required=True)
    p = sub.add_parser('memory-status');p.add_argument('--store', required=True);p.add_argument('--contract');p.add_argument('--scale')
    for name in ('function-of', 'slack', 'floors', 'hmr-debt'):
        p = sub.add_parser(name);p.add_argument('input')
        p.add_argument('--at', type=int, required=name == 'function-of')
        p.add_argument('--clean-inputs', default='')
        p.add_argument('--from',dest='from_at',type=int,default=0)
        p.add_argument('--range-unit',choices=('operation','text-line'),default='operation',
                       help='zero-based native operation cuts (default), or canonical KMX text lines')
        p.add_argument('--domain')
        p.add_argument('--observations',help='source-bound reachable-boundary observation receipt; sampled evidence remains sampled')
        p.add_argument('--max-operations',type=int,default=10000)
        p.add_argument('--max-variables',type=int,default=128)
        p.add_argument('--max-terms',type=int,default=65536)
        p.add_argument('--max-product-terms',type=int,default=1048576)
        p.add_argument('--max-assignments',type=int,default=65536)
        if name in {'function-of', 'slack','hmr-debt'}:
            p.add_argument('--timeout', type=float, default=5,
                           help='search seconds after window semantics are available')
        if name == 'function-of':
            p.add_argument('--wire', type=int, required=True);p.add_argument('--support', required=True)
            p.add_argument('--degree', type=int, default=2);p.add_argument('--max-forms', type=int, default=4096)
        elif name == 'hmr-debt':
            p.add_argument('--measurement',type=int,required=True,help='zero-based position of the HMR instruction')
            p.add_argument('--cuts',required=True,help='later cuts for reconstructing its measured predicate')
            p.add_argument('--support',help='surviving quantum wires; defaults to all nonconstant distinct functions')
            p.add_argument('--degree',type=int,default=3);p.add_argument('--max-forms',type=int,default=4096)
            p.add_argument('--max-outcomes',type=int,default=1024)
        elif name == 'slack':
            p.add_argument('--wires')
            p.add_argument('--max-inputs', type=int, default=16)
        else:
            p.add_argument('--outputs')
            p.add_argument('--exact',action='store_true')
            p.add_argument('--max-and',type=int,default=3)
            p.add_argument('--timeout',type=float,default=1.0)
            p.add_argument('--max-states',type=int,default=10000)
            p.add_argument('--max-expansions',type=int,default=100000)
            p.add_argument('--max-outcomes',type=int,default=1024)
    for name in ('inspect-file', 'optimize', 'discover', 'evaluate-file', 'note'):
        p=sub.add_parser(name);p.add_argument('input');p.add_argument('--out',required=True)
        if name=='note':p.add_argument('--note',required=True)
        if name=='evaluate-file':
            p.add_argument('candidate');p.add_argument('--hypothesis');p.add_argument('--regions')
            p.add_argument('--domain')
            p.add_argument('--max-proof-attempts',type=int,default=64)
            p.add_argument('--max-proof-operations',type=int,help='override the operation cap on a changed-span or domain proof')
            p.add_argument('--measurement-mode', choices=('coupled', 'independent-map'), default='coupled')
            p.add_argument('--internal-bits', default='', help='explicitly private classical outcome bits for independent-map checking')
        if name in {'inspect-file','discover','optimize','evaluate-file'}:p.add_argument('--clean-inputs',default='')
        if name in {'discover','inspect-file'}:
            p.add_argument('--range',dest='scan_range')
            p.add_argument('--range-unit', choices=('operation', 'text-line'), default='operation',
                           help='zero-based operation offsets (default) or original KMX text lines')
            p.add_argument('--growth-window',type=int,default=None,
                           help='streaming window cap; defaults to explicit --max-window, otherwise 8192')
            p.add_argument('--quiet-progress',action='store_true')
        p.add_argument('--max-window',type=int,default=None if name in {'discover','inspect-file'} else 256)
        p.add_argument('--max-variables',type=int,default=64)
        p.add_argument('--max-terms',type=int,default=4096)
        p.add_argument('--cost',default='Q*(toffoli+t/4)')
        if name=='optimize':p.add_argument('--max-candidates',type=int,default=64)
        if name in {'optimize','evaluate-file'}:
            p.add_argument('--accounting',choices=('static','expected'),default='static')
            p.add_argument('--external-distribution',help='JSON joint distribution of external classical input bits')
    for command in ('restructure','search'):
        p=sub.add_parser(command);p.add_argument('input');p.add_argument('--out',required=True)
        p.add_argument('--domain',help='complete finite assignment or address-promise JSON at the input of this artifact')
        p.add_argument('--max-assignments',type=int,default=65536,help='finite-domain construction row budget')
        p.add_argument('--clean-inputs',default='');p.add_argument('--cost',default='Q*(toffoli+t/4)')
        p.add_argument('--internal-bits',default='',help='existing classical outputs explicitly private at the complete search boundary')
        p.add_argument('--rounds',type=int,default=2);p.add_argument('--beam',type=int,default=4)
        p.add_argument('--max-candidates',type=int,default=24);p.add_argument('--archive')
        p.add_argument('--max-finalists',type=int,default=24,help='additional distinct structural alternatives to price and independently check')
        p.add_argument('--goal',choices=('discoveries','best-cost'),default='discoveries',
                       help='collect different cheaper designs, or stop once a known objective floor is reached')
        p.add_argument('--max-architectures',type=int,default=1024,help='joint representation and factor-graph proposal budget per portfolio')
        p.add_argument('--max-operations',type=int,default=10000);p.add_argument('--max-variables',type=int,default=128)
        p.add_argument('--max-terms',type=int,default=65536)
        p.add_argument('--max-product-terms',type=int,default=65536)
        p.add_argument('--range',dest='scan_range',help='complete episode START:STOP within the parent')
        p.add_argument('--range-unit',choices=('operation','text-line'),default='operation')
        p.add_argument('--width',type=int,help='explicit physical parent layout, including untouched helpers')
        p.add_argument('--accounting',choices=('static','expected'),default='static')
        p.add_argument('--external-distribution')
        p.add_argument('--deep-only',action='store_true',help='retain verified structural candidates')
        p.add_argument('--seconds',type=float,default=30.0 if command=='search' else None,
                       help='shared wall budget for starting or continuing constructor trials')
        p.add_argument('--trial-seconds',type=float,default=4.0 if command=='search' else None,
                       help='hard wall limit for each isolated Linux constructor')
        p.add_argument('--cancel-file',help='stop current and future trials when this path exists; search defaults to OUT/STOP')
    p=sub.add_parser('observe-boundary');p.add_argument('input');p.add_argument('--at',type=int,required=True)
    p.add_argument('--out',required=True);p.add_argument('--clean-inputs',default='')
    p.add_argument('--assignments',help='JSON list of initial assignments; omitted means bounded exhaustive initial cube')
    p.add_argument('--max-inputs',type=int,default=4096);p.add_argument('--max-rows',type=int,default=65536)
    p.add_argument('--max-outcomes',type=int,default=1024);p.add_argument('--max-operations',type=int,default=10000)
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
        if args.cmd in EXPERIMENT_COMMANDS:
            from .experiment_cli import execute
            return execute(args)
        if args.cmd=='observe-boundary':
            from .frontends import load_circuit
            from .boundary_observations import observe_boundary
            if Path(args.input).resolve()==Path(args.out).resolve():
                raise StrataError('observation output would overwrite input circuit')
            result=observe_boundary(load_circuit(args.input),args.at,
                assignments=json.loads(Path(args.assignments).read_text()) if args.assignments else None,
                clean_inputs=_wire_list(args.clean_inputs),max_inputs=args.max_inputs,max_rows=args.max_rows,
                max_outcomes=args.max_outcomes,max_operations=args.max_operations,output=args.out)
            print(json.dumps(result,indent=2));return 0
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
        if args.cmd=='hmr-debt':
            from .frontends import load_circuit
            from .hmr_debt import hmr_debt
            if args.observations or args.from_at or args.at is not None:
                raise StrataError('use --measurement and --cuts relative to the supplied artifact; use --domain for reachable inputs')
            result=hmr_debt(load_circuit(args.input),args.measurement,_wire_list(args.cuts),
                            support=None if args.support is None else _wire_list(args.support),
                            clean_inputs=_wire_list(args.clean_inputs),domain=args.domain,unit=args.range_unit,
                            **{k:getattr(args,k) for k in ('degree','max_forms','timeout','max_operations','max_variables',
                                'max_terms','max_product_terms','max_assignments','max_outcomes')})
            print(json.dumps(result,indent=2));return 0
        if args.cmd in {'function-of', 'slack', 'floors'}:
            from .frontends import load_circuit
            from .inference import functions_at, is_function_of, information_slack
            load_started = monotonic()
            circuit = load_circuit(args.input)
            load_seconds = monotonic() - load_started
            query_started = monotonic()
            from .domains import query_text_coordinates
            query_from,query_at=query_text_coordinates(circuit,args.from_at,args.at,unit=args.range_unit)
            options=dict(from_at=query_from,clean_inputs=_wire_list(args.clean_inputs),domain=args.domain,observations=args.observations,
                         max_operations=args.max_operations,max_variables=args.max_variables,max_terms=args.max_terms,
                         max_product_terms=args.max_product_terms,max_assignments=args.max_assignments)
            if args.cmd == 'function-of':
                result = is_function_of(circuit, args.wire, query_at, _wire_list(args.support),
                                        timeout=args.timeout, degree=args.degree, max_forms=args.max_forms, **options)
            elif args.cmd == 'slack':
                result = information_slack(circuit, query_at,
                    wires=None if args.wires is None else _wire_list(args.wires),
                    timeout=args.timeout, max_inputs=args.max_inputs, **options)
            else:
                from .floors import circuit_floors
                result=circuit_floors(circuit,at=query_at,outputs=None if args.outputs is None else _wire_list(args.outputs),
                    exact=args.exact,max_and=args.max_and,timeout=args.timeout,max_states=args.max_states,
                    max_expansions=args.max_expansions,max_outcomes=args.max_outcomes,**options)
            result['canonical_text_window']=dict(from_at=query_from,at=result.get('at',query_at),coordinate_unit='text-line')
            resolved_at=args.at
            if args.range_unit=='operation' and resolved_at is None and result.get('at') is not None:
                from .domains import window_lines
                selected,_=window_lines(circuit,query_from,result['at'],max_lines=args.max_operations)
                resolved_at=args.from_at+sum(line.split()[0] not in {'REGISTER','APPEND_TO_REGISTER'} for line in selected)
            result.update(coordinate_unit=args.range_unit,from_at=args.from_at,
                          at=resolved_at if args.range_unit=='operation' else result.get('at',query_at))
            result.setdefault('timings', {}).update(load_seconds=load_seconds, query_seconds=monotonic()-query_started)
            print(json.dumps(result, indent=2, default=str))
            return 0
        if args.cmd in {'restructure','search'}:
            from .restructure import restructure_file
            region=None
            if args.scan_range:
                parts=args.scan_range.split(':')
                if len(parts)!=2:raise StrataError('range must be START:STOP')
                region=tuple(map(int,parts))
            result=restructure_file(args.input,args.out,clean_inputs=_wire_list(args.clean_inputs),cost=args.cost,
                rounds=args.rounds,beam=args.beam,max_candidates=args.max_candidates,archive=args.archive,
                max_operations=args.max_operations,max_variables=args.max_variables,max_terms=args.max_terms,
                max_product_terms=args.max_product_terms,scan_range=region,range_unit=args.range_unit,width=args.width,
                accounting=args.accounting,external_distribution=json.loads(Path(args.external_distribution).read_text()) if args.external_distribution else None,
                deep_only=args.deep_only or args.cmd=='search',internal_bits=_bit_list(args.internal_bits),
                domain=args.domain,max_assignments=args.max_assignments,
                fast=args.cmd=='search',search_seconds=args.seconds,trial_seconds=args.trial_seconds,max_finalists=args.max_finalists,
                goal=args.goal,max_architectures=args.max_architectures,
                cancel_file=args.cancel_file or (Path(args.out)/'STOP' if args.cmd=='search' else None))
            print(json.dumps(result,indent=2,default=str));return 0
        if args.cmd in {'inspect-file','optimize','discover','evaluate-file','note'}:
            from .file_workflow import inspect_file, optimize_file
            options=dict(cost=args.cost,max_window=256 if args.max_window is None else args.max_window,
                         max_variables=args.max_variables,max_terms=args.max_terms)
            if args.cmd in {'inspect-file','discover','optimize','evaluate-file'}:options['clean_inputs']=_wire_list(args.clean_inputs)
            if args.cmd in {'optimize','evaluate-file'}:
                options.update(accounting=args.accounting,external_distribution=json.loads(Path(args.external_distribution).read_text()) if args.external_distribution else None)
            if args.cmd in {'discover','inspect-file'}:
                scan_range=None
                if args.scan_range:
                    parts=args.scan_range.split(':')
                    if len(parts)!=2:raise StrataError('range must be START:STOP')
                    scan_range=(int(parts[0]),int(parts[1]))
                growth_window = (args.growth_window if args.growth_window is not None
                                 else args.max_window if args.max_window is not None else 8192)
                options.update(scan_range=scan_range,range_unit=args.range_unit,growth_window=growth_window,
                    progress=None if args.quiet_progress else lambda row:print(json.dumps(row),file=sys.stderr,flush=True))
            if args.cmd=='optimize':result=optimize_file(args.input,args.out,max_candidates=args.max_candidates,**options)
            elif args.cmd=='discover':
                from .discovery import discovery_brief
                result=discovery_brief(args.input,args.out,**options)
            elif args.cmd=='evaluate-file':
                from .discovery import evaluate_candidate
                hypothesis=json.loads(Path(args.hypothesis).read_text()) if args.hypothesis else None
                regions=json.loads(Path(args.regions).read_text()) if args.regions else None
                result=evaluate_candidate(args.input,args.candidate,args.out,hypothesis=hypothesis,regions=regions,
                                          domain=args.domain,max_proof_attempts=args.max_proof_attempts,
                                          max_proof_operations=args.max_proof_operations,
                                          measurement_mode=args.measurement_mode,internal_bits=_bit_list(args.internal_bits),**options)
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
