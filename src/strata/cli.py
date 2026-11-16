import argparse
import json
from dataclasses import asdict
from pathlib import Path
from . import open as open_project
from .model import StrataError, digest
from .project import Design


def main(argv=None):
    parser = argparse.ArgumentParser(description='Strata contract restructuring and evidence')
    parser.add_argument('--project', default='strata.toml')
    sub = parser.add_subparsers(dest='cmd', required=True)
    sub.add_parser('init'); sub.add_parser('best')
    for name in ['analyze', 'lift', 'gaps']:
        p = sub.add_parser(name); p.add_argument('task')
        if name == 'lift': p.add_argument('--confirm', action='store_true')
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
        project = open_project(args.project)
        if args.cmd == 'init': result = project.init()
        elif args.cmd == 'best': result=project.best()
        elif args.cmd == 'replay': result=project.replay(args.design)
        elif args.cmd == 'diff':
            a,b=Design(project,args.a),Design(project,args.b)
            result=dict(architecture_changed=a.tree.architecture!=b.tree.architecture, before=asdict(a.tree), after=asdict(b.tree), cost_delta=b.record['cost']-a.record['cost'])
        elif args.cmd in ['analyze','lift','gaps','price']:
            task=project.task(args.task)
            circuit=project.frontend.load(project.root/task.config['base'])
            if args.cmd=='analyze':result=project.frontend.analyze(circuit,project.store.path/'analysis'/args.task)
            elif args.cmd=='lift':
                tree,coverage=project.frontend.lift(circuit)
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
    except (StrataError, KeyError, OSError, ValueError) as exc:
        print(json.dumps(dict(error=str(exc))))
        return 2


if __name__=='__main__':
    raise SystemExit(main())
