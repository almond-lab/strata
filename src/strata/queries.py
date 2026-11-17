"""Small agent answers with handles to complete evidence."""
from .model import digest
from .engine import applicable_moves


class Queries:
    def __init__(self, project):
        self.project = project

    def tried(self, move, contract):
        rows = self.project.store.tried(contract, move)
        return dict(count=len(rows), results=[dict(handle=h, status=v['status'], reopen=v['reopen']) for h, v in rows[:10]])

    def debts(self, design, region):
        from dataclasses import asdict
        return [asdict(x) for x in design.tree.find(region).contract.debts]

    def applies(self, patch, design):
        return dict(applies=patch.applies(design.tree), scope='contract fingerprint and occurrence; verification still required')

    def moves(self, design):
        data = applicable_moves(design.tree)
        h = self.project.store.put('query', data)
        return dict(count=len(data), preview=data[:5], handle=h)

    def _circuit(self, design=None):
        from .project import Design
        if design is None:
            if len(self.project.tasks)!=1:return None,None
            design=next(iter(self.project.tasks.values())).base_design()
        if isinstance(design,str):design=Design(self.project,design)
        design.validate()
        task=self.project.task(design.record['task'])
        return task.parse_artifact(self.project.store.read_blob(design.record['artifact'])),design

    def holders(self, design, peak):
        from .inference import functions_at
        circuit,design=self._circuit(design)
        if not hasattr(circuit,'num_qubits'):return dict(status='unknown',reason='select a circuit component for ownership queries')
        clean=[q for r in design.tree.contract.registers if r.role=='clean' for q in r.wires]
        state=functions_at(circuit,peak,clean_inputs=clean)
        if state['status']!='exact':return state
        names={q:f'{r.name}[{i}]' for r in design.tree.contract.registers for i,q in enumerate(r.wires)}
        holders=[]
        for q,poly in state['functions'].items():
            variables=sorted({x for term in poly for x in term})
            original=next(iter(next(iter(poly)))) if len(poly)==1 and len(next(iter(poly)))==1 else None
            holders.append(dict(wire=q,input_object=names.get(original),depends_on=[names.get(x,f'q{x}') for x in variables],
                                polynomial=[sorted(t) for t in sorted(poly,key=lambda t:(len(t),tuple(sorted(t))))],known_zero=not poly))
        return dict(status='exact_values',at=state['at'],holders=holders,scope=state['scope'],
                    allocation_ownership='unknown without allocation events; names describe inferred contents, not allocator objects')

    def is_function_of(self, wire, at, support, *, design=None, **limits):
        from .inference import is_function_of
        circuit,design=self._circuit(design)
        if design is None:return dict(status='unknown',reason='select a design in a multi-task project')
        if not hasattr(circuit,'num_qubits'):return dict(status='unknown',reason='select a circuit component')
        clean=[q for r in design.tree.contract.registers if r.role=='clean' for q in r.wires]
        return is_function_of(circuit,wire,at,support,clean_inputs=clean,**limits)

    def slack(self, design, at=None, *, wires=None, **limits):
        from .inference import information_slack
        circuit,design=self._circuit(design)
        if not hasattr(circuit,'num_qubits'):return dict(status='unknown',reason='select a circuit component')
        clean=[q for r in design.tree.contract.registers if r.role=='clean' for q in r.wires]
        return information_slack(circuit,at,wires=wires,clean_inputs=clean,**limits)

    def research(self, design, hypotheses=None, policy=None, dependencies=()):
        from .research import ResearchRound
        plan=ResearchRound(self.project,design,policy,dependencies).plan(hypotheses)
        return dict(handle=plan["handle"], selected=len(plan["selected"]), deferred=len(plan["deferred"]),
                    preview=[dict(mechanism=x["hypothesis"]["mechanism"], regions=x["episode"]["regions"], readiness=x["readiness"]) for x in plan["selected"][:5]])
