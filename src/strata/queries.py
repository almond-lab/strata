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

    def holders(self, design, peak):
        return dict(status='unknown', reason='physical object ownership needs generator allocation evidence', peak=peak)

    def is_function_of(self, wire, at, support):
        return dict(status='unknown', wire=wire, at=at, support=support,
                    reason='no exact functional-dependency certificate; sampled equality is insufficient')
