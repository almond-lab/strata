"""Restricted arithmetic, never eval. Bounds and best constructions stay distinct."""
import ast
import math
import operator
from .model import StrataError

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.Pow: operator.pow}


class Formula:
    def __init__(self, expression, functions=None):
        self.expression = expression
        self.tree = ast.parse(expression.replace('^', '**'), mode='eval').body
        self.functions = functions or {}

    def __call__(self, values):
        def visit(n):
            if isinstance(n, ast.Constant) and type(n.value) in (int, float):
                return n.value
            if isinstance(n, ast.Name) and n.id in values:
                return values[n.id]
            if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
                return (-1 if isinstance(n.op, ast.USub) else 1) * visit(n.operand)
            if isinstance(n, ast.BinOp) and type(n.op) in OPS:
                a, b = visit(n.left), visit(n.right)
                if isinstance(n.op, ast.Pow) and abs(b) > 16:
                    raise StrataError('exponent exceeds safety bound')
                return OPS[type(n.op)](a, b)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in self.functions and not n.keywords:
                return self.functions[n.func.id](*[visit(x) for x in n.args])
            raise StrataError(f'unsupported or unbound formula element: {ast.dump(n)}')
        result = float(visit(self.tree))
        if not math.isfinite(result) or result < 0:
            raise StrataError('cost must be finite and nonnegative')
        return result


def portfolio(formula, pairs):
    if formula != 'geomean(task.base_cost / task.cost)':
        raise StrataError('unsupported portfolio expression; supply a Python score adapter')
    if not pairs or any(a <= 0 or b <= 0 for a, b in pairs):
        raise StrataError('portfolio needs positive costs for every task')
    return math.exp(sum(math.log(a / b) for a, b in pairs) / len(pairs))


def rank_gaps(tree, ledgers, floors, formula, total):
    rows = []
    for r in tree.walk():
        ledger = ledgers.get(r.id)
        if ledger is None:
            continue
        applicable = [f for f in floors if f.contract == r.contract.fingerprint]
        for f in applicable:
            # Library is an upper bound; small exact results do not extrapolate.
            gap = None
            if f.is_full_lower_bound and f.resource in ledger and f.resource != 'Q':
                trial = dict(total)
                trial[f.resource] = max(0, total[f.resource] - max(0, ledger[f.resource] - f.value))
                gap = formula(total) - formula(trial)
            rows.append(dict(region=r.id, cost=ledger, floor=f.value, floor_kind=f.kind,
                             scope=f.scale, weighted_gap=gap, evidence=f.evidence,
                             note='opportunity bound, not measured savings'))
        if not applicable:
            rows.append(dict(region=r.id, cost=ledger, floor=None, floor_kind='unknown', weighted_gap=None))
    return sorted(rows, key=lambda x: -(x['weighted_gap'] if x['weighted_gap'] is not None else -1))
