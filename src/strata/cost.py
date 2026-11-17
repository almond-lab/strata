"""Restricted arithmetic, never eval. Bounds and best constructions stay distinct."""
import ast
import math
import operator
from .model import StrataError

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.Pow: operator.pow}
SAFE_MATH = {'log': math.log, 'ln': math.log, 'exp': math.exp,
             'sqrt': math.sqrt, 'min': min, 'max': max, 'abs': abs}


class ContextFunction:
    """Trusted adapter hook supplied the current resource mapping explicitly."""
    def __init__(self, function):
        self.function = function

    def call(self, values, arguments):
        return self.function(values, *arguments)

    def __call__(self, *arguments):
        return self.function({}, *arguments)


class Formula:
    def __init__(self, expression, functions=None):
        self.expression = expression
        if not isinstance(expression, str) or len(expression) > 16_384:
            raise StrataError('formula must be a bounded expression string')
        try:
            self.tree = ast.parse(expression.replace('^', '**'), mode='eval').body
        except (SyntaxError, RecursionError) as exc:
            raise StrataError('invalid formula expression') from exc
        if sum(1 for _ in ast.walk(self.tree)) > 512:
            raise StrataError('formula exceeds expression complexity limit')
        self.functions = {**SAFE_MATH, **(functions or {})}

    def __call__(self, values):
        def visit(n):
            if isinstance(n, ast.Constant) and type(n.value) in (int, float):
                return finite(n.value)
            if isinstance(n, ast.Name) and n.id in values:
                return finite(values[n.id])
            if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
                return (-1 if isinstance(n.op, ast.USub) else 1) * visit(n.operand)
            if isinstance(n, ast.BinOp) and type(n.op) in OPS:
                a, b = visit(n.left), visit(n.right)
                if isinstance(n.op, ast.Pow) and abs(b) > 16:
                    raise StrataError('exponent exceeds safety bound')
                return finite(OPS[type(n.op)](a, b))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in self.functions and not n.keywords:
                function = self.functions[n.func.id]
                arguments = [visit(x) for x in n.args]
                return finite(function.call(values, arguments) if isinstance(function, ContextFunction) else function(*arguments))
            raise StrataError(f'unsupported or unbound formula element: {ast.dump(n)}')
        def finite(value):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise StrataError('formula values must be finite real numbers')
            return float(value)
        try:
            result = float(visit(self.tree))
        except (ValueError, TypeError, ArithmeticError, RecursionError) as exc:
            raise StrataError('formula arithmetic domain or range error') from exc
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
