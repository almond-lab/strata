"""Exact expected execution counts for native classical control.

Fresh HMR outcomes are independent fair primitive bits in the native basis
trajectory model. Derived BIT values and condition snapshots are Boolean
functions of those same primitives, so correlated conditions are never priced
by multiplying marginal probabilities. External bits have no implicit prior.
"""
from collections import Counter
from fractions import Fraction
from functools import lru_cache
import re
import shlex

import kickmix as km

from .factored import DecisionDAG, DecisionBudget
from .kickmix import Kickmix
from .model import StrataError
from .semantic import _limits, _parse


MODEL = 'independent uniform HMR primitive outcomes; deterministic BIT functions and snapshot conditions; declared joint external classical distribution'


def _distribution(value, maximum):
    if value is None:
        return [(dict(), Fraction(1))], set()
    if not isinstance(value, (list, tuple)) or not value or len(value) > maximum:
        raise StrataError('external_distribution must be a nonempty bounded list of {values, probability} joint rows')
    rows, keys = [], None
    for row in value:
        if not isinstance(row, dict) or set(row) != {'values', 'probability'} or not isinstance(row['values'], dict):
            raise StrataError('external distribution rows require values and probability')
        values = row['values']
        if any(not isinstance(b, str) or not re.fullmatch(r'b[0-9]+', b) or type(v) not in {int, bool} or v not in {0, 1}
               for b, v in values.items()):
            raise StrataError('external distribution values must assign bN bits to zero or one')
        if keys is None: keys = set(values)
        if set(values) != keys:
            raise StrataError('external distribution rows must assign the same bits')
        try:
            probability = Fraction(str(row['probability']))
        except (ValueError, ZeroDivisionError):
            raise StrataError('external distribution probabilities must be exact rational numbers') from None
        if probability < 0:
            raise StrataError('external distribution probabilities must be nonnegative')
        rows.append((dict(values), probability))
    if sum(p for _, p in rows) != 1:
        raise StrataError('external distribution probabilities must sum exactly to one')
    return rows, keys


def conditional_gate_accounting(circuit, *, external_distribution=None, max_operations=100000,
                                max_variables=512, max_nodes=262144, max_apply_steps=1048576,
                                max_distribution_rows=65536):
    """Return static and exact expected counts, or explicit unknown values.

    ``external_distribution`` is a joint distribution, e.g.
    ``[{"values": {"b0": 0, "b1": 0}, "probability": "1/2"}, ...]``.
    Unspecified initial bits stay arbitrary; they are never assumed fair.
    Rational strings make every reported expectation JSON-safe and exact.
    Gate activation excludes quantum control values: a CCX executes whenever
    its classical condition is active, regardless of its quantum input bits.
    """
    _limits(max_operations=max_operations, max_variables=max_variables, max_nodes=max_nodes,
            max_apply_steps=max_apply_steps, max_distribution_rows=max_distribution_rows)
    distribution, declared = _distribution(external_distribution, max_distribution_rows)
    native = circuit if isinstance(circuit, km.Circuit) else km.Circuit(circuit)
    static_resources = Kickmix().measured(native)
    lines = str(native).splitlines()
    base = dict(static_resources=static_resources, model=MODEL,
                assumptions=dict(hmr_outcomes='independent_uniform_primitives',
                                 external_classical_distribution=external_distribution,
                                 unspecified_external_bits='arbitrary; no probability assumed'),
                instruction_scope='condition stack instructions count once; metadata uses native SDK weights; ordinary instruction bodies count when active',
                fixed_resources=['Q', 'reaction_depth'])
    if len(lines) > max_operations:
        return dict(base, status='unknown', reason='operation_budget', expected_resources=None)
    parsed, bits, outcomes = [], set(), []
    for at, line in enumerate(lines):
        item = _parse(line)
        # Classical execution accounting does not require a sign-only phase
        # model: fractional native rotations have the same activation rule.
        if item.get('reason') == 'unsupported_fractional_phase':
            words = shlex.split(line)
            condition = words[-1] if len(words) > 2 and words[-2] == 'if' else None
            item = dict(status='supported', op='Z_POW', bits=(), condition=condition)
        if item['status'] == 'opaque':
            return dict(base, status='unknown', reason=item['reason'], at=at, expected_resources=None)
        parsed.append((at, line, item))
        bits.update(item.get('bits', ()))
        if item.get('condition'): bits.add(item['condition'])
        if item['op'] == 'HMR': outcomes.append(f'm{len(outcomes)}')
    if len(bits) + len(outcomes) > min(max_variables, 512):
        return dict(base, status='unknown', reason='variable_budget', expected_resources=None)
    dag = DecisionDAG(sorted(bits) + outcomes, max_nodes=max_nodes, max_steps=max_apply_steps)
    rows, stack = [], []
    gate_counts, expected_counts = Counter(), Counter()
    metric_totals = dict(toffoli=Fraction(0), t=Fraction(0), rotations=Fraction(0), instructions=Fraction(0))
    metric_unknown, unknown_external = set(), set()
    probability_steps = 0

    @lru_cache(None)
    def conditional_probability(root, row_index):
        nonlocal probability_steps
        probability_steps += 1
        if probability_steps > max_apply_steps:
            raise DecisionBudget('condition_probability_budget')
        if root < 2: return Fraction(root)
        level, low, high = dag.nodes[root]
        variable = dag.variables[level]
        values = distribution[row_index][0]
        if variable in values:
            return conditional_probability(high if values[variable] else low, row_index)
        p0, p1 = conditional_probability(low, row_index), conditional_probability(high, row_index)
        if variable.startswith('m'):
            return None if p0 is None or p1 is None else (p0 + p1) / 2
        return p0 if p0 == p1 else None

    def probability(root):
        total = Fraction(0)
        for index, (_, weight) in enumerate(distribution):
            if not weight: continue
            value = conditional_probability(root, index)
            if value is None: return None
            total += weight * value
        return total

    weights_cache = {}
    try:
        classical = {b: dag.variable(b) for b in sorted(bits)}
        measurement = 0
        for at, line, item in parsed:
            op = item['op']
            condition = classical[item['condition']] if item.get('condition') else 1
            if item['status'] == 'metadata':
                active = 1
            elif op == 'PUSH_CONDITION':
                # Native PUSH snapshots its bit even inside an inactive scope.
                stack.append(condition)
                active = 1
            elif op == 'POP_CONDITION':
                if stack: stack.pop()
                active = 1
            else:
                active = condition
                for value in stack: active = dag.product(active, value)
            p = probability(active)
            support = sorted(dag.support(active))
            missing = sorted(set(support) & (bits - declared)) if p is None else []
            unknown_external.update(missing)
            gate_counts[op] += 1
            if p is None: expected_counts[op] = None
            elif expected_counts.get(op, 0) is not None: expected_counts[op] += p
            words = shlex.split(line)
            unconditional = ' '.join(words[:-2]) if item.get('condition') else line
            # PUSH requires its if operand; its cost is administrative only.
            key = (op, words[2] if op == 'Z_POW' else None)
            if key not in weights_cache:
                if op == 'PUSH_CONDITION': weights = dict(toffoli=0, t=0, rotations=0, instructions=1)
                else:
                    instruction = km.Circuit(unconditional)
                    single = instruction.operation_statistics()
                    weights = dict(toffoli=single.num_toffoli_like, t=single.num_t_like,
                                   rotations=single['Z_POW'], instructions=len(instruction))
                weights_cache[key] = weights
            for metric, weight in weights_cache[key].items():
                if weight and p is None: metric_unknown.add(metric)
                elif p is not None: metric_totals[metric] += weight * p
            rows.append(dict(at=at, op=op, condition_probability=None if p is None else str(p),
                             primitive_support=support, missing_external_distribution=missing))
            if op in {'BIT_INVERT', 'BIT_STORE0', 'BIT_STORE1'}:
                bit = item['bits'][0]
                old = classical[bit]
                delta = active if op == 'BIT_INVERT' else dag.product(active, dag.xor(old, int(op == 'BIT_STORE1')))
                classical[bit] = dag.xor(old, delta)
            elif op == 'HMR':
                bit = item['bits'][0]
                outcome = dag.variable(outcomes[measurement]); measurement += 1
                classical[bit] = dag.xor(classical[bit], dag.product(active, dag.xor(classical[bit], outcome)))
        if stack:
            return dict(base, status='unknown', reason='unclosed_condition_scope', expected_resources=None)
    except DecisionBudget as exc:
        return dict(base, status='unknown', reason=str(exc), expected_resources=None,
                    scanned_instructions=len(rows))
    expected_resources = {metric: None if metric in metric_unknown else str(value) for metric, value in metric_totals.items()}
    return dict(base, status='unknown' if metric_unknown else 'exact',
                reason='external_classical_distribution_required' if metric_unknown else None,
                static_gate_counts=dict(gate_counts),
                expected_gate_counts={op: None if value is None else str(value) for op, value in expected_counts.items()},
                expected_resources=expected_resources, conditions=rows,
                unknown_external_bits=sorted(unknown_external),
                decision_nodes=len(dag.nodes), probability_steps=probability_steps)


def measured_resources(circuit, *, accounting='static', external_distribution=None, **budgets):
    """Native resource keys with an explicit static or exact-expected basis.

    Q and reaction_depth retain their static physical/scheduling definitions.
    Expected additive metrics retain exact rational strings in metadata, and
    numeric values for the existing formula evaluator. Unknown expectations
    raise instead of silently reverting to static resource counts.
    """
    native = circuit if isinstance(circuit, km.Circuit) else km.Circuit(circuit)
    if accounting in {'static', 'static_emitted'}:
        return Kickmix().measured(native)
    if accounting not in {'expected', 'expected_executed'}:
        raise StrataError('accounting must be static or expected')
    receipt = conditional_gate_accounting(native, external_distribution=external_distribution, **budgets)
    if receipt['status'] != 'exact':
        suffix = ', '.join(receipt.get('unknown_external_bits', ()))
        raise StrataError('expected accounting unavailable: ' + receipt['reason'] + (': ' + suffix if suffix else ''))
    result = dict(receipt['static_resources'])
    for metric, exact in receipt['expected_resources'].items():
        value = Fraction(exact)
        result[metric] = value.numerator if value.denominator == 1 else float(value)
    result.update(accounting='expected_executed', expected_exact=receipt['expected_resources'],
                  expected_model=MODEL, external_distribution=external_distribution,
                  fixed_resources=receipt['fixed_resources'])
    return result
