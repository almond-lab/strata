"""Declarative numerical gates supplement, never replace, the official verifier."""
import operator
import re
from fractions import Fraction
from .model import StrataError


def check_acceptance(config, resources):
    ops={'<=':operator.le,'>=':operator.ge,'<':operator.lt,'>':operator.gt,'==':operator.eq}
    for name, rule in config.items():
        if name in {'kind','tolerance'}:
            # Phase tolerance semantics belong to the pinned domain verifier.
            continue
        if name=='max_fail':
            key='fail_rate';op='<=';value=str(rule)
        else:
            key=name
            m=re.fullmatch(r'\s*(<=|>=|<|>|==)\s*([+\-\d.eE/]+)\s*',str(rule))
            if not m:raise StrataError('unsupported acceptance rule: '+name)
            op,value=m.groups()
        if key not in resources:raise StrataError('verifier omitted acceptance resource: '+key)
        try: threshold=float(Fraction(value))
        except (ValueError,ZeroDivisionError) as exc:raise StrataError('invalid acceptance threshold') from exc
        if not ops[op](resources[key],threshold):raise StrataError('acceptance failed: '+name)
