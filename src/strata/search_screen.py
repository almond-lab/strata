"""Cheap eligibility tests for the installed structural search families."""
from math import isfinite
from pathlib import Path
from time import monotonic

from .model import StrataError


def positive_seconds(value, name):
    if value is not None and (type(value) not in (int, float) or not isfinite(value) or value <= 0):
        raise StrataError(f'{name} must be a positive finite number')


def stop_reason(deadline, cancel_file):
    if cancel_file is not None and Path(cancel_file).exists():
        return 'cancel_requested'
    if deadline is not None and monotonic() >= deadline:
        return 'search_time_budget'
    return None


def exact_features(semantics, clean):
    if semantics.get('status') != 'exact':
        return {}
    deltas = {}
    for name, value in semantics['outputs'].items():
        q = int(name)
        initial = set() if q in clean else {frozenset({q})}
        poly = {frozenset(term) for term in value['monomials']} ^ initial
        if poly:
            deltas[q] = poly
    support = {q for p in deltas.values() for t in p for q in t}
    return dict(sign=bool(semantics['phase']['monomials']), changed=len(deltas),
                dirty_targets=len(set(deltas) - set(clean)), support=len(support),
                target_dependent=bool(support & deltas.keys()))


def inapplicable(name, features, context):
    """Reject a constructor only under one of its actual preconditions."""
    small = 'joint_boundary_synthesis'
    if name=='fractional_observable_search' and not context.get('fractional_phase'):
        return 'no fractional rotations in the source'
    schedule = 'joint_storage_recomputation'
    cleanup = 'measurement_debt_lowering'
    value = {small, schedule, cleanup, 'algebraic_boundary_synthesis',
             'affine_representation_search','output_basis_search','consumer_functional_decomposition','architecture_portfolio'}
    if context.get('classical_trajectory') and name in {
            'algebraic_boundary_synthesis','affine_representation_search','output_basis_search',
            'architecture_portfolio','affine_transport_factorization','mixed_observable_search','consumer_functional_decomposition'}:
        return 'source has classical trajectory operations outside this constructor'
    if name == small and context.get('semantics',{}).get('reason') == 'reset_not_proven_zero':
        return 'joint value map unavailable because the boundary reset lacks a zero proof'
    if name == schedule and (context.get('joint_result') or {}).get('status') != 'constructed':
        return 'no exact product graph to schedule'
    if name == cleanup and all((context.get(k) or {}).get('status') != 'constructed'
                               for k in ('joint_result','algebraic_result')):
        return 'no verified product graph to lower'
    if name == small and len(context['clean_inputs']) < 2:
        return 'joint bank constructor requires two clean wires'
    if not features:
        return None
    if name in value and (features['sign'] or features['target_dependent']):
        return 'requires a phase-free preserved-input XOR update'
    if name == 'affine_transport_factorization' and (features['sign'] or not features['target_dependent']):
        return 'requires zero sign and a target-dependent update'
    if name == small and features['support'] > 8:
        return 'exact joint constructor exceeds eight supporting inputs'
    if name == 'output_basis_search' and features['dirty_targets'] < 2:
        return 'fewer than two dirty output accumulators'
    if name == 'architecture_portfolio' and features['changed'] < 2:
        return 'requires two changed outputs for a joint architecture witness'
    if name == 'consumer_functional_decomposition' and features['support'] < 3:
        return 'fewer than three supporting inputs'
    if name == 'mixed_observable_search' and not (features['sign'] and features['changed']):
        return 'requires both sign and changed values'
    if name == 'factored_observable_search' and (not features['sign'] or features['changed']):
        return 'requires a nonzero sign with all values restored'
    return None


def at_zero_floor(value, cost):
    return value == 0 and ''.join(cost.split()) in {
        'Q*toffoli','Q*(toffoli+t/4)','toffoli','Q**2*toffoli','instructions'}
