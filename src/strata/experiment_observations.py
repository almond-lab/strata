"""Append-only correction of infrastructure failures, never scientific verdicts."""
from __future__ import annotations

from .model import StrataError


def infrastructure_evidence(store, finish):
    verifier = finish.get('verifier')
    if isinstance(verifier, str):
        verifier = store.get(verifier)
    if not isinstance(verifier, dict):
        return None
    receipt = verifier.get('receipt', {})
    if not isinstance(receipt, dict):
        receipt = {}
    if verifier.get('returncode') not in (None, 0):
        return dict(kind='verifier_process_exit', returncode=verifier['returncode'])
    # A historical wrapper may have returned exit zero around a failed child.
    # Require structured machine evidence; prose alone cannot erase a rejection.
    for source in (receipt, receipt.get('process', {}), receipt.get('error', {})):
        if not isinstance(source, dict):
            continue
        code = source.get('returncode')
        if type(code) is int and code < 0:
            return dict(kind='child_signal', returncode=code)
        status = source.get('status', source.get('kind'))
        if status in {'ERROR', 'TIMEOUT', 'OOM', 'infrastructure_error',
                      'infrastructure_failure', 'resource_exhausted', 'timed_out',
                      'out_of_memory', 'cpu_limit', 'signal'}:
            return dict(kind='structured_infrastructure_failure', status=status)
    return None


def invalidate_verification(store, result, *, actor, reason):
    """Retain a correction linked to a machine-backed failed verification.

    This removes no bytes or receipts and supplies no replacement verdict. The
    same artifact/draw must still receive an accepted, independently bound check.
    """
    if any(not isinstance(v, str) or not v.strip() for v in (actor, reason)):
        raise StrataError('invalidation requires nonempty actor and reason')
    finish = store.get(result, 'experiment_finish')
    if finish.get('status') == 'accepted_by_verifier':
        raise StrataError('cannot invalidate an accepted scientific observation')
    evidence = infrastructure_evidence(store, finish)
    if evidence is None:
        raise StrataError('invalidation requires retained machine evidence of infrastructure failure; scientific rejection is not overridable')
    value = dict(result=result, experiment=finish['experiment'], actor=actor,
                 reason=reason, evidence=evidence,
                 scope='infrastructure correction only; original retained, replacement verification required')
    return dict(record=store.put('experiment_verification_invalidation', value), **value)
