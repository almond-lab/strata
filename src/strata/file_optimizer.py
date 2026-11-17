"""Adapter-free restructuring of bounded permutation/sign-phase regions.

The input circuit supplies the behavior oracle.  Exact Boolean ANF supplies a
candidate's *entire* output function, rather than a local gate-pattern rewrite.
The implemented construction removes restored internal materializations when
their effects can be applied directly to outputs and phase.  Every boundary
wire, including apparent scratch, retains an arbitrary initial value.
"""
from __future__ import annotations

import hashlib
import heapq
import shlex

import kickmix as km

from .cost import Formula
from .kickmix import Kickmix
from .model import StrataError
from .semantic import ARITY, equivalent_windows, lift_regions


DEFAULT_OBJECTIVE = 'Q*(toffoli+t/4)'


def _hash(circuit):
    return hashlib.sha256(str(circuit).encode()).hexdigest()


def _written(lines):
    written = set()
    for line in lines:
        words = shlex.split(line)
        if not words:
            continue
        if words[0] in {'X', 'CX', 'CCX'}:
            written.add(int(words[-1][1:]))
        elif words[0] == 'SWAP':
            written.update(int(q[1:]) for q in words[1:])
    return written


def _resynthesize(semantics, *, max_operations):
    """Apply phase on original inputs, then topologically ordered XOR updates.

    For an update t ^= F(inputs), t must run before every changed input read by
    F.  A cycle, a self-dependent update, or a degree beyond available native
    gates is rejected.  Thus later updates never read an overwritten original.
    """
    if semantics.get('measurement_variables') or semantics.get('classical_inputs') or semantics.get('classical_outputs'):
        return None, 'constructor_requires_quantum_only_permutation'
    if semantics.get('is_permutation') is False:
        return None, 'constructor_requires_permutation'
    changes = {int(q): p['monomials'] for q, p in semantics['changes'].items()}
    phase = semantics['phase']['monomials']
    if any(len(t) > 3 for t in phase):
        return None, 'phase_degree_exceeds_three'
    if any(len(t) > 2 for terms in changes.values() for t in terms):
        return None, 'output_delta_degree_exceeds_two'
    if any(q in term for q, terms in changes.items() for term in terms):
        return None, 'output_delta_depends_on_own_original'

    edges = {q: set() for q in changes}
    incoming = {q: 0 for q in changes}
    for q, terms in changes.items():
        for other in set(v for term in terms for v in term) & changes.keys():
            edges[q].add(other)
            incoming[other] += 1
    ready = [q for q in incoming if incoming[q] == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        q = heapq.heappop(ready)
        order.append(q)
        for other in sorted(edges[q]):
            incoming[other] -= 1
            if incoming[other] == 0:
                heapq.heappush(ready, other)
    if len(order) != len(changes):
        return None, 'cyclic_original_input_dependencies'

    count = len(phase) + sum(map(len, changes.values()))
    if count > max_operations:
        return None, 'candidate_operation_budget'
    emitted = []
    for term in phase:
        op = ('NEG', 'Z', 'CZ', 'CCZ')[len(term)]
        emitted.append(' '.join([op] + [f'q{q}' for q in term]))
    for q in order:
        for term in changes[q]:
            op = ('X', 'CX', 'CCX')[len(term)]
            emitted.append(' '.join([op] + [f'q{v}' for v in term] + [f'q{q}']))
    return emitted, None


def _removed_materializations(semantics, emitted):
    remaining = _written(emitted)
    return [dict(episode, initial_value_preserved=True)
            for feature in semantics['features']
            if feature['kind'] == 'compute_consume_uncompute'
            for episode in feature['restored_wires']
            if episode['wire'] not in remaining]


def _limits(max_operations, max_terms, max_wires):
    values = dict(max_operations=max_operations, max_terms=max_terms, max_variables=max_wires)
    if any(type(v) is not int or v <= 0 for v in values.values()):
        raise StrataError('optimizer budgets must be positive integers')
    return values


def optimize_circuit(circuit, formula=None, *, max_operations=256,
                     max_terms=4096, max_wires=64, region_order=None,
                     max_candidates=None, clean_inputs=(),accounting='static',external_distribution=None):
    """Return ``(candidate Circuit, artifact-bound JSON report)``.

    ``circuit`` is native text or a loaded kickmix Circuit; file ingestion uses
    ``Kickmix().load(path)``.  ``formula`` is a Formula or restricted expression
    over emitted resource counts.  Disjoint exact regions are considered in
    file order, with each substitution required to lower the complete current
    circuit's objective.  ``region_order`` ranks region IDs for a bounded search;
    the first ``max_candidates`` cheaply eligible functions are selected and
    then substituted in file order.  Unlisted IDs follow in file order, and IDs
    absent from this lift are ignored.  Metadata, opaque operations, measurement and condition
    boundaries remain byte-for-byte unchanged in canonical native text.
    """
    limits = _limits(max_operations, max_terms, max_wires)
    if max_candidates is not None and (type(max_candidates) is not int or max_candidates < 0):
        raise StrataError('max_candidates must be a nonnegative integer or None')
    if region_order is not None and (isinstance(region_order, (str, bytes)) or
                                     not isinstance(region_order, (list, tuple)) or
                                     any(not isinstance(r, str) for r in region_order)):
        raise StrataError('region_order must be a sequence of region IDs')
    if isinstance(circuit, str):
        circuit = km.Circuit(circuit)
    clean_inputs = tuple(sorted(set(clean_inputs)))
    if any(type(q) is not int or not 0 <= q < circuit.num_qubits for q in clean_inputs):
        raise StrataError('clean inputs must be physical wires of the supplied circuit')
    formula = Formula(formula or DEFAULT_OBJECTIVE) if isinstance(formula, (str, type(None))) else formula
    if not isinstance(formula, Formula):
        raise StrataError('file optimization requires a restricted Formula or expression')
    frontend = Kickmix()
    from .accounting import measured_resources
    measure=lambda c:measured_resources(c,accounting=accounting,external_distribution=external_distribution)
    baseline_resources = measure(circuit)
    baseline_cost = formula(baseline_resources)
    original_lines = str(circuit).splitlines()
    candidate_lines = list(original_lines)
    current_resources, current_cost = baseline_resources, baseline_cost
    lifted = lift_regions(circuit, max_window=max_operations,
                          max_variables=max_wires, max_terms=max_terms, clean_inputs=clean_inputs)
    prepared = {}
    for region in lifted['regions']:
        semantics = region['semantics']
        emitted, reason = _resynthesize(semantics, max_operations=max_operations)
        removed = _removed_materializations(semantics, emitted) if emitted is not None else []
        if reason is None and not removed:
            reason = 'no_internal_materialization_removed'
        if reason is None and not semantics['changes'] and not semantics['phase']['monomials']:
            reason = 'materialization_has_no_proven_net_boundary_effect'
        prepared[region['id']] = dict(emitted=emitted, removed=removed, reason=reason)
    eligible = [rid for rid, data in prepared.items() if data['reason'] is None]
    eligible_set = set(eligible)
    priority = list(dict.fromkeys([rid for rid in (region_order or []) if rid in eligible_set] + eligible))
    selected = set(priority if max_candidates is None else priority[:max_candidates])
    ranks = {rid: i for i, rid in enumerate(priority)}
    attempted_order = []
    rows, offset = [], 0
    for region in lifted['regions']:
        start, stop, semantics = region['start'], region['stop'], region['semantics']
        row = dict(id=region['id'], start=start, stop=stop, kind=region['kind'],
                   status='skipped', structural_change=False)
        rows.append(row)
        plan = prepared[region['id']]
        emitted, reason, removed = plan['emitted'], plan['reason'], plan['removed']
        if reason is not None:
            row['reason'] = reason
            continue
        row['selection_rank'] = ranks[region['id']]
        if region['id'] not in selected:
            row.update(status='deferred', reason='candidate_budget', disposition='open',
                       note='Eligible inferred function not attempted under this search budget.')
            continue
        attempted_order.append(region['id'])
        before = original_lines[start:stop]
        boundary_clean = tuple(sorted(map(int, region.get('boundary_zero_evidence', {}))))
        row['boundary_zero_evidence'] = region.get('boundary_zero_evidence', {})
        omitted_wires = {episode['wire'] for episode in removed}
        without_materialization = [line for line in before
                                   if not (_written([line]) & omitted_wires)]
        contribution = equivalent_windows(before, without_materialization, clean_inputs=boundary_clean, **limits)
        if contribution['equivalent'] is not False:
            row['reason'] = 'materialization_has_no_proven_net_boundary_effect'
            continue
        row.update(mechanism='resynthesize_output_functions_and_phase',
                   removed_materializations=removed,
                   materialization_contribution=contribution,
                   inferred=dict(changes=semantics['changes'], phase=semantics['phase'],
                                 assumptions=semantics['assumptions']))
        check = equivalent_windows(before, emitted, clean_inputs=boundary_clean, **limits)
        row['check'] = check
        if check['equivalent'] is not True:
            row['reason'] = 'exact_equivalence_not_established'
            continue
        at = start + offset
        trial_lines = candidate_lines[:at] + emitted + candidate_lines[stop + offset:]
        trial = km.Circuit('\n'.join(trial_lines))
        if trial.num_qubits != circuit.num_qubits:
            row['reason'] = 'boundary_width_would_change'
            continue
        measured = measure(trial)
        trial_cost = formula(measured)
        row['before'] = dict(resources=current_resources, cost=current_cost,
                             region_operations=len(before))
        row['after'] = dict(resources=measured, cost=trial_cost,
                            region_operations=len(emitted))
        if trial_cost >= current_cost:
            row['reason'] = 'static_objective_did_not_decrease'
            continue
        row.update(status='accepted', structural_change=True,
                   candidate_start=at, candidate_stop=at + len(emitted),
                   source_sha256=_hash('\n'.join(before)),
                   replacement_sha256=_hash('\n'.join(emitted)),
                   explanation='Removed writes that materialized and restored intermediate values; '
                               'applied their inferred effects directly to output XOR updates and sign phase. '
                               'Original intermediate input values remain part of the exact function.')
        candidate_lines = trial_lines
        offset += len(emitted) - (stop - start)
        current_resources, current_cost = measured, trial_cost

    candidate = km.Circuit('\n'.join(candidate_lines))
    report = dict(schema_version='1',
                  clean_inputs=list(clean_inputs),
                  status='improved' if current_cost < baseline_cost else 'unchanged',
                  baseline=dict(sha256=_hash(circuit), resources=baseline_resources, cost=baseline_cost),
                  candidate=dict(sha256=_hash(candidate), resources=current_resources, cost=current_cost),
                  objective=dict(expression=formula.expression, accounting=baseline_resources['accounting'],
                                 external_distribution=external_distribution,
                                 authority='native SDK resource counts; not an official task score'),
                  regions=rows, coverage=lifted['coverage'], opaque=lifted['unknown'],
                  metadata=lifted['metadata'], limits=limits,
                  selection_priority=priority, attempted_order=attempted_order,
                  search_budget=dict(max_candidates=max_candidates, eligible=len(eligible),
                                     attempted=len(attempted_order), deferred=len(eligible)-len(selected),
                                     policy='priority_selection_then_original_file_order'),
                  limitations=[
                      'A bounded semantic restructuring subset, not universal circuit optimization.',
                      'Only affine/quadratic XOR updates with acyclic original-input dependencies '
                      'and sign-phase polynomials of degree at most three are constructed.',
                      'No clean ancillas, width reduction, approximate phases, measurement debt '
                      'elimination, or domain acceptance assumptions are inferred.',
                      'Opaque operations and classical condition scopes are preserved; '
                      'exact regional substitutions compose across them.',
                      'Strict improvements are selected greedily under the supplied static objective; '
                      'no optimality or official task score is claimed.'])
    report['checks'] = verify_optimization(circuit, candidate, report, formula=formula, **dict(
        max_operations=max_operations, max_terms=max_terms, max_wires=max_wires, clean_inputs=clean_inputs,
        accounting=accounting,external_distribution=external_distribution))
    if report['checks']['equivalent'] is not True:
        raise StrataError('internal file optimization replay check failed: ' + report['checks']['reason'])
    return candidate, report


def verify_optimization(original, candidate, report, *, formula=None, max_operations=256,
                        max_terms=4096, max_wires=64, clean_inputs=(),accounting='static',external_distribution=None):
    """Recheck artifact binding and the proof of every accepted substitution.

    Report success flags are never trusted.  This verifies equality of all
    untouched slices and exact all-input state and phase equality of each
    changed slice.  Emitted resources and objective values are recomputed too.
    Supply ``formula`` to pin a replay's objective or provide configured Formula
    functions.  It does not certify a domain specification or official score.
    """
    limits = _limits(max_operations, max_terms, max_wires)
    if isinstance(original, str):
        original = km.Circuit(original)
    if isinstance(candidate, str):
        candidate = km.Circuit(candidate)
    clean_inputs = tuple(sorted(set(clean_inputs)))
    from .accounting import measured_resources
    measure=lambda c:measured_resources(c,accounting=accounting,external_distribution=external_distribution)
    base_hash, candidate_hash = _hash(original), _hash(candidate)
    result = dict(equivalent=False, kind='exact_symbolic_disjoint_substitution',
                  baseline_sha256=base_hash, candidate_sha256=candidate_hash,
                  all_inputs_arbitrary=not bool(clean_inputs), clean_inputs=list(clean_inputs), phase='exact', width_preserved=False,
                  boundaries_preserved=False, checked_regions=0)

    def fail(reason):
        return dict(result, reason=reason)

    try:
        if list(clean_inputs) != report.get('clean_inputs', []):
            return fail('clean_input_promise_mismatch')
        if (report['baseline']['sha256'] != base_hash or
                report['candidate']['sha256'] != candidate_hash):
            return fail('artifact_hash_mismatch')
        if original.num_qubits != candidate.num_qubits:
            return fail('boundary_width_changed')
        result['width_preserved'] = True
        old, new = str(original).splitlines(), str(candidate).splitlines()
        eligible = lift_regions(original, max_window=max_operations,
                                max_variables=max_wires, max_terms=max_terms, clean_inputs=clean_inputs)['regions']
        old_at = new_at = 0
        for row in report['regions']:
            if row['status'] != 'accepted':
                continue
            start, stop = row['start'], row['stop']
            at, end = row['candidate_start'], row['candidate_stop']
            if any(type(i) is not int for i in (start, stop, at, end)):
                return fail('invalid_region_indices')
            if not (old_at <= start < stop <= len(old) and new_at <= at <= end <= len(new)):
                return fail('overlapping_or_invalid_regions')
            if old[old_at:start] != new[new_at:at]:
                return fail('outside_region_changed')
            region = next((r for r in eligible if r['start'] == start and r['stop'] == stop), None)
            if region is None:
                return fail('region_crosses_opaque_or_metadata_boundary')
            boundary_evidence = region.get('boundary_zero_evidence', {})
            if row.get('boundary_zero_evidence', {}) != boundary_evidence:
                return fail('boundary_zero_evidence_mismatch')
            before, after = old[start:stop], new[at:end]
            if any(not shlex.split(line) or shlex.split(line)[0] not in ARITY for line in after):
                return fail('replacement_contains_unsupported_or_metadata_operation')
            if (row['source_sha256'] != _hash('\n'.join(before)) or
                    row['replacement_sha256'] != _hash('\n'.join(after))):
                return fail('region_hash_mismatch')
            check = equivalent_windows(before, after, clean_inputs=tuple(sorted(map(int, boundary_evidence))), **limits)
            if check['equivalent'] is not True:
                return fail('region_equivalence_not_established')
            old_at, new_at = stop, end
            result['checked_regions'] += 1
        if old[old_at:] != new[new_at:]:
            return fail('outside_region_changed')
        if (measure(original) != report['baseline']['resources'] or
                measure(candidate) != report['candidate']['resources']):
            return fail('resource_measurement_mismatch')
        if (report['objective'].get('external_distribution') != external_distribution or
                report['objective']['accounting'] != report['baseline']['resources']['accounting']):
            return fail('accounting_model_mismatch')
        objective = (Formula(formula or report['objective']['expression'])
                     if isinstance(formula, (str, type(None))) else formula)
        if (not isinstance(objective, Formula) or
                objective.expression != report['objective']['expression']):
            return fail('objective_mismatch')
        baseline_cost = objective(report['baseline']['resources'])
        candidate_cost = objective(report['candidate']['resources'])
        if (baseline_cost != report['baseline']['cost'] or
                candidate_cost != report['candidate']['cost']):
            return fail('objective_value_mismatch')
        if candidate_cost > baseline_cost:
            return fail('objective_increased')
        expected_status = 'improved' if candidate_cost < baseline_cost else 'unchanged'
        if report['status'] != expected_status:
            return fail('improvement_status_mismatch')
    except (KeyError, TypeError, ValueError, SyntaxError, ArithmeticError):
        return fail('malformed_report')
    result.update(equivalent=True, boundaries_preserved=True, reason='verified')
    return result
