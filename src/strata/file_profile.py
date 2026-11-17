"""Native resource profiles that prioritize recovered algebraic regions.

The objective projection removes a region's additive static counts while
holding global width fixed.  It is a ceiling on savings from those counts under
a supported monotone objective, never an achievable saving or a semantic floor.
Native drawings and wire-access traces are supporting views, not liveness proofs.
"""
from __future__ import annotations

import ast
from collections import Counter
import html
import hashlib
import json
from pathlib import Path
import shlex

import kickmix as km

from .cost import Formula
from .kickmix import Kickmix
from .model import StrataError


NATIVE_ANALYSIS_MAX_OPS = 100_000
REGION_DIAGRAM_MAX_OPS = 100
ADDITIVE = ('toffoli', 't', 'rotations', 'instructions')


def _monotone_additive_objective(node):
    """A small sufficient certificate; other valid objectives remain unknown."""
    if isinstance(node, ast.Constant):
        return type(node.value) in (int, float) and node.value >= 0
    if isinstance(node, ast.Name):
        return node.id in {'Q', *ADDITIVE}
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, (ast.Add, ast.Mult)):
            return _monotone_additive_objective(node.left) and _monotone_additive_objective(node.right)
        if isinstance(node.op, ast.Div):
            return (_monotone_additive_objective(node.left) and isinstance(node.right, ast.Constant)
                    and type(node.right.value) in (int, float) and node.right.value > 0)
        if isinstance(node.op, ast.Pow):
            return (_monotone_additive_objective(node.left) and isinstance(node.right, ast.Constant)
                    and type(node.right.value) in (int, float) and 0 <= node.right.value <= 16)
    return False


def _write_evidence(lines, semantics):
    writes, instructions = Counter(), Counter()
    for line in lines:
        words = shlex.split(line)
        if not words:
            continue
        op = words[0]
        targets = words[-1:] if op in {'X', 'CX', 'CCX'} else words[1:] if op == 'SWAP' else []
        for target in targets:
            writes[int(target[1:])] += 1
        if targets:
            instructions[tuple(words)] += 1
    episodes = [episode for feature in semantics['features']
                if feature['kind'] == 'compute_consume_uncompute'
                for episode in feature['restored_wires']]
    return dict(materialization_count=len({e['wire'] for e in episodes}),
                materializations=episodes,
                repeated_write_count=sum(max(0, n - 1) for n in writes.values()),
                repeated_native_write_count=sum(max(0, n - 1) for n in instructions.values()),
                writes_by_wire={str(q): n for q, n in sorted(writes.items())},
                interpretation='Repeated target writes and repeated instructions are signals; '
                               'they do not independently prove recomputation or removable work.')


def _search_directions(region, evidence, phase_context):
    features = {f['kind'] for f in region['semantics']['features']}
    directions = []
    if evidence['materialization_count']:
        directions.append(dict(mechanism='remove_restored_materializations',
                               target='Apply inferred output and sign-phase functions directly; preserve arbitrary helper inputs.',
                               readiness='bounded_exact_semantics_available'))
    if features & {'bilinear_accumulator', 'quadratic_accumulator', 'polynomial_accumulator'}:
        directions.append(dict(mechanism='share_or_factor_accumulator_terms',
                               target='Compare common terms across output deltas under the complete static objective.',
                               readiness='derivation_and_equivalence_check_required'))
    if 'affine_transform' in features:
        directions.append(dict(mechanism='resynthesize_affine_map',
                               target='Preserve the complete binary matrix, affine offset, and exact sign phase.',
                               readiness='construction_and_equivalence_check_required'))
    if 'phase_polynomial' in features:
        directions.append(dict(mechanism='combine_exact_sign_phase_terms',
                               target='Use boundary-input phase polynomials while preserving output dependencies.',
                               readiness='construction_and_equivalence_check_required'))
    if phase_context['adjacent_opaque_barriers']:
        directions.append(dict(mechanism='resolve_opaque_context',
                               target='Establish separate semantics for adjacent conditions, measurements, or unsupported operations.',
                               readiness='unknown_context_not_eligible_for_cross_boundary_rewrite'))
    if not directions:
        directions.append(dict(mechanism='derive_structure_from_exact_map',
                               target='Identify useful dependencies in the complete bounded Boolean map.',
                               readiness='higher_level_structure_unidentified'))
    return directions


def profile_regions(circuit, lift_report, out, formula='Q*(toffoli+t/4)', max_graphs=6):
    """Return a JSON report and create native views plus ``profile.json``.

    Region identifiers and offsets are taken from ``lift_report`` unchanged.
    ``candidate_order`` supplies a direct optimizer priority list; callers must
    use the same lift budgets to recover those identifiers.  Only exact regions
    containing multiple operations enter that list.  Native overview analysis
    is capped at NATIVE_ANALYSIS_MAX_OPS; semantic coverage retains its own scope.
    """
    if type(max_graphs) is not int or max_graphs < 0:
        raise StrataError('max_graphs must be a nonnegative integer')
    if isinstance(circuit, str):
        circuit = km.Circuit(circuit)
    elif isinstance(circuit, Path):
        circuit = km.Circuit.from_file(circuit)
    formula = Formula(formula) if isinstance(formula, str) else formula
    if not isinstance(formula, Formula):
        raise StrataError('profiling requires a restricted Formula or expression')
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    frontend = Kickmix()
    global_resources = frontend.measured(circuit)
    global_cost = formula(global_resources)
    native_dir = out / 'native'
    native_dir.mkdir(exist_ok=True)
    # Reusing an output directory must not publish a previous run's graph when
    # this run skips or fails to generate it.
    for name in ('flame.svg', 'diagram.html', 'analysis.json', 'index.html'):
        (native_dir / name).unlink(missing_ok=True)
    native = frontend.analyze(circuit, native_dir, max_ops=NATIVE_ANALYSIS_MAX_OPS)
    native_links = ' · '.join(f'<a href="{name}">{label}</a>' for name, label in
                             [('analysis.json', 'Structured native analysis'),
                              ('flame.svg', 'SDK flame chart'), ('diagram.html', 'SDK diagram')]
                             if (native_dir / name).is_file())
    (native_dir / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Native analysis</title>'
                                          '<h1>Native analysis</h1><p>Static emitted metrics and bounded views; '
                                          'wire traces do not prove exact liveness.</p>' + native_links)
    trace = {row['at']: row['may_be_nonzero'] for row in native['liveness']['trace']}
    lines = str(circuit).splitlines()
    monotone = _monotone_additive_objective(formula.tree)
    barriers_at_start, barriers_at_stop = {}, {}
    for unknown in lift_report.get('unknown', []):
        summary = dict(start=unknown['start'], stop=unknown['stop'], reason=unknown.get('reason', 'unknown'))
        barriers_at_start.setdefault(unknown['start'], []).append(summary)
        barriers_at_stop.setdefault(unknown['stop'], []).append(summary)
    rows = []
    excluded_primitives = 0
    for region in lift_report['regions']:
        semantics = region.get('semantics', {})
        if semantics.get('status') != 'exact':
            continue
        if semantics['operation_count'] < 2:
            excluded_primitives += 1
            continue
        start, stop = region['start'], region['stop']
        if not (type(start) is int and type(stop) is int and 0 <= start < stop <= len(lines)):
            raise StrataError('profile region lies outside the supplied native circuit')
        selected = lines[start:stop]
        measured = frontend.measured(km.Circuit('\n'.join(selected)))
        static = {name: measured[name] for name in ADDITIVE}
        # Q is the same global resource in both evaluations.  A restored wire
        # is not a proof that global allocated width can be reduced.
        hypothetical = dict(global_resources)
        for name in ADDITIVE:
            hypothetical[name] = max(0, global_resources[name] - static[name])
        projection_error = None
        try:
            hypothetical_cost = formula(hypothetical)
            removal_delta = global_cost - hypothetical_cost
        except (StrataError, ArithmeticError) as exc:
            hypothetical_cost = removal_delta = None
            projection_error = str(exc)
        ceiling = max(0.0, removal_delta) if monotone and removal_delta is not None else None
        evidence = _write_evidence(selected, semantics)
        adjacent = barriers_at_stop.get(start, []) + barriers_at_start.get(stop, [])
        phase = semantics['phase']
        phase_context = dict(within_region='exact_sign_polynomial',
                             term_count=phase['term_count'], degree=phase['degree'],
                             input_dependencies=sorted({q for term in phase['monomials'] for q in term}, key=lambda value:(isinstance(value,str),str(value))),
                             adjacent_opaque_barriers=adjacent,
                             cross_boundary='unresolved_opaque_context' if adjacent else 'not_inferred_from_local_semantics')
        region_trace = [trace[i] for i in range(start, stop) if i in trace]
        row = dict(id=region['id'], start=start, stop=stop, kind=region['kind'],
                   features=[f['kind'] for f in semantics['features']],
                   boundary_basis=region.get('boundary_basis', 'unknown'),
                   boundary_width=semantics['boundary_width'],
                   modified_width=len(semantics['modified_wires']),
                   preserved_width=len(semantics['preserved_wires']),
                   static_resources=static,
                   static_paid_units=static['toffoli'] + static['t'] / 4,
                   local_reaction_depth=measured['reaction_depth'],
                   objective_removal_ceiling=ceiling,
                   removal_projection=dict(global_before=global_resources,
                                           global_after=hypothetical,
                                           cost_before=global_cost, cost_after=hypothetical_cost,
                                           hypothetical_delta=removal_delta,
                                           fixed_resources=['Q', 'reaction_depth'],
                                           status='monotone_fixed_resource_ceiling' if ceiling is not None else 'ceiling_unknown',
                                           reason=projection_error or (None if monotone else 'objective_monotonicity_or_nonadditive_dependency_unestablished')),
                   floor=None, floor_kind='unknown', achievable_saving=None,
                   phase_dependency=phase_context,
                   wire_signal=dict(native_reported_peak=max(region_trace) if region_trace else None,
                                    scope='full_region' if len(region_trace) == stop-start else 'prefix_or_unscanned',
                                    meaning='native may-nonzero heuristic; neither exact liveness nor a clean-lease proof'),
                   **evidence)
        row['search'] = _search_directions(region, evidence, phase_context)
        rows.append(row)
    rows.sort(key=lambda r: (-(r['objective_removal_ceiling'] or 0),
                             -r['static_paid_units'], -r['materialization_count'],
                             -r['repeated_write_count'], r['boundary_width'],
                             -(r['wire_signal']['native_reported_peak'] if r['wire_signal']['native_reported_peak'] is not None else -1),
                             len(r['phase_dependency']['adjacent_opaque_barriers']), r['start']))
    for rank, row in enumerate(rows, 1):
        row['priority'] = rank
    region_dir = out / 'regions'
    if max_graphs and rows:
        region_dir.mkdir(exist_ok=True)
    for row in rows[:max_graphs]:
        graph_stop = min(row['stop'], row['start'] + REGION_DIAGRAM_MAX_OPS)
        excerpt = km.Circuit('\n'.join(lines[row['start']:graph_stop]))
        path = region_dir / f'region-{row["priority"]:04d}.html'
        try:
            path.write_text(excerpt.html_diagram())
            row['diagram'] = dict(status='generated', path=str(path), start=row['start'], stop=graph_stop,
                                  coverage='full_region' if graph_stop == row['stop'] else 'prefix')
        except (ValueError, RuntimeError) as exc:
            row['diagram'] = dict(status='unavailable', reason=str(exc))
    views = {name: str(native_dir / filename) for name, filename in
             [('native_index', 'index.html'), ('native_analysis', 'analysis.json'),
              ('native_flame', 'flame.svg'), ('native_diagram', 'diagram.html')]
             if (native_dir / filename).is_file()}
    views.update(profile=str(out / 'profile.json'), index=str(out / 'index.html'))
    report = dict(schema_version='1', baseline=dict(resources=global_resources, cost=global_cost,
                                                   sha256=hashlib.sha256(str(circuit).encode()).hexdigest()),
                  objective=dict(expression=formula.expression, accounting='static_emitted',
                                 monotone_additive_ceiling_supported=monotone,
                                 ceiling_scope='Delete all region additive counts with global Q and reaction_depth fixed; '
                                               'not a realizable replacement, attainable saving, or semantic lower bound.'),
                  ranked_regions=rows, candidate_order=[row['id'] for row in rows],
                  priority_basis=['fixed-global-resource objective removal ceiling', 'static paid units',
                                  'restored intermediate materializations', 'repeated target writes',
                                  'smaller boundary width', 'larger native may-nonzero heuristic peak',
                                  'fewer adjacent opaque barriers'],
                  native_analysis=dict(coverage=native['coverage'], scanned=native['scanned'],
                                       operation_budget=NATIVE_ANALYSIS_MAX_OPS,
                                       reaction_depth=global_resources['reaction_depth'],
                                       wire_signal=dict(native_reported_peak=native['liveness']['peak'],
                                                        arbitrary_boundary_width=global_resources['Q'],
                                                        status='heuristic_not_exact_liveness',
                                                        note='Native may-nonzero trace uses register initialization and reset hints; '
                                                             'arbitrary input semantics and conditional context require separate reasoning.'),
                                       allocation_kind=native['allocation']['kind'],
                                       flame_status=native.get('flame', {}).get('status', 'generated')),
                  semantic_coverage=lift_report['coverage'], views=views,
                  excluded_primitive_regions=excluded_primitives,
                  unknowns=dict(opaque_or_bounded_regions=len(lift_report.get('unknown', [])),
                                high_level_stage_coverage=None, information_rank_floor=None,
                                clean_lease_proof=None, executed_cost=None,
                                phase_or_measurement_debt='unresolved_outside_exact_windows'),
                  limitations=[
                      'Exact local algebra does not identify complete arithmetic, decoder, or other task stages.',
                      'Static emitted counts do not establish executed cost or an official task score.',
                      'First/last wire access and may-nonzero traces do not prove exact liveness, '
                      'zero initialization, clean release, or reducible global width.',
                      'Repeated writes may be different updates or inverse operations; they are search signals.',
                      'Native flame and HTML views are auxiliary; candidate_order is the programmatic search input.'])
    (out / 'profile.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    links = ' · '.join(f'<a href="{html.escape(str(Path(path).relative_to(out)))}">{html.escape(name)}</a>'
                       for name, path in views.items() if name != 'index')
    table_rows = []
    for row in rows:
        label = html.escape(row['id'])
        if row.get('diagram', {}).get('status') == 'generated':
            relative = Path(row['diagram']['path']).relative_to(out)
            label = f'<a href="{html.escape(str(relative))}">{label}</a>'
        cells = [str(row['priority']), label] + [html.escape(str(value)) for value in
                [row['kind'], row['static_resources']['toffoli'], row['objective_removal_ceiling'],
                 row['materialization_count'], row['boundary_width']]]
        table_rows.append('<tr>' + ''.join(f'<td>{value}</td>' for value in cells) + '</tr>')
    table = ''.join(table_rows)
    (out / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Strata search profile</title>'
                                   '<h1>Ranked semantic regions</h1><p>Static resource removal ceilings are search priorities, '
                                   'not achievable savings. Global width is held fixed. Wire traces do not prove clean leases.</p>'
                                   f'<p>{links}</p><table><thead><tr><th>Priority</th><th>Region</th><th>Kind</th>'
                                   '<th>Toffoli-like</th><th>Objective removal ceiling</th><th>Materializations</th>'
                                   f'<th>Boundary width</th></tr></thead><tbody>{table}</tbody></table>')
    return report
