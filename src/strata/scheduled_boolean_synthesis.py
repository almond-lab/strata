"""Executable storage/recomputation schedules for complete XOR/AND witnesses."""
from fractions import Fraction
import shlex

import kickmix as km
import numpy as np

from .boolean_synthesis import _affine_update, _product_update, _layout
from .cost import Formula
from .floors import verify_minimum_witness
from .kickmix import Kickmix
from .model import StrataError, digest
from .pebbling import plan_reversible_pebbles


def scheduled_xor_and_witness(receipt, input_rows, output_rows, *, input_wires,
                               target_wires, clean_wires, num_qubits,
                               max_states=100000, max_operations=100000,
                               max_native_cells=1048576, symbolic_dag=None,
                               symbolic_reference=None, symbolic_clean_inputs=None, symbolic_limits=None):
    """Lower every feasible storage/paid-work frontier schedule to native gates.

    Products used by later products form the kernel DAG. Final products update
    arbitrary dirty targets directly; one observable-consumer step applies all
    output deltas before cleanup. Each schedule is independently native replayed
    on all supplied rows, both zero and nonzero target assignments. Targets are
    never controls, proving translation for every dirty target value.
    """
    if symbolic_dag is None:
        verify_minimum_witness(receipt, input_rows, output_rows)
    else:
        from .symbolic_boolean_graph import verify_symbolic_graph
        symbolic_limits = dict(max_variables=512, max_nodes=262144, max_apply_steps=1048576) | (symbolic_limits or {})
        verify_symbolic_graph(symbolic_reference, symbolic_dag,
                              clean_inputs=symbolic_clean_inputs, max_operations=max_operations,
                              max_variables=symbolic_limits['max_variables'], max_terms=symbolic_limits['max_nodes'],
                              max_product_terms=max(1, symbolic_limits['max_apply_steps'] // 32))
    if type(num_qubits) is not int or num_qubits < 0:
        raise StrataError('invalid synthesis boundary width')
    for name, value in [('max_operations', max_operations), ('max_native_cells', max_native_cells)]:
        if type(value) is not int or value <= 0: raise StrataError(f'{name} must be positive')
    inputs = _layout(input_wires, num_qubits, 'input wires')
    targets = _layout(target_wires, num_qubits, 'target wires')
    clean = sorted(_layout(clean_wires, num_qubits, 'clean wires'))
    if set(inputs) & set(targets) or set(inputs + targets) & set(clean):
        raise StrataError('scheduled inputs, targets and clean workspace must be disjoint')
    if symbolic_dag is None and (len(inputs) != len(input_rows[0]) or len(targets) != len(output_rows[0])):
        raise StrataError('scheduled wire layout does not match witness row widths')
    gates = receipt['witness'] if symbolic_dag is None else symbolic_dag['gates']
    output_masks = receipt['output_coefficients'] if symbolic_dag is None else symbolic_dag['output_coefficients']
    if symbolic_dag is not None and (inputs != symbolic_dag['input_wires'] or targets != symbolic_dag['target_wires']):
        raise StrataError('symbolic schedule wire layout differs from its DAG')
    offset = len(inputs) + 1
    materialized = {i for gate in gates for name in ('left_coefficients', 'right_coefficients')
                    for i in range(len(gates)) if gate[name] >> (offset + i) & 1}

    def referenced(mask):
        return {i for i in range(len(gates)) if mask >> (offset + i) & 1}

    def consumer(available):
        emitted = []
        for mask, target in zip(output_masks, targets):
            affine = mask
            for index in sorted(referenced(mask) - materialized):
                affine ^= 1 << (offset + index)
                gate = gates[index]
                emitted.extend(_product_update(gate['left_coefficients'], gate['right_coefficients'],
                                                available[:len(inputs) + index], target))
            emitted.extend(_affine_update(affine, available, target))
        return emitted

    # Dummy distinct wires price complete affine operand preparation and
    # restoration before planning. Actual slot bindings do not alter this bill.
    dummy = list(range(len(inputs) + len(gates)))
    kernels = []
    required = set()
    for mask in output_masks:
        required.update(referenced(mask) & materialized)
        for index in referenced(mask) - materialized:
            required.update(referenced(gates[index]['left_coefficients'] | gates[index]['right_coefficients']))
    for index in sorted(materialized):
        gate = gates[index]
        block = _product_update(gate['left_coefficients'], gate['right_coefficients'],
                                dummy[:len(inputs) + index], len(dummy))
        paid = sum(line.startswith('CCX ') for line in block)
        masks = gate['left_coefficients'] | gate['right_coefficients']
        deps = sorted(referenced(masks)) + [f'x{i}' for i in range(len(inputs)) if masks >> (i + 1) & 1]
        kernels.append(dict(id=index, dependencies=deps, width=1, compute_cost=paid, uncompute_cost=paid))
    dummy_consumer = consumer(dummy)
    consumer_cost = sum(line.startswith('CCX ') for line in dummy_consumer)
    public = set(range(num_qubits)) - set(clean)
    planning = plan_reversible_pebbles(kernels, sorted(required), inputs=[f'x{i}' for i in range(len(inputs))],
        input_width=len(inputs), fixed_width=len(public) - len(inputs), max_width=len(public) + len(clean),
        max_states=max_states, consume_cost=consumer_cost)
    alternatives, failures = [], []
    for planned in planning['alternatives']:
        needed = planned['peak_width'] - len(public)
        slots = clean[:needed]
        free, placement = set(slots), {}
        available = inputs + [None] * len(gates)
        emitted, ledger = [], []
        for action in planned['schedule']:
            if action['action'] == 'consume':
                block = consumer(available)
            else:
                index = action['node']; gate = gates[index]
                if action['action'] == 'compute':
                    if not free: raise StrataError('scheduled product storage exceeds its planned peak')
                    wire = min(free); free.remove(wire); placement[index] = wire
                    available[len(inputs) + index] = wire
                wire = placement[index]
                block = _product_update(gate['left_coefficients'], gate['right_coefficients'],
                                        available[:len(inputs) + index], wire)
                if action['action'] == 'uncompute':
                    block = list(reversed(block))
                    available[len(inputs) + index] = None
                    free.add(placement.pop(index))
            ledger.append(dict(action, start=len(emitted), stop=len(emitted) + len(block),
                               instructions=len(block), toffoli=sum(line.startswith('CCX ') for line in block)))
            emitted.extend(block)
            if len(emitted) > max_operations: break
        if len(emitted) > max_operations:
            failures.append(dict(reason='scheduled_native_operation_budget', peak_width=planned['peak_width']))
            continue
        if placement: raise StrataError('scheduled native emission leaves live product storage')
        retained = public | set(slots)
        physical_width = max(retained, default=-1) + 1
        native = km.Circuit('\n'.join(emitted))
        boundary_instructions = 0
        if native.num_qubits < physical_width:
            emitted.extend([f'X q{physical_width - 1}'] * 2)
            boundary_instructions = 2
            native = km.Circuit('\n'.join(emitted))
        if len(native) > max_operations:
            failures.append(dict(reason='scheduled_native_operation_budget', peak_width=planned['peak_width']))
            continue
        resources = Kickmix().measured(native)
        ledger_bill = dict(Q=physical_width, toffoli=sum(step['toffoli'] for step in ledger),
                           t=0, rotations=0,
                           instructions=sum(step['instructions'] for step in ledger) + boundary_instructions)
        if any(resources[name] != value for name, value in ledger_bill.items()):
            raise StrataError('scheduled kernel resource ledger disagrees with native resources')
        if Fraction(planned['cost']) != resources['toffoli']:
            raise StrataError('planned paid work disagrees with complete native schedule')
        # Validate with the SDK, not with the planner's Boolean transition rule.
        # Both runs cover every supplied input row; target translation extends
        # this finite native check to arbitrary initial dirty target words.
        if symbolic_dag is not None:
            from .symbolic_boolean_graph import verify_symbolic_native
            symbolic_check = verify_symbolic_native(symbolic_reference, native, clean_inputs=symbolic_clean_inputs,
                                                     max_operations=max_operations, max_native_cells=max_native_cells,
                                                     **symbolic_limits)
            if symbolic_check.get('equivalent') is not True:
                failures.append(dict(reason='symbolic_schedule_map_unproved', proof=symbolic_check)); continue
        shots = 0 if symbolic_dag is not None else len(input_rows)
        if max(1, num_qubits) * shots > max_native_cells:
            failures.append(dict(reason='scheduled_native_replay_cell_budget', requested_cells=num_qubits * shots))
            continue
        for dirty in (() if symbolic_dag is not None else (0, 1)):
            sim = km.Simulator(batch_size=shots, seed=983)
            expected = {q: np.zeros(shots, dtype=bool) for q in range(num_qubits)}
            for position, q in enumerate(inputs):
                values = np.array([row[position] for row in input_rows], dtype=bool)
                sim.write_across_shots(km.q(q), values); expected[q] = values
            for position, q in enumerate(targets):
                initial = np.full(shots, bool(dirty))
                sim.write_across_shots(km.q(q), initial)
                expected[q] = initial ^ np.array([row[position] for row in output_rows], dtype=bool)
            sim.do(native)
            if any(not np.array_equal(sim.read_across_shots(km.q(q)), value) for q, value in expected.items()):
                raise StrataError('scheduled native full-boundary replay mismatch')
            if sim.read_phase_flipped_across_shots().any():
                raise StrataError('scheduled native phase replay mismatch')
        alternatives.append(dict(status='constructed', candidate=str(native), resources=resources,
            schedule=planned, kernel_ledger=ledger, resource_ledger=ledger_bill,
            ledger_matches_native=True, used_clean_wires=slots,
            omitted_clean_wires=[q for q in clean if q not in slots],
            width_scope='physical Q includes public-index gaps; planned peak counts live values and may be smaller',
            proof=symbolic_check if symbolic_dag is not None else dict(status='verified', method='independent_kickmix_full_boundary_replay',
                       input_rows=len(input_rows), target_replays=2,
                       dirty_targets='arbitrary by XOR-only writes; target wires are never controls',
                       workspace_restored=True, phase_zero=True,
                       scope='supplied witness rows; source-boundary equality is a separate obligation')))
    return dict(status='constructed' if alternatives else 'unknown',
                reason=None if alternatives else 'no complete executable schedule within budgets',
                family='scheduled_joint_xor_and', alternatives=alternatives, planning=planning,
                failures=failures, kernel_dag=kernels, consumer_products=consumer_cost,
                native_frontier_complete=planning['pareto_complete'] and not failures,
                frontier_scope='logical live width and Toffoli count for this fixed witness DAG and lowering',
                claim_scope='complete Boolean-kernel storage and recomputation schedules for the supplied joint function witness',
                no_global_algorithmic_optimality_claim=True)


def scheduled_boundary_synthesis(circuit, *, joint_result=None, cost='Q*(toffoli+t/4)',
                                 resource_measure=None, max_schedule_states=100000,
                                 max_native_cells=1048576, **options):
    """Construct and select a host-priced schedule for a recovered whole boundary."""
    from .boolean_synthesis import joint_boundary_synthesis
    from .inference import functions_at
    if joint_result is None: joint_result = joint_boundary_synthesis(circuit, **options)
    if joint_result.get('status') != 'constructed':
        return dict(status='unknown', reason='joint boundary synthesis unavailable', joint_synthesis=joint_result)
    contract = joint_result['synthesis_contract']
    emitted = scheduled_xor_and_witness(joint_result['exact_search'], contract['input_rows'], contract['output_rows'],
        **{key: contract[key] for key in ('input_wires', 'target_wires', 'clean_wires', 'num_qubits')},
        max_states=max_schedule_states, max_operations=options.get('max_operations', 10000),
        max_native_cells=max_native_cells)
    if emitted['status'] != 'constructed': return emitted
    clean = set(joint_result['proof']['clean_inputs'])
    limits = {name: options[name] for name in ('max_operations', 'max_variables', 'max_terms', 'max_product_terms') if name in options}
    original = functions_at(circuit, clean_inputs=clean, **limits)
    if original['status'] != 'exact':
        return dict(status='unknown', reason='scheduled source replay unavailable', semantics=original)
    formula = Formula(cost) if isinstance(cost, str) else cost
    valid, symbolic_failures = [], []
    for alternative in emitted['alternatives']:
        omitted = set(alternative['omitted_clean_wires'])
        metadata = []
        for line in str(circuit).splitlines():
            words = shlex.split(line, comments=True)
            if not words or words[0] not in {'REGISTER', 'APPEND_TO_REGISTER', 'INFO_TAG'}: continue
            if words[0] == 'APPEND_TO_REGISTER' and any(w == f'q{q}' for q in omitted for w in words): continue
            metadata.append(line)
        native = km.Circuit('\n'.join(metadata + [alternative['candidate']]))
        replay = functions_at(native, clean_inputs=clean & set(range(native.num_qubits)), **limits)
        if replay['status'] != 'exact':
            symbolic_failures.append(dict(reason='scheduled_candidate_symbolic_budget', semantics=replay))
            continue
        if replay['phase'] != original['phase'] or any(replay['functions'].get(q, set()) != value
                                                       for q, value in original['functions'].items()):
            raise StrataError('scheduled whole-boundary symbolic replay mismatch')
        resources = Kickmix().measured(native)
        objective_resources = resource_measure(native) if resource_measure is not None else resources
        alternative.update(candidate=str(native), resources=resources, objective_resources=objective_resources,
                           cost=formula(objective_resources),
                           whole_boundary_check=dict(status='equivalent',
                               method='exact_full_quantum_boundary_ANF_and_sign_replay',
                               scope=joint_result['proof']['scope']))
        # Native metadata has its SDK-defined weight (register declarations are
        # not executable instructions), rather than a text-line count.
        alternative['resource_ledger']['instructions'] += len(km.Circuit('\n'.join(metadata)))
        if any(resources[name] != value for name, value in alternative['resource_ledger'].items()):
            raise StrataError('scheduled full metadata ledger mismatch')
        valid.append(alternative)
    if not valid: return dict(status='unknown', reason='scheduled candidate symbolic replay budgets', scheduling=emitted)
    best = min(valid, key=lambda row: (row['cost'], row['resources']['Q'], row['resources']['toffoli']))
    structural = dict(joint_result.get('structural_witness', {}))
    restored = {q for q in clean if not original['functions'][q]}
    retired = sorted(set(structural.get('materialized_clean_wires', ())) & restored - set(best['used_clean_wires']))
    structural.update(retired_scratch_wires=retired, retained_product_wires=best['used_clean_wires'],
                      final_products_materialized=False,
                      boundary_outputs_synthesized_jointly=contract['target_wires'],
                      schedule_recomputations=best['schedule']['recomputations'],
                      removed_materialization_bank=len(retired) >= 2 and structural.get('source_nonlinear_dataflow_depth', 0) >= 2)
    return dict(status='constructed', candidate=best['candidate'], family='scheduled_joint_xor_and',
                alternatives=valid, planning=emitted['planning'], kernel_dag=emitted['kernel_dag'],
                selected_cost=best['cost'], objective=cost if isinstance(cost, str) else cost.expression,
                selection_scope='lowest host objective among emitted alternatives; arbitrary objectives are not optimized over every possible schedule',
                native_frontier_complete=emitted['native_frontier_complete'] and not symbolic_failures,
                failures=emitted['failures'] + symbolic_failures,
                structural_witness=structural,
                proof=best['whole_boundary_check'], resource_bill=best['resource_ledger'],
                claim_scope=emitted['claim_scope'], no_global_algorithmic_optimality_claim=True,
                objective_scope='parent_context' if resource_measure is not None else 'supplied_circuit',
                witness_hash=digest(dict(source=str(circuit), selected=best['candidate'],
                                         clean=sorted(clean), search=joint_result['exact_search']['contract_hash'])))
