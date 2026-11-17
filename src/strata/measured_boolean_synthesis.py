"""Measured cleanup of certified product nodes in complete Boolean schedules.

Only newly allocated classical outcomes are compiler-private. Changing existing
classical outputs still requires an explicit private-bit contract. Candidate
comparison is an independent basis/sign map proof, never a coupled-outcome proof.
"""
from fractions import Fraction
import shlex

import kickmix as km
import numpy as np

from .accounting import measured_resources
from .boolean_synthesis import _affine_update, _product_update, _layout
from .factored import DecisionBudget
from .factored_semantic import _prepare
from .floors import verify_minimum_witness
from .kickmix import Kickmix
from .model import StrataError, digest
from .cost import Formula
from .pebbling import plan_reversible_pebbles
from .scheduled_boolean_synthesis import scheduled_xor_and_witness


def _cleanup_template(left, right, available, target, bit):
    """Use the exact operand preparation already used by native product lowering."""
    if (any(type(mask) is not int or mask < 0 or mask >> (len(available) + 1) for mask in (left, right))
            or type(target) is not int or target < 0):
        raise StrataError('invalid measured product operand mask or target')
    selected = [wire for index, wire in enumerate(available) if (left | right) >> (index + 1) & 1]
    if any(type(wire) is not int or wire < 0 for wire in selected):
        raise StrataError('measured cleanup operand is unavailable')
    if len(selected) != len(set(selected)):
        raise StrataError('measured cleanup operand wires alias')
    if target in {q for q in available if q is not None}:
        raise StrataError('measured product target aliases an operand wire')
    block = _product_update(left, right, available, target)
    products = [i for i, line in enumerate(block) if line.startswith('CCX ')]
    if len(products) != 1:
        return dict(mode='reverse', lines=list(reversed(block)), reason='affine_or_constant_product')
    index = products[0]
    before, after = block[:index], block[index + 1:]
    if after != list(reversed(before)) or any(line.split()[0] not in {'X', 'CX'} for line in before):
        return dict(mode='reverse', lines=list(reversed(block)), reason='unsupported_operand_preparation')
    words = block[index].split()
    if words[-1] != f'q{target}':
        raise StrataError('product kernel target disagrees with cleanup target')
    repair = f'CZ {words[1]} {words[2]} if b{bit}'
    return dict(mode='hmr', lines=[f'HMR q{target} b{bit}', *before, repair, *after],
                repair=repair, prepared_controls=words[1:3],
                reason='HMR phase is repaid by the same current affine operand product')


def prove_product_cleanup(prefix, *, left, right, available, target, bit,
                          clean_inputs=(), max_operations=100000, max_variables=512,
                          max_nodes=262144, max_apply_steps=1048576):
    """Require the actual current target value to equal the current operand product.

    This proof rejects dirty initial targets, overwritten parents, aliases and
    unavailable operands. A refusal is not permission to measure. The ordinary
    inverse kernel remains a possible fallback, subject to its own final check.
    """
    if type(bit) is not int or bit < 0:
        raise StrataError('cleanup classical bit must be nonnegative')
    template = _cleanup_template(left, right, available, target, bit)
    if template['mode'] != 'hmr': return dict(status='not_applicable', **template)
    native = prefix if isinstance(prefix, km.Circuit) else km.Circuit(prefix)
    if bit < native.num_bits:
        return dict(status='not_applicable', reason='cleanup outcome bit is not fresh')
    dag, state, clean = _prepare(str(native).splitlines(), clean_inputs, max_operations,
                                  max_variables, max_nodes, max_apply_steps)
    if 'status' in state:
        return dict(status='unknown', reason=state['reason'], apply_steps=0 if dag is None else dag.steps)
    if target not in state['state']:
        return dict(status='not_applicable', reason='product target has no established computed value')
    try:
        def value(mask):
            result = int(bool(mask & 1))
            for index, wire in enumerate(available, 1):
                if mask >> index & 1:
                    if wire is None or wire not in state['state']:
                        raise StrataError('cleanup operand has no current value')
                    result = dag.xor(result, state['state'][wire])
            return result
        product = dag.product(value(left), value(right))
        if product != state['state'][target]:
            return dict(status='not_applicable', reason='target_not_current_operand_product',
                        witness={str(q): v for q, v in dag.witness(dag.xor(product, state['state'][target])).items()})
    except (DecisionBudget, StrataError) as exc:
        return dict(status='unknown', reason=str(exc), apply_steps=dag.steps)
    return dict(status='proved', **template, proof='canonical current target equals current affine operand product',
                apply_steps=dag.steps, decision_nodes=len(dag.nodes), clean_inputs=sorted(clean))


def verify_measured_witness_candidate(candidate, receipt, input_rows, output_rows, *,
                                       input_wires, target_wires, clean_wires, num_qubits,
                                       internal_bits=(), max_operations=100000, max_variables=512,
                                       max_nodes=262144, max_apply_steps=1048576,
                                       max_native_cells=1048576):
    """Prove outcome-independent final values/sign and cross-check native replay.

    Truth rows bind the desired function. Canonical decisions independently
    prove zero phase, restored workspace, no dependence on formal outcomes, and
    arbitrary dirty-target translation. Every nonprivate classical output must
    preserve its arbitrary initial value.
    """
    verify_minimum_witness(receipt, input_rows, output_rows)
    native = candidate if isinstance(candidate, km.Circuit) else km.Circuit(candidate)
    if native.num_qubits > num_qubits:
        return dict(status='different', equivalent=False, reason='expanded_quantum_boundary')
    hidden = {f'b{b}' for b in internal_bits}
    dag, state, clean = _prepare(str(native).splitlines(), clean_wires, max_operations,
                                  max_variables, max_nodes, max_apply_steps)
    if 'status' in state: return dict(status='unknown', equivalent=None, reason=state['reason'])
    if state['phase']:
        return dict(status='different', equivalent=False, reason='measurement_phase_not_repaired',
                    counterexample={str(q): v for q, v in dag.witness(state['phase']).items()})
    try:
        deltas = {}
        for q, value in state['state'].items():
            if q in target_wires:
                delta = dag.xor(value, state['initial'][q])
                if not dag.support(delta) <= set(input_wires):
                    return dict(status='different', equivalent=False, reason='target_delta_depends_on_outcome_or_dirty_target')
                deltas[q] = delta
            elif value != state['initial'][q]:
                return dict(status='different', equivalent=False, reason='workspace_or_public_value_not_restored', wire=q)
        for bit, value in state['classical'].items():
            if bit not in hidden and value != dag.variable(bit):
                return dict(status='different', equivalent=False, reason='visible_classical_output_changed', bit=bit)
        for row, expected in zip(input_rows, output_rows):
            assignment = dict(zip(input_wires, row))
            actual = [dag.evaluate(deltas.get(q, 0), assignment) for q in target_wires]
            if actual != list(expected):
                return dict(status='different', equivalent=False, reason='witness_output_mismatch',
                            counterexample=dict(inputs=assignment, actual=actual, expected=expected))
    except DecisionBudget as exc:
        return dict(status='unknown', equivalent=None, reason=str(exc))
    cells = (num_qubits + native.num_bits) * len(input_rows)
    if cells > max_native_cells:
        return dict(status='unknown', equivalent=None, reason='measured_native_replay_cell_budget', requested_cells=cells)
    for dirty in (0, 1):
        sim = km.Simulator(batch_size=len(input_rows), seed=311 + dirty)
        expected = {q: np.zeros(len(input_rows), dtype=bool) for q in range(num_qubits)}
        for position, q in enumerate(input_wires):
            values = np.array([row[position] for row in input_rows], dtype=bool)
            sim.write_across_shots(km.q(q), values); expected[q] = values
        for position, q in enumerate(target_wires):
            initial = np.full(len(input_rows), bool(dirty))
            sim.write_across_shots(km.q(q), initial)
            expected[q] = initial ^ np.array([row[position] for row in output_rows], dtype=bool)
        sim.do(native)
        if any(not np.array_equal(sim.read_across_shots(km.q(q)), value) for q, value in expected.items()):
            return dict(status='different', equivalent=False, reason='independent_native_output_mismatch')
        if sim.read_phase_flipped_across_shots().any():
            return dict(status='different', equivalent=False, reason='independent_native_phase_mismatch')
    return dict(status='equivalent', equivalent=True, measurement_mode='independent-map',
                evidence='canonical outcome-independent values and exact zero sign plus independent native replay',
                all_measurement_outcomes=True, workspace_restored=True, phase_zero=True,
                internal_bits=sorted(internal_bits), input_rows=len(input_rows), native_target_replays=2,
                dirty_targets='arbitrary; exact deltas depend only on specified input wires',
                scope='specified witness input rows and all formal outcomes; exact basis/sign maps, not a general channel theorem')


def measured_xor_and_witness(receipt, input_rows, output_rows, *, input_wires, target_wires,
                             clean_wires, num_qubits, classical_bit_offset=0, internal_bits=(),
                             max_states=100000, max_operations=100000, max_variables=512,
                             max_nodes=262144, max_apply_steps=1048576, max_native_cells=1048576,
                             symbolic_dag=None, symbolic_reference=None, symbolic_clean_inputs=None):
    """Compare reverse cleanup with guarded HMR cleanup of complete DAG schedules."""
    if type(classical_bit_offset) is not int or classical_bit_offset < 0:
        raise StrataError('classical_bit_offset must be a nonnegative parent classical width')
    if any(type(bit) is not int or bit < 0 for bit in internal_bits):
        raise StrataError('internal_bits must contain nonnegative classical IDs')
    layout = dict(input_wires=list(input_wires), target_wires=list(target_wires),
                  clean_wires=list(clean_wires), num_qubits=num_qubits)
    reverse = scheduled_xor_and_witness(receipt, input_rows, output_rows, **layout,
        max_states=max_states, max_operations=max_operations, max_native_cells=max_native_cells,
        symbolic_dag=symbolic_dag, symbolic_reference=symbolic_reference, symbolic_clean_inputs=symbolic_clean_inputs,
        symbolic_limits=dict(max_variables=max_variables, max_nodes=max_nodes, max_apply_steps=max_apply_steps))
    inputs = _layout(layout['input_wires'], num_qubits, 'inputs')
    targets = _layout(layout['target_wires'], num_qubits, 'targets')
    clean = sorted(_layout(layout['clean_wires'], num_qubits, 'clean'))
    gates = receipt['witness'] if symbolic_dag is None else symbolic_dag['gates']
    output_masks = receipt['output_coefficients'] if symbolic_dag is None else symbolic_dag['output_coefficients']
    offset = len(inputs) + 1
    kernels = [dict(row) for row in reverse['kernel_dag']]
    materialized = {row['id'] for row in kernels}
    dummy = list(range(len(inputs) + len(gates)))
    for row in kernels:
        gate = gates[row['id']]
        cleanup = _cleanup_template(gate['left_coefficients'], gate['right_coefficients'],
                                    dummy[:len(inputs) + row['id']], len(dummy), classical_bit_offset)
        row['uncompute_cost'] = sum(line.split()[0] in {'CCX', 'CCZ', 'CSWAP'} for line in cleanup['lines'])
    plan = plan_reversible_pebbles(kernels, reverse['planning']['outputs'],
        inputs=reverse['planning']['inputs'], input_width=reverse['planning']['input_width'],
        fixed_width=reverse['planning']['fixed_width'], max_width=reverse['planning']['max_width'],
        max_states=max_states, consume_cost=reverse['consumer_products'])
    alternatives, failures = [], list(reverse['failures'])
    for alternative in reverse['alternatives']:
        alternative = dict(alternative)
        alternative.update(cleanup_strategy='reverse', introduced_private_bits=[],
                           required_internal_bits=sorted(internal_bits), measurement_mode='independent-map')
        alternatives.append(alternative)

    def referenced(mask):
        return {i for i in range(len(gates)) if mask >> (offset + i) & 1}

    def consumer(available):
        emitted = []
        for mask, target in zip(output_masks, targets):
            affine = mask
            for index in sorted(referenced(mask) - materialized):
                affine ^= 1 << (offset + index); gate = gates[index]
                emitted.extend(_product_update(gate['left_coefficients'], gate['right_coefficients'],
                                                available[:len(inputs) + index], target))
            emitted.extend(_affine_update(affine, available, target))
        return emitted

    public = set(range(num_qubits)) - set(clean)
    for scheduled in plan['alternatives']:
        slots = clean[:scheduled['peak_width'] - len(public)]
        available, placement, free = inputs + [None] * len(gates), {}, set(slots)
        emitted, ledger, fresh, fallbacks = [], [], [], []
        proof_steps = 0
        for action in scheduled['schedule']:
            proof = None
            if action['action'] == 'consume':
                block = consumer(available)
            else:
                index = action['node']; gate = gates[index]
                if action['action'] == 'compute':
                    if not free: raise StrataError('measured schedule exceeds its helper allocation')
                    wire = min(free); free.remove(wire); placement[index] = wire
                    available[len(inputs) + index] = wire
                wire = placement[index]
                block = _product_update(gate['left_coefficients'], gate['right_coefficients'],
                                        available[:len(inputs) + index], wire)
                if action['action'] == 'uncompute':
                    bit = classical_bit_offset + len(fresh)
                    proof = (prove_product_cleanup('\n'.join(emitted), left=gate['left_coefficients'],
                        right=gate['right_coefficients'], available=available[:len(inputs) + index],
                        target=wire, bit=bit, clean_inputs=clean, max_operations=max_operations,
                        max_variables=max_variables, max_nodes=max_nodes,
                        max_apply_steps=max_apply_steps - proof_steps) if proof_steps < max_apply_steps else
                        dict(status='unknown', reason='cleanup_proof_global_apply_budget'))
                    proof_steps += proof.get('apply_steps', 0)
                    if proof['status'] == 'proved' and proof_steps <= max_apply_steps:
                        block = proof['lines']; fresh.append(bit)
                    else:
                        block = list(reversed(block)); fallbacks.append(dict(node=index, reason=proof['reason']))
                    available[len(inputs) + index] = None; free.add(placement.pop(index))
            ledger.append(dict(action, start=len(emitted), stop=len(emitted) + len(block),
                               instructions=len(block), toffoli=sum(line.split()[0] in {'CCX', 'CCZ', 'CSWAP'} for line in block),
                               hmr=sum(line.startswith('HMR ') for line in block), cleanup_proof=proof))
            emitted.extend(block)
            if len(emitted) > max_operations: break
        if len(emitted) > max_operations:
            failures.append(dict(reason='measured_native_operation_budget')); continue
        if placement: raise StrataError('measured schedule did not return product storage')
        if not fresh: continue
        physical_width = max(public | set(slots), default=-1) + 1
        native = km.Circuit('\n'.join(emitted)); boundary_instructions = 0
        if native.num_qubits < physical_width:
            emitted += [f'X q{physical_width - 1}'] * 2; boundary_instructions = 2
            native = km.Circuit('\n'.join(emitted))
        if len(native) > max_operations:
            failures.append(dict(reason='measured_native_operation_budget')); continue
        private = sorted(set(internal_bits) | set(fresh))
        if symbolic_dag is None:
            proof = verify_measured_witness_candidate(native, receipt, input_rows, output_rows, **layout,
                internal_bits=private, max_operations=max_operations, max_variables=max_variables,
                max_nodes=max_nodes, max_apply_steps=max_apply_steps, max_native_cells=max_native_cells)
        else:
            from .symbolic_boolean_graph import verify_symbolic_native
            proof = verify_symbolic_native(symbolic_reference, native, clean_inputs=symbolic_clean_inputs,
                internal_bits=private, max_operations=max_operations, max_variables=max_variables,
                max_nodes=max_nodes, max_apply_steps=max_apply_steps, max_native_cells=max_native_cells)
        if proof.get('equivalent') is not True:
            failures.append(dict(reason='measured_candidate_map_unproved', proof=proof)); continue
        resources = Kickmix().measured(native)
        bill = dict(Q=physical_width, toffoli=sum(step['toffoli'] for step in ledger), t=0, rotations=0,
                    instructions=sum(step['instructions'] for step in ledger) + boundary_instructions)
        if any(resources[name] != value for name, value in bill.items()):
            raise StrataError('complete measured cleanup ledger disagrees with native resources')
        if not fallbacks and Fraction(scheduled['cost']) != resources['toffoli']:
            raise StrataError('measured schedule paid-work plan disagrees with emitted cost')
        alternatives.append(dict(status='constructed', candidate=str(native), resources=resources,
            cleanup_strategy='guarded_hmr', schedule=scheduled, kernel_ledger=ledger,
            resource_ledger=bill, ledger_matches_native=True, cleanup_fallbacks=fallbacks,
            used_clean_wires=slots, omitted_clean_wires=[q for q in clean if q not in slots],
            introduced_private_bits=fresh, required_internal_bits=private,
            measurement_mode='independent-map', proof=proof,
            measurement_bill=dict(hmr=len(fresh), classical_width=native.num_bits,
                                  fresh_classical_bits=len(fresh), repair_gates=len(fresh),
                                  scope='full native HMR, correction, operand restoration and classical allocation; no standalone HMR price is invented')))
    for alternative in alternatives:
        alternative['resources_static'] = dict(alternative['resources'])
        try:
            alternative['resources_expected'] = measured_resources(alternative['candidate'], accounting='expected',
                max_operations=max_operations, max_variables=max_variables,
                max_nodes=max_nodes, max_apply_steps=max_apply_steps)
        except StrataError as exc:
            alternative['resources_expected'] = None
            alternative['expected_accounting_obstruction'] = str(exc)
    return dict(status='constructed' if alternatives else 'unknown', family='measured_joint_xor_and',
                alternatives=alternatives, planning=plan, reverse_planning=reverse['planning'], kernel_dag=kernels,
                failures=failures, classical_bit_offset=classical_bit_offset,
                measurement_mode='independent-map',
                witness_basis='verified_symbolic_polynomial_DAG' if symbolic_dag is not None else 'verified_truth_row_XOR_AND_witness',
                truth_rows_enumerated=0 if symbolic_dag is not None else len(input_rows),
                scope='complete fixed-DAG schedules; HMR only for proved current products with available parents',
                no_global_algorithmic_optimality_claim=True)


def measured_boundary_synthesis(circuit, *, joint_result=None, algebraic_result=None, clean_inputs=(), internal_bits=(),
                                  classical_bit_offset=None, resource_measure=None,
                                  cost='Q*(toffoli+t/4)', accounting='static', external_distribution=None,
                                  max_schedule_states=100000, max_native_cells=1048576, **options):
    """Select a fully checked measured/reverse schedule using the host objective.

    A fragment caller must supply the full parent's classical bit width as
    ``classical_bit_offset``. Existing bits become private only through explicit
    ``internal_bits``. Fresh compiler outputs are separately reported so the
    host can validate the same scope before splicing the candidate.
    """
    from .boolean_synthesis import joint_boundary_synthesis
    from .measurement_equivalence import equivalent_independent_outputs
    native_source = circuit if isinstance(circuit, km.Circuit) else km.Circuit(circuit)
    offset = native_source.num_bits if classical_bit_offset is None else classical_bit_offset
    if type(offset) is not int or offset < native_source.num_bits:
        raise StrataError('fresh cleanup outcomes must start above the source and parent classical layout')
    clean, private = sorted(set(clean_inputs)), sorted(set(internal_bits))
    symbolic = algebraic_result is not None
    if symbolic:
        if algebraic_result.get('status') != 'constructed':
            return dict(status='unknown', reason='symbolic algebraic DAG unavailable', algebraic_synthesis=algebraic_result)
        source_proposal = algebraic_result
        dag = algebraic_result['dag']
        contract = dict(input_wires=dag['input_wires'], target_wires=dag['target_wires'],
                        clean_wires=sorted(set(clean) - set(dag['target_wires'])), num_qubits=native_source.num_qubits)
        required_existing = set()
    elif joint_result is None:
        joint_result = joint_boundary_synthesis(native_source, clean_inputs=clean, internal_bits=private, **options)
    if not symbolic and joint_result.get('status') != 'constructed':
        return dict(status='unknown', reason='joint function witness unavailable',
                    required_internal_bits=joint_result.get('required_internal_bits', []), joint_synthesis=joint_result)
    if not symbolic:
        source_proposal = joint_result
        contract = joint_result['synthesis_contract']
        required_existing = {int(bit[1:]) for bit in joint_result['proof'].get('internal_classical_outputs', [])}
    if not required_existing <= set(private):
        return dict(status='unknown', reason='existing classical outputs require explicit private-bit permission',
                    required_internal_bits=sorted(required_existing))
    if set(source_proposal['proof']['clean_inputs']) != set(clean):
        raise StrataError('measured synthesis clean promise differs from its joint witness')
    limits = dict(max_operations=options.get('max_operations', 10000),
                  max_variables=options.get('max_variables', 128), max_nodes=options.get('max_terms', 65536),
                  max_apply_steps=32 * options.get('max_product_terms', 65536))
    built = measured_xor_and_witness(None if symbolic else joint_result['exact_search'],
        None if symbolic else contract['input_rows'], None if symbolic else contract['output_rows'],
        **{key: contract[key] for key in ('input_wires', 'target_wires', 'clean_wires', 'num_qubits')},
        classical_bit_offset=offset, internal_bits=private, max_states=max_schedule_states,
        max_native_cells=max_native_cells,
        symbolic_dag=dag if symbolic else None, symbolic_reference=native_source if symbolic else None,
        symbolic_clean_inputs=clean if symbolic else None, **limits)
    formula = Formula(cost) if isinstance(cost, str) else cost
    valid, failures = [], list(built['failures'])
    for alternative in built['alternatives']:
        omitted = set(alternative['omitted_clean_wires'])
        metadata = []
        for line in str(native_source).splitlines():
            words = shlex.split(line, comments=True)
            if not words or words[0] not in {'REGISTER', 'APPEND_TO_REGISTER', 'INFO_TAG'}: continue
            if words[0] == 'APPEND_TO_REGISTER' and any(word == f'q{q}' for q in omitted for word in words): continue
            metadata.append(line)
        candidate = km.Circuit('\n'.join(metadata + [alternative['candidate']]))
        check = equivalent_independent_outputs(str(native_source), str(candidate), clean_inputs=clean,
            internal_bits=alternative['required_internal_bits'], **limits)
        if check.get('equivalent') is not True:
            failures.append(dict(reason='measured_source_independent_map_unproved', check=check)); continue
        resources = Kickmix().measured(candidate)
        objective_resources = (resource_measure(candidate) if resource_measure is not None else
                               measured_resources(candidate, accounting=accounting, external_distribution=external_distribution))
        alternative['resource_ledger']['instructions'] += len(km.Circuit('\n'.join(metadata)))
        if any(resources[name] != value for name, value in alternative['resource_ledger'].items()):
            raise StrataError('measured source metadata bill disagrees with full native cost')
        alternative.update(candidate=str(candidate), resources=resources, resources_static=resources,
                           objective_resources=objective_resources, cost=formula(objective_resources),
                           whole_boundary_check=check)
        try:
            alternative['resources_expected'] = measured_resources(candidate, accounting='expected',
                                                                      external_distribution=external_distribution)
        except StrataError as exc:
            alternative['resources_expected'] = None; alternative['expected_accounting_obstruction'] = str(exc)
        valid.append(alternative)
    if not valid:
        return dict(status='unknown', reason='no measured or reverse schedule passed independent source checking',
                    failures=failures, lowering=built)
    best = min(valid, key=lambda row: (row['cost'], row['resources']['Q'], row['resources']['toffoli']))
    structural = dict(source_proposal.get('structural_witness', {}))
    # A declared-zero input may itself be a wanted nonzero output. Only the
    # witness's restored helper set can be retired, never those target wires.
    retired = sorted(set(structural.get('materialized_clean_wires', ())) &
                     set(contract['clean_wires']) - set(best['used_clean_wires']))
    structural.update(retired_scratch_wires=retired, retained_product_wires=best['used_clean_wires'],
                      measured_cleanup_nodes=len(best['introduced_private_bits']),
                      schedule_recomputations=best['schedule']['recomputations'],
                      removed_materialization_bank=len(retired) >= 2 and structural.get('source_nonlinear_dataflow_depth', 0) >= 2)
    return dict(status='constructed', family='measured_joint_xor_and', candidate=best['candidate'],
                alternatives=valid, planning=built['planning'], kernel_dag=built['kernel_dag'], failures=failures,
                introduced_private_bits=best['introduced_private_bits'],
                required_internal_bits=best['required_internal_bits'], classical_bit_offset=offset,
                measurement_mode='independent-map', proof=best['whole_boundary_check'],
                structural_witness=structural, resource_bill=best['resource_ledger'],
                selected_cost=best['cost'], selected_cleanup_strategy=best['cleanup_strategy'],
                witness_basis=built['witness_basis'], truth_rows_enumerated=built['truth_rows_enumerated'],
                objective_scope='parent_context' if resource_measure is not None else 'supplied_circuit',
                selection_scope='host objective among fully emitted reverse and guarded-HMR alternatives; fixed-DAG paid-work frontier only',
                no_global_algorithmic_optimality_claim=True,
                witness_hash=digest(dict(source=str(native_source), candidate=best['candidate'], clean=clean,
                                         existing_private_bits=private, introduced_private_bits=best['introduced_private_bits'],
                                         contract=digest(dag) if symbolic else joint_result['exact_search']['contract_hash'])))


def measured_symbolic_boundary_synthesis(circuit, *, algebraic_result, **options):
    """Schedule and measure an independently replayed symbolic polynomial DAG."""
    return measured_boundary_synthesis(circuit, algebraic_result=algebraic_result, **options)
