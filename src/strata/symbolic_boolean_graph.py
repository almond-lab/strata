"""Independent contracts for lowering polynomial DAGs without truth tables."""
import kickmix as km
import numpy as np

from .boolean_synthesis import _layout
from .model import StrataError


def verify_symbolic_graph(reference, dag, *, clean_inputs, max_operations=100000,
                           max_variables=512, max_terms=262144, max_product_terms=1048576):
    """Replay coefficients against every recovered final ANF, with no row surrogate."""
    from .algebraic_synthesis import _Budget, _Bound, _replay
    from .inference import functions_at
    native = reference if isinstance(reference, km.Circuit) else km.Circuit(reference)
    clean = set(_layout(clean_inputs, native.num_qubits, 'clean inputs'))
    inputs = _layout(dag['input_wires'], native.num_qubits, 'symbolic inputs')
    targets = _layout(dag['target_wires'], native.num_qubits, 'symbolic targets')
    if set(inputs) & (set(targets) | clean):
        raise StrataError('symbolic graph inputs overlap targets or zero inputs')
    state = functions_at(native, clean_inputs=clean, max_operations=max_operations,
                         max_variables=max_variables, max_terms=max_terms, max_product_terms=max_product_terms)
    if state['status'] != 'exact' or state['phase']:
        raise StrataError('symbolic graph requires exact zero-phase source values')
    deltas = {q: frozenset(value ^ (set() if q in clean else {frozenset({q})}))
              for q, value in state['functions'].items()
              if value != (set() if q in clean else {frozenset({q})})}
    if set(targets) != set(deltas):
        raise StrataError('symbolic graph omits or changes source boundary targets')
    if len(dag['gates']) > max_operations:
        raise StrataError('symbolic graph operation budget')
    budget = _Budget(32 * max_product_terms, max_terms, max_terms, max_product_terms, 10.0)
    try:
        proof = _replay(inputs, dag['gates'], dag['output_coefficients'], [deltas[q] for q in targets], budget)
    except _Bound as exc:
        raise StrataError(str(exc)) from exc
    return dict(proof, scope='complete source output ANFs under the declared zero promise',
                truth_rows_enumerated=0, minimum_claimed=False, replay_steps=budget.steps)


def verify_symbolic_native(reference, candidate, *, clean_inputs, internal_bits=(),
                            max_operations=100000, max_variables=512, max_nodes=262144,
                            max_apply_steps=1048576, max_native_cells=1048576):
    """All-input/all-outcome map proof with a bounded independent SDK cross-check."""
    from .measurement_equivalence import equivalent_independent_outputs
    before = reference if isinstance(reference, km.Circuit) else km.Circuit(reference)
    after = candidate if isinstance(candidate, km.Circuit) else km.Circuit(candidate)
    check = equivalent_independent_outputs(str(before), str(after), clean_inputs=clean_inputs,
        internal_bits=internal_bits, max_operations=max_operations, max_variables=max_variables,
        max_nodes=max_nodes, max_apply_steps=max_apply_steps)
    if check.get('equivalent') is not True: return check
    width = max(before.num_qubits, after.num_qubits)
    cells = 32 * (width + max(before.num_bits, after.num_bits))
    if cells > max_native_cells:
        return dict(status='unknown', equivalent=None, reason='symbolic_schedule_native_cell_budget', requested_cells=cells)
    rng = np.random.default_rng(7103)
    initial = rng.integers(0, 2, size=(width, 32)).astype(bool)
    for q in clean_inputs:
        if q < width: initial[q] = False
    simulations = []
    for circuit in (before, after):
        sim = km.Simulator(batch_size=32, seed=441)
        for q in range(width): sim.write_across_shots(km.q(q), initial[q])
        sim.do(circuit); simulations.append(sim)
    if any(not np.array_equal(*(sim.read_across_shots(km.q(q)) for sim in simulations)) for q in range(width)):
        return dict(status='different', equivalent=False, reason='symbolic_schedule_native_output_mismatch')
    if not np.array_equal(*(sim.read_phase_flipped_across_shots() for sim in simulations)):
        return dict(status='different', equivalent=False, reason='symbolic_schedule_native_phase_mismatch')
    return dict(check, truth_rows_enumerated=0,
                native_cross_check=dict(shots=32, status='passed', scope='sample only; canonical decisions prove every input and outcome'))
