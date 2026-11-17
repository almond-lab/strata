"""Exact native trajectory receipts without distributing products of sums.

This backend is deliberately distinct from the sparse-ANF schema. It proves
all boundary inputs and fresh outcomes symbolically; no sampling or inferred
clean inputs are used. Bounded decision graphs remain an explicit unknown.
"""
from .factored import DecisionDAG, DecisionBudget


def _parsed(lines, limits):
    from .semantic import _lines, _parse
    rows, wires, bits, measurements = [], set(), set(), 0
    for index, line in enumerate(_lines(lines)):
        if index >= limits['max_operations']:
            return dict(status='bounded', reason='operation_budget', at=index,
                        limit=limits['max_operations'])
        item = _parse(line)
        if item['status'] == 'opaque':
            return dict(status='opaque', reason=item['reason'], at=index, instruction=line.rstrip())
        if item['status'] == 'metadata': continue
        rows.append((index, line.rstrip(), item))
        wires.update(item['wires']); bits.update(item['bits'])
        if item['condition']: bits.add(item['condition'])
        measurements += item['op'] == 'HMR'
    if len(wires) + len(bits) + measurements > limits['max_variables']:
        return dict(status='bounded', reason='variable_budget',
                    requested=len(wires)+len(bits)+measurements, limit=limits['max_variables'])
    if len(wires) + len(bits) + measurements > 512:
        return dict(status='bounded', reason='decision_depth_budget',
                    requested=len(wires)+len(bits)+measurements, limit=512)
    return rows, wires, bits, measurements


def _execute(parsed, dag, clean):
    rows, wires, bits, _ = parsed
    initial = {q: 0 if q in clean else dag.variable(q) for q in sorted(wires)}
    state, classical = dict(initial), {b: dag.variable(b) for b in sorted(bits)}
    phase, stack, events, resets = 0, [], [], []
    xor, mul = dag.xor, dag.product
    index, line = None, None
    try:
        for index, line, item in rows:
            if index % 128 == 0 and len(dag.nodes) > 1024:
                dag.collect([*initial.values(),*state.values(),*classical.values(),phase,*stack,
                             *(e[field] for e in events for field in ('active','measured_value'))])
            op, qs = item['op'], item['wires']
            active = classical[item['condition']] if item['condition'] else 1
            if op == 'PUSH_CONDITION':
                stack.append(active); continue
            if op == 'POP_CONDITION':
                if stack: stack.pop()
                continue
            for condition in stack: active = mul(active, condition)
            if op in {'BIT_STORE0', 'BIT_STORE1', 'BIT_INVERT'}:
                bit = item['bits'][0]; old = classical[bit]
                classical[bit] = xor(old, active if op == 'BIT_INVERT' else mul(active, xor(old, int(op == 'BIT_STORE1'))))
            elif op == 'HMR':
                q, bit = qs[0], item['bits'][0]; name = f'm{len(events)}'
                outcome, old = dag.variable(name), state[q]
                phase = xor(phase, mul(mul(old, outcome), active))
                events.append(dict(at=index, wire=q, bit=bit, variable=name, active=active, measured_value=old))
                classical[bit] = xor(classical[bit], mul(active, xor(classical[bit], outcome)))
                state[q] = xor(old, mul(active, old))
            elif op == 'R':
                if mul(active, state[qs[0]]):
                    return dict(status='opaque', reason='reset_not_proven_zero', at=index, instruction=line, wire=qs[0])
                resets.append(dict(at=index, wire=qs[0], proof='canonical_active_times_value_is_zero'))
            elif op == 'X': state[qs[0]] = xor(state[qs[0]], active)
            elif op in {'CX', 'CCX'}:
                delta = active
                for q in qs[:-1]: delta = mul(delta, state[q])
                state[qs[-1]] = xor(state[qs[-1]], delta)
            elif op in {'SWAP', 'CSWAP'}:
                a, b = qs[-2:]
                control = mul(active, state[qs[0]]) if op == 'CSWAP' else active
                delta = mul(control, xor(state[a], state[b]))
                state[a], state[b] = xor(state[a], delta), xor(state[b], delta)
            else:
                delta = active
                for q in qs: delta = mul(delta, state[q])
                if op != 'Z_POW' or item['sign_exponent']: phase = xor(phase, delta)
    except DecisionBudget as exc:
        return dict(status='bounded', reason=str(exc), at=index, instruction=line,
                    decision_nodes=len(dag.nodes), apply_steps=dag.steps,
                    limits=dict(max_nodes=dag.max_nodes, max_apply_steps=dag.max_steps))
    if stack: return dict(status='opaque', reason='unclosed_condition_scope', depth=len(stack))
    return dict(initial=initial, state=state, classical=classical, phase=phase,
                events=events, resets=resets, wires=sorted(wires), bits=sorted(bits), operations=len(rows))


def _prepare(lines, clean_inputs, max_operations, max_variables, max_nodes, max_apply_steps):
    from .semantic import _limits
    _limits(max_operations=max_operations, max_variables=max_variables,
            max_nodes=max_nodes, max_apply_steps=max_apply_steps)
    clean = frozenset(clean_inputs)
    if any(type(q) is not int or q < 0 for q in clean):
        raise ValueError('clean_inputs must contain nonnegative physical wire IDs')
    parsed = _parsed(lines, dict(max_operations=max_operations, max_variables=max_variables))
    if isinstance(parsed, dict): return None, parsed, clean
    _, wires, bits, count = parsed
    # Classical choices precede quantum data: controlled pivot permutations
    # then share the quantum decision subgraphs rather than ANF expansions.
    variables = sorted(bits) + sorted(wires) + [f'm{i}' for i in range(count)]
    dag = DecisionDAG(variables, max_nodes=max_nodes, max_steps=max_apply_steps)
    try: result = _execute(parsed, dag, clean)
    except DecisionBudget as exc:
        result = dict(status='bounded', reason=str(exc), at=None, instruction=None)
    return dag, result, clean


def _receipt(dag, result, clean):
    if 'status' in result: return result
    roots = {f'q{q}': value for q, value in result['state'].items()}
    roots.update(result['classical']); roots['phase'] = result['phase']
    for event in result['events']:
        for field in ('active', 'measured_value'):
            roots[f'{event["variable"]}.{field}'] = event[field]
    graph = dag.export(roots)
    changed = [q for q in result['wires'] if result['state'][q] != result['initial'][q]]
    outcomes = [e['variable'] for e in result['events']]
    phase_support = dag.support(result['phase'])
    return dict(status='exact_factored', kind='factored_boolean_trajectory',
                inputs=result['wires'], quantum_inputs=result['wires'], classical_inputs=result['bits'],
                boundary_width=len(result['wires']), operation_count=result['operations'],
                clean_input_assumptions=sorted(clean), measurement_variables=outcomes,
                measurement_events=[{k:v for k,v in event.items() if k not in {'active','measured_value'}} |
                                    {field: f'{event["variable"]}.{field}' for field in ('active','measured_value')}
                                    for event in result['events']],
                modified_wires=changed, preserved_wires=[q for q in result['wires'] if q not in changed],
                phase_zero=result['phase'] == 0,
                effects=dict(clean_outputs=[q for q,v in result['state'].items() if v == 0],
                             reset_proofs=result['resets'],
                             phase_debts=[dict(measurement=m, status='outstanding' if m in phase_support else 'discharged',
                                               proof='canonical_final_phase_independence_from_outcome') for m in outcomes]),
                factored=graph, evidence='canonical_reduced_ordered_boolean_decisions',
                limits=dict(max_nodes=dag.max_nodes,max_apply_steps=dag.max_steps),
                semantic_scope='native_outcome_parametric_basis_trajectories',
                assumptions='declared clean inputs zero; all other quantum/classical inputs and fresh outcomes arbitrary')


def analyze_factored(lines, *, clean_inputs=(), max_operations=10000, max_variables=512,
                     max_nodes=262144, max_apply_steps=1048576):
    dag, result, clean = _prepare(lines, clean_inputs, max_operations, max_variables, max_nodes, max_apply_steps)
    return _receipt(dag, result, clean)


def equivalent_factored(original, candidate, *, clean_inputs=(), max_operations=10000,
                        max_variables=512, max_nodes=262144, max_apply_steps=1048576):
    from .semantic import _limits
    _limits(max_operations=max_operations, max_variables=max_variables, max_nodes=max_nodes, max_apply_steps=max_apply_steps)
    clean = frozenset(clean_inputs)
    if any(type(q) is not int or q < 0 for q in clean): raise ValueError('invalid clean inputs')
    limits = dict(max_operations=max_operations, max_variables=max_variables)
    before, after = _parsed(original, limits), _parsed(candidate, limits)
    if isinstance(before, dict) or isinstance(after, dict):
        return dict(status='unknown', equivalent=None, before=before if isinstance(before,dict) else None,
                    after=after if isinstance(after,dict) else None, reason='factored_semantics_unavailable')
    wires, bits = sorted(before[1] | after[1]), sorted(before[2] | after[2])
    count = max(before[3], after[3])
    if len(wires)+len(bits)+count > max_variables:
        return dict(status='unknown', equivalent=None, reason='variable_budget')
    if len(wires)+len(bits)+count > 512:
        return dict(status='unknown', equivalent=None, reason='decision_depth_budget',limit=512)
    outcomes = [f'm{i}' for i in range(count)]
    dag = DecisionDAG(bits+wires+outcomes, max_nodes=max_nodes, max_steps=max_apply_steps)
    try:
        a = _execute(before, dag, clean)
        if 'status' not in a:
            dag.pins.update([*a['state'].values(),*a['classical'].values(),a['phase'],
                             *(e[field] for e in a['events'] for field in ('active','measured_value'))])
        b = _execute(after, dag, clean)
        if 'status' in a or 'status' in b:
            return dict(status='unknown', equivalent=None, before=a, after=b, reason='factored_semantics_unavailable')
        def provenance(r):
            return [(e['wire'],e['bit'],e['active'],e['measured_value']) for e in r['events']]
        if provenance(a) != provenance(b):
            return dict(status='unknown', equivalent=None, reason='measurement_event_alignment_unproved')
        changes = [(q, a['state'].get(q, 0 if q in clean else dag.variable(q)),
                       b['state'].get(q, 0 if q in clean else dag.variable(q))) for q in wires]
        changes += [(bit, a['classical'].get(bit, dag.variable(bit)), b['classical'].get(bit, dag.variable(bit))) for bit in bits]
        changes += [('phase',a['phase'],b['phase'])]
        differences = [(name,x,y) for name,x,y in changes if x != y]
        result = dict(status='different' if differences else 'equivalent', equivalent=not differences,
                      evidence='canonical_reduced_ordered_boolean_decisions',
                      clean_input_assumptions=sorted(clean), measurement_variables=outcomes,
                      semantic_scope='native_outcome_parametric_basis_trajectories',
                      phase_equal=a['phase']==b['phase'], decision_nodes=len(dag.nodes), apply_steps=dag.steps)
        if differences:
            name,x,y=differences[0]; assignment=dag.witness(dag.xor(x,y))
            result['counterexample']=dict(kind='phase' if name=='phase' else 'output' if isinstance(name,int) else 'classical_output',
                wire=name, inputs={str(q):assignment[q] for q in wires}, classical_inputs={bit:assignment[bit] for bit in bits},
                measurement_outcomes={m:assignment[m] for m in outcomes},
                before=dag.evaluate(x,assignment), after=dag.evaluate(y,assignment),
                before_phase=dag.evaluate(a['phase'],assignment), after_phase=dag.evaluate(b['phase'],assignment))
        return result
    except DecisionBudget as exc:
        return dict(status='unknown', equivalent=None, reason=str(exc), decision_nodes=len(dag.nodes), apply_steps=dag.steps)


def lift_lookup_table(circuit, *, clean_inputs=(), max_operations=60000, max_variables=512,
                      max_nodes=262144, max_apply_steps=1048576, max_address_bits=12,
                      max_table_cells=1048576, max_native_cells=524288, native_replay=True):
    """Prove an XOR table for arbitrary dirty targets and every HMR branch.

    Address support is recovered from canonical output deltas. The complete
    symbolic proof establishes target independence, zero workspace and phase
    before native address enumeration. A zero-target replay alone is never the
    proof. Restricted reachable domains must be supplied to the separate finite
    domain checker; this routine never silently assumes such a promise.
    """
    from .semantic import _limits
    _limits(max_address_bits=max_address_bits, max_table_cells=max_table_cells,
            max_native_cells=max_native_cells)
    lines = str(circuit).splitlines() if not isinstance(circuit, (str,list,tuple)) else circuit
    dag, result, clean = _prepare(lines, clean_inputs, max_operations, max_variables, max_nodes, max_apply_steps)
    receipt = _receipt(dag, result, clean)
    if receipt['status'] != 'exact_factored': return dict(status='unknown', reason='complete_factored_boundary_unavailable', semantics=receipt)
    def reject(reason, **extra):
        return dict(status='not_applicable', reason=reason, semantics=receipt, **extra)
    if result['phase']:
        return reject('lookup has unresolved phase debt or a nonzero boundary phase',
                      phase_witness={str(q):v for q,v in dag.witness(result['phase']).items()})
    leaked = [q for q in clean if result['state'].get(q,0)]
    if leaked: return reject('declared clean lookup workspace is not restored', wires=sorted(leaked))
    try:
        deltas = {q: dag.xor(value,result['initial'][q]) for q,value in result['state'].items()
                  if q not in clean and value != result['initial'][q]}
    except DecisionBudget as exc:
        return dict(status='bounded', reason=str(exc), semantics=receipt)
    support = set().union(*(dag.support(value) for value in deltas.values()))
    if any(not isinstance(q,int) for q in support):
        return reject('lookup outputs depend on classical inputs or measurement outcomes')
    if support & deltas.keys():
        return reject('lookup deltas depend on dirty targets; an explicit reachable-domain proof is required',
                      target_dependencies=sorted(support & deltas.keys()))
    addresses, targets = sorted(support), sorted(deltas)
    if len(addresses) > max_address_bits or (1<<len(addresses))*max(1,len(targets)) > max_table_cells:
        return dict(status='bounded', reason='lookup_table_budget', address_bits=len(addresses),
                    requested_cells=(1<<len(addresses))*len(targets),
                    limits=dict(max_address_bits=max_address_bits,max_table_cells=max_table_cells), semantics=receipt)
    table=[]
    for address in range(1<<len(addresses)):
        assignment={q:(address>>i)&1 for i,q in enumerate(addresses)}
        table.append(sum(dag.evaluate(deltas[q],assignment)<<i for i,q in enumerate(targets)))
    anf={}
    for i,q in enumerate(targets):
        coefficients=[value>>i&1 for value in table]
        for bit in range(len(addresses)):
            for mask in range(len(coefficients)):
                if mask>>bit&1: coefficients[mask] ^= coefficients[mask^(1<<bit)]
        anf[str(q)]=[[wire for bit,wire in enumerate(addresses) if mask>>bit&1]
                     for mask,value in enumerate(coefficients) if value]
    replay=dict(status='not_run')
    if native_replay:
        import kickmix as km
        import numpy as np
        native=circuit if isinstance(circuit,km.Circuit) else km.Circuit('\n'.join(lines) if not isinstance(lines,str) else lines)
        cells=(native.num_qubits+native.num_bits)*len(table)
        if cells>max_native_cells:
            return dict(status='bounded',reason='native_replay_cell_budget',requested_cells=cells,
                        limit=max_native_cells,physical_qubits=native.num_qubits,
                        physical_bits=native.num_bits,address_count=len(table),semantics=receipt)
        sim=km.Simulator(batch_size=len(table),seed=431)
        for bit,q in enumerate(addresses):
            sim.write_across_shots(km.q(q),np.array([(value>>bit)&1 for value in range(len(table))],dtype=bool))
        sim.do(native)
        actual=[0]*len(table)
        for bit,q in enumerate(targets):
            for row,value in enumerate(sim.read_across_shots(km.q(q))): actual[row] |= int(value)<<bit
        restored=all(not sim.read_across_shots(km.q(q)).any() for q in clean if q<native.num_qubits)
        preserved=all(list(sim.read_across_shots(km.q(q)).astype(int))==[(value>>bit)&1 for value in range(len(table))]
                      for bit,q in enumerate(addresses))
        phase_zero=not sim.read_phase_flipped_across_shots().any()
        passed=actual==table and restored and preserved and phase_zero
        replay=dict(status='passed' if passed else 'failed', address_count=len(table),
                    targets_equal=actual==table, workspace_restored=restored, address_preserved=preserved, phase_zero=phase_zero,
                    scope='native enumeration at zero target inputs and sampled native outcomes; all-input/all-outcome proof is the canonical decision receipt')
        if not passed: return dict(status='unknown',reason='native_lookup_replay_mismatch',native_replay=replay,semantics=receipt)
    return dict(status='exact',kind='xor_lookup_table',address_wires=addresses,target_wires=targets,
                table=table,table_encoding='target bits little endian; address wires little endian',
                deltas=anf,clean_input_assumptions=sorted(clean),semantics=receipt,native_replay=replay,
                proof=dict(dirty_target_independence=True,workspace_restored=True,phase_zero=True,
                           all_measurement_outcomes=True,all_other_inputs_arbitrary=True),
                domain_scope='all boundary values except explicitly zero clean inputs',
                evidence='canonical decisions for full boundary plus exhaustive address table and Mobius ANF conversion')
