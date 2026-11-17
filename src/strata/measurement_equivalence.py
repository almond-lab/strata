"""Exact outcome-independent basis/sign map comparison for declared outputs.

This is a deliberately restricted measurement equivalence rule: both programs
must have identical public value and sign maps independent of every HMR outcome.
Classical outputs may be omitted only through an explicit private-bit contract.
"""
from .factored import DecisionDAG, DecisionBudget
from .factored_semantic import _parsed, _execute
from .model import StrataError


def equivalent_independent_outputs(original,candidate,*,clean_inputs=(),internal_bits=(),
                                   max_operations=10000,max_variables=512,max_nodes=262144,
                                   max_apply_steps=1048576,required_zero_outputs=()):
    from .semantic import _limits
    _limits(max_operations=max_operations,max_variables=max_variables,max_nodes=max_nodes,max_apply_steps=max_apply_steps)
    internal_bits=tuple(internal_bits)
    private={f'b{bit}' for bit in internal_bits}
    if any(type(bit)is not int or bit<0 for bit in internal_bits):
        raise StrataError('internal bits must be nonnegative classical bit indices')
    clean=set(clean_inputs)
    if any(type(q)is not int or q<0 for q in clean):
        raise StrataError('clean inputs must be nonnegative physical wires')
    limits=dict(max_operations=max_operations,max_variables=max_variables)
    left=_parsed(original,limits);right=_parsed(candidate,limits)
    base=dict(kind='exact_outcome_independent_basis_sign_maps',equivalent=None,
              clean_inputs=sorted(clean),internal_bits=sorted(internal_bits),
              scope='all arbitrary public boundary inputs under declared clean promise; exact basis values/sign phase',
              quantum_channel_equivalence=False)
    for side,parsed in (('original',left),('candidate',right)):
        if isinstance(parsed,dict):return dict(base,status='unknown',side=side,reason=parsed.get('reason'))
    wires=left[1]|right[1];bits=left[2]|right[2]
    outcomes={f'm{i}' for i in range(max(left[3],right[3]))}
    variables=sorted(bits)+sorted(wires)+sorted(outcomes,key=lambda s:int(s[1:]))
    if len(variables)>min(max_variables,512):
        return dict(base,status='unknown',reason='independent_map_variable_budget')
    dag=DecisionDAG(variables,max_nodes=max_nodes,max_steps=max_apply_steps)
    try:
        a=_execute(left,dag,clean)
        if 'status' in a:return dict(base,status='unknown',side='original',reason=a['reason'])
        # The second execution can collect garbage. Pin every first-program
        # public root used in equality and independence checks.
        dag.pins.update([*a['initial'].values(),*a['state'].values(),*a['classical'].values(),a['phase']])
        b=_execute(right,dag,clean)
        if 'status' in b:return dict(base,status='unknown',side='candidate',reason=b['reason'])
        def initial(q):return 0 if q in clean else dag.variable(q)
        for q in required_zero_outputs:
            for name,state in [('original',a),('candidate',b)]:
                value=state['state'].get(q,initial(q))
                if value:
                    return dict(base,status='different',equivalent=False,reason='clean_output_not_restored',
                                side=name,wire=q,counterexample={str(k):v for k,v in dag.witness(value).items()})
        pairs=[(f'q{q}',a['state'].get(q,initial(q)),b['state'].get(q,initial(q))) for q in sorted(wires)]
        pairs += [(bit,a['classical'].get(bit,dag.variable(bit)),b['classical'].get(bit,dag.variable(bit)))
                  for bit in sorted(bits-private)]
        pairs.append(('phase',a['phase'],b['phase']))
        for name,x,y in pairs:
            if dag.support(x)&outcomes or dag.support(y)&outcomes:
                return dict(base,status='unknown',reason='public_value_or_phase_depends_on_outcomes',output=name)
        for name,x,y in pairs:
            difference=dag.xor(x,y)
            if difference:
                return dict(base,status='different',equivalent=False,reason='public_value_or_phase_mismatch',
                            output=name,counterexample={str(k):v for k,v in dag.witness(difference).items()})
    except DecisionBudget as exc:
        return dict(base,status='unknown',reason=str(exc))
    return dict(base,status='equivalent',equivalent=True,reason='verified',
                original_outcomes=left[3],candidate_outcomes=right[3],
                outcome_independence_proved=True,decision_nodes=len(dag.nodes),apply_steps=dag.steps,
                note='No equality of private measurement histories is required; general outcome-dependent channels remain unsupported.')
