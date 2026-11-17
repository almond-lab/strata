"""Bounded, exact HMR trajectories for one explicit boundary input at a time.

Canonical Boolean decisions represent every fresh outcome, including outcomes
under nonlinear conditions. Fractional phase uses Boolean bit planes of the
exact dyadic exponent modulo two; no floating-point comparison is involved.
The native runner separately replays branch zero as an implementation check.
"""
from fractions import Fraction

from .factored import BooleanDAG, DecisionBudget, DecisionDAG
from .fractional_equivalence import add_phase_bits
from .model import StrataError


class SymbolicHMR:
    def __init__(self, program, *, check_time, max_nodes, max_apply_steps,
                 max_operations, max_variables, max_phase_bits):
        self.program, self.check_time = program, check_time
        self.limits = dict(max_nodes=max_nodes, max_apply_steps=max_apply_steps,
                           max_operations=max_operations, max_variables=max_variables,
                           max_phase_bits=max_phase_bits)
        if max_nodes < 2:
            raise StrataError('symbolic HMR decision node budget exceeded; no certificate')
        if program.hmr_count > min(max_variables, 512):
            raise StrataError('symbolic HMR variable/depth budget exceeded; no certificate')
        # Canonical native circuit text contains the exact dyadic exponents.
        self.exponents = {at: Fraction(words[2]) % 2 for at, words in program.affine_ops
                          if words[0] == 'Z_POW'}
        denominator = max((v.denominator for v in self.exponents.values()), default=1)
        if any(v.denominator & (v.denominator - 1) for v in self.exponents.values()):
            raise StrataError('symbolic HMR requires exact dyadic phase; no certificate')
        self.phase_bits = denominator.bit_length()
        if self.phase_bits > max_phase_bits:
            raise StrataError('symbolic HMR phase budget exceeded; no certificate')
        self.denominator = denominator
        self.dag = DecisionDAG(range(program.hmr_count), max_nodes=max_nodes,
                               max_steps=max_apply_steps)
        self.operations = self.proved_cases = 0
        self.expression_nodes = 2

    def receipt(self, *, native_branches=()):
        return dict(mode='per_input_reduced_ordered_decisions',
                    all_measurement_outcomes=True, exact_phase=True,
                    semantic_scope='exact native basis-value and dyadic-phase trajectories for the explicitly enumerated boundary inputs; not a general quantum-channel equivalence theorem',
                    outcome_semantics='fresh independent formal bits in instruction order, including inactive measurement slots; no outcome-distribution claim',
                    initial_state_model='declared clean registers zero; declared public and dirty values enumerate the domain; undeclared classical storage zero',
                    proof_applies_to='proved_cases only; all-input completion requires the enclosing lab receipt to be exhaustive',
                    proved_cases=self.proved_cases,
                    measurement_variables=self.program.hmr_count,
                    outcome_order='instruction_order',
                    phase_representation='dyadic_bit_planes_mod_2', phase_bits=self.phase_bits,
                    decision_nodes=len(self.dag.nodes), apply_steps=self.dag.steps,
                    expression_nodes=self.expression_nodes,
                    symbolic_operations=self.operations, limits=dict(self.limits),
                    native_cross_check_branches=list(native_branches))

    def check(self, inputs, registers, expected, expected_phase):
        """Return an exact failing outcome assignment, or prove all outcomes."""
        try:
            return self._check(inputs, registers, expected, expected_phase)
        except (DecisionBudget, RecursionError) as exc:
            raise StrataError(f'symbolic HMR budget exceeded ({exc}); no certificate') from exc

    def _check(self, inputs, registers, expected, expected_phase):
        dag = self.dag
        # Keep global apply accounting while reclaiming previous input's roots.
        dag.collect([])
        state, bits = {}, {}
        for register in registers:
            target = state if register.namespace == 'quantum' else bits
            target.update((wire, (inputs[register.name] >> j) & 1)
                          for j, wire in enumerate(register.wires))
        phase, stack, measurement = [0] * self.phase_bits, [], 0
        # Private classical arithmetic may be huge but irrelevant on this
        # boundary input. Keep its exact expressions lazy until an observable
        # quantum/phase effect or declared classical output needs a decision.
        expressions, converted = BooleanDAG(), {0: 0, 1: 1}
        xor, mul = dag.xor, dag.product

        def decision(root):
            if root not in converted:
                node = expressions.nodes[root]
                if node['op'] == 'variable':
                    value = dag.variable(int(node['variable'][1:]))
                else:
                    operation = xor if node['op'] == 'xor' else mul
                    value = 0 if node['op'] == 'xor' else 1
                    for child in node['arguments']:
                        value = operation(value, decision(child))
                converted[root] = value
            return converted[root]

        def add_phase(active, coefficient):
            coefficient = int(coefficient * self.denominator) % (2 * self.denominator)
            add_phase_bits(dag, phase, active, coefficient)

        def branch_for(root):
            assignment = dag.witness(root)
            return assignment, sum(value << m for m, value in assignment.items())

        def counterexample(root):
            assignment, branch = branch_for(root)
            mismatch = {}
            for register, want in zip(registers, expected):
                values = state if register.namespace == 'quantum' else bits
                actual = sum(dag.evaluate(values.get(wire, 0) if register.namespace == 'quantum'
                                          else decision(values.get(wire, 0)), assignment) << j
                             for j, wire in enumerate(register.wires))
                if actual != want:
                    mismatch[register.name] = dict(actual=actual, expected=want)
            actual_phase = Fraction(sum(dag.evaluate(root, assignment) << j
                                        for j, root in enumerate(phase)), self.denominator)
            return dict(input=inputs, hmr_branch=branch, values=mismatch,
                        phase=str(actual_phase), expected_phase=str(expected_phase))

        for at, words in self.program.affine_ops:
            self.check_time()
            self.operations += 1
            if self.operations > self.limits['max_operations']:
                raise DecisionBudget('symbolic_operation_budget')
            self.expression_nodes = max(self.expression_nodes, len(expressions.nodes))
            if len(expressions.nodes) > self.limits['max_nodes']:
                raise DecisionBudget('symbolic_expression_node_budget')
            if self.operations % 1024 == 0 and len(dag.nodes) > self.limits['max_nodes'] * 3 // 4:
                converted = {0: 0, 1: 1}
                dag.collect([*state.values(), *phase])
            op = words[0]
            if op in {'REGISTER', 'APPEND_TO_REGISTER', 'INFO_TAG'}:
                continue
            active = bits.get(int(words[-1][1:]), 0) if 'if' in words else 1
            if op == 'PUSH_CONDITION':
                stack.append(active)
                continue
            if op == 'POP_CONDITION':
                if stack:
                    stack.pop()
                continue
            for condition in stack:
                active = expressions.product(active, condition)
            args = words[1:words.index('if')] if 'if' in words else words[1:]
            qs = [int(word[1:]) for word in args if word.startswith('q')]
            if op.startswith('BIT_'):
                bit = int(args[0][1:])
                old = bits.get(bit, 0)
                bits[bit] = expressions.xor(old, active if op == 'BIT_INVERT' else
                                expressions.product(active, expressions.xor(old, int(op == 'BIT_STORE1'))))
            elif op == 'HMR':
                q, bit = qs[0], int(args[1][1:])
                outcome = expressions.variable(f'm{measurement}') if active else 0
                measurement += 1
                old, old_bit = state[q], bits.get(bit, 0)
                if old:
                    effect = mul(decision(active), old)
                    add_phase(mul(effect, decision(outcome)), 1)
                    state[q] = xor(old, effect)
                bits[bit] = expressions.xor(old_bit, expressions.product(active, expressions.xor(old_bit, outcome)))
            elif op == 'R':
                bad = mul(decision(active), state[qs[0]]) if state[qs[0]] else 0
                if bad:
                    _, branch = branch_for(bad)
                    return dict(input=inputs, hmr_branch=branch, instruction=at,
                                reset_wire=qs[0], reason='reset_nonzero',
                                values={f'q{qs[0]}': dict(actual=1, expected=0)})
            elif op == 'X':
                state[qs[0]] = xor(state[qs[0]], decision(active))
            elif op in {'CX', 'CCX'}:
                delta = 1
                for q in qs[:-1]:
                    delta = mul(delta, state[q])
                if delta:
                    state[qs[-1]] = xor(state[qs[-1]], mul(decision(active), delta))
            elif op in {'SWAP', 'CSWAP'}:
                a, b = qs[-2:]
                delta = xor(state[a], state[b])
                if op == 'CSWAP':
                    delta = mul(delta, state[qs[0]])
                if delta:
                    delta = mul(decision(active), delta)
                    state[a], state[b] = xor(state[a], delta), xor(state[b], delta)
            elif op in {'NEG', 'Z', 'CZ', 'CCZ', 'Z_POW'}:
                delta = 1
                for q in qs:
                    delta = mul(delta, state[q])
                if delta:
                    add_phase(mul(decision(active), delta), self.exponents[at] if op == 'Z_POW' else 1)
            else:
                raise StrataError('unsupported symbolic HMR instruction; no certificate')
        for register, want in zip(registers, expected):
            values = state if register.namespace == 'quantum' else bits
            for j, wire in enumerate(register.wires):
                actual = values.get(wire, 0)
                if register.namespace == 'classical':
                    actual = decision(actual)
                bad = xor(actual, (want >> j) & 1)
                if bad:
                    return counterexample(bad)
        self.expression_nodes = max(self.expression_nodes, len(expressions.nodes))
        if self.expression_nodes > self.limits['max_nodes']:
            raise DecisionBudget('symbolic_expression_node_budget')
        scaled_phase = expected_phase * self.denominator
        if scaled_phase.denominator != 1:
            return counterexample(1)
        for j, actual in enumerate(phase):
            bad = xor(actual, (int(scaled_phase) >> j) & 1)
            if bad:
                return counterexample(bad)
        self.check_time()
        self.proved_cases += 1
        return None
