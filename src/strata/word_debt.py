"""Exact flat private HMR/sign cleanup by individual debt identities."""
import operator
from functools import reduce

from .semantic import _parse


def _layout(original, candidate, private, limits):
    """Verify every unchanged gate, each prescribed toggle, and scratch noninterference."""
    if max(len(original), len(candidate)) > limits['max_operations']:
        return None
    records = []
    epochs = {}
    last = {}
    expected = []
    wires = set()
    for index, line in enumerate(original):
        item = _parse(line)
        op = item['op']
        if item['status'] == 'metadata':
            expected.append(line)
            continue
        if item['status'] != 'supported':
            return None
        qs = tuple(item['wires'])
        wires.update(qs)
        bit = item.get('condition')
        if op == 'HMR' and not bit:
            label = item['bits'][0]
            if label not in private:
                return None
            epoch = (index, qs[0], label)
            epochs[label] = epoch
            records.append((index, op, qs, epoch))
        elif bit and op in ('NEG', 'Z', 'CZ'):
            epoch = epochs.get(bit)
            if epoch is None:
                return None
            _, wire, _ = epoch
            gate = ('X', 'CX', 'CCX')[len(qs)]
            expected.append(' '.join([gate, *(f'q{q}' for q in qs), f'q{wire}']))
            last[epoch] = index
            records.append((index, 'repair', qs, epoch))
        elif not bit and op in ('X', 'CX', 'CCX', 'SWAP', 'R', 'NEG', 'Z', 'CZ', 'CCZ'):
            expected.append(line)
            records.append((index, op, qs, None))
        else:
            return None
    if expected != candidate or len(wires) + len(epochs) + len(last) > min(512, limits['max_variables']):
        return None
    # Only active scratch differs between the measured and reversible trajectories.
    # All preserved operations, including phases, must avoid it until repayment.
    active = set()
    for index, op, qs, epoch in records:
        if active.intersection(qs):
            return None
        if op == 'HMR':
            if epoch not in last:
                return None
            active.add(epoch[1])
        elif op == 'repair' and index == last[epoch]:
            active.remove(epoch[1])
    return records, wires, last


def _obligations(records, state, last, one, xor, product):
    """Accumulate m*(old_value XOR phase terms) for each private HMR epoch."""
    debts = {}
    for index, op, qs, epoch in records:
        if op == 'HMR':
            debts[epoch] = state[qs[0]]
            state[qs[0]] = 0
        elif op == 'repair':
            term = reduce(product, (state[q] for q in qs), one)
            debts[epoch] = xor(debts[epoch], term)
            if index == last[epoch]:
                yield epoch[0], 'phase_debt', debts.pop(epoch)
        elif op == 'R':
            yield index, 'zero_reset', state[qs[0]]
            state[qs[0]] = 0
        elif op in ('X', 'CX', 'CCX'):
            target = qs[-1]
            term = reduce(product, (state[q] for q in qs[:-1]), one)
            state[target] = xor(state[target], term)
        elif op == 'SWAP':
            a, b = qs
            state[a], state[b] = state[b], state[a]


def _projection(bit, rows):
    stride = 1 << bit
    pattern = ((1 << stride) - 1) << stride
    period = stride * 2
    while period < rows:
        pattern |= pattern << period
        period *= 2
    return pattern


def prove_word_cleanup(original, candidate, clean_inputs, internal_bits, limits):
    """Prove every projected basis assignment, each private outcome, and ordinary resets.

    The first pass over-approximates inputs of every measured value and repair.
    Among identical common operations only scratch differences can survive between
    edits. The layout check rules out all reads of those differences. A vanished
    debt both erases the reversible scratch and cancels the source's outcome sign.
    """
    private = {f'b{b}' for b in internal_bits}
    parsed = _layout(original, candidate, private, limits)
    if parsed is None:
        return None
    records, wires, last = parsed
    if not last:
        return None
    public = sorted(wires.difference(clean_inputs))
    lookup = {q: 1 << i for i, q in enumerate(public)}
    dependencies = {q: lookup.get(q, 0) for q in wires}
    # Boolean union is a conservative dependency bound for both XOR and AND.
    support = 0
    for _, _, bits in _obligations(records, dependencies, last, 0, operator.or_, operator.or_):
        support |= bits
    inputs = [q for i, q in enumerate(public) if support & (1 << i)]
    rows = 1 << len(inputs)
    word_steps = 4 * len(records) * ((rows + 63) // 64)
    if rows > min(65536, limits['max_terms']) or word_steps > 32 * limits['max_product_terms']:
        return None
    values = {q: 0 for q in wires}
    for bit, q in enumerate(inputs):
        values[q] = _projection(bit, rows)
    checks = []
    for index, kind, mask in _obligations(records, values, last, (1 << rows) - 1, operator.xor, operator.and_):
        if mask:
            return None
        checks.append(dict(at=index, obligation=kind))
    return dict(kind='exact_private_word_cleanup', equivalent=True, status='equivalent', reason='verified',
                outcome_independence_proved=True, quantum_channel_equivalence=False,
                scope='flat private HMR erasures; all declared public basis inputs and exact sign',
                clean_inputs=sorted(clean_inputs), internal_bits=sorted(internal_bits),
                original_outcomes=len(last), candidate_outcomes=0,
                checked_input_wires=inputs, assignments=rows, obligations=checks, packed_word_step_bound=word_steps)
