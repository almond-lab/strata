"""Prove conservative compact boundary promises for the registered chemistry loaders."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import kickmix as km
import numpy as np

from strata.promised_domain import parse_address_promise
from strata.semantic import _parse


ADDRESSES = 1220
ALL = (1 << ADDRESSES) - 1
ZERO, ONE, UNKNOWN = (ALL, 0), (0, ALL), (0, 0)
WORDS = [f'word_{i}' for i in range(5)]
REGISTERS = ['selection', *WORDS]


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _and(a, b):
    return a[0] | b[0], a[1] & b[1]


def _xor(a, b):
    return (a[0] & b[0]) | (a[1] & b[1]), (a[0] & b[1]) | (a[1] & b[0])


def _select(condition, yes, no):
    # If the control is unknown, equal proven output bits still survive.
    return tuple((condition[1] & yes[i]) | (condition[0] & no[i]) | (yes[i] & no[i]) for i in (0, 1))


def _bits(values):
    return sum(int(bit) << i for i, bit in enumerate(values))


def derive(tape, start, active_physical_qs, clean_local, work):
    """Write an address_promise only if its admitted set is a proper restriction.

    Each integer mask represents all 1,220 addresses simultaneously. At bit a,
    'known zero' or 'known one' is true only when that wire holds the given
    value for *every* choice of the forty dirty inputs and measurement outcomes.
    If neither holds, the abstract value is unknown. There is no enumeration
    or independence approximation of dirty words in the universal proof.
    """
    active, clean = list(active_physical_qs), sorted(clean_local)
    work = Path(work)
    path = work / 'domain.json'
    if path.exists():
        raise ValueError('Expected a fresh directory for the loader promise')
    if (active != sorted(set(active)) or any(type(q) is not int or not 0 <= q < tape.width for q in active) or
            len(clean) != len(set(clean)) or any(type(i) is not int or not 0 <= i < len(active) for i in clean)):
        raise ValueError('Expected distinct compacted loader wires in physical order and in-range clean indices')
    layout = tape.register_qubit_order
    if (tape.register_names != REGISTERS or set(layout) != set(REGISTERS) or len(layout['selection']) != 11 or
            any(len(layout[name]) != 8 for name in WORDS)):
        raise ValueError('Loader requires one 11-bit address and five eight-bit quantum words')
    public = [q for name in REGISTERS for q in layout[name]]
    if len(set(public)) != 51 or any(type(q) is not int or not 0 <= q < tape.width for q in public):
        raise ValueError('Expected distinct physical quantum input bindings')
    if not 0 <= start <= tape.ops:
        raise ValueError('Loader boundary exceeds the native tape')
    summary = dict(applied=False, legal_parent_addresses=ADDRESSES, dirty_word_bits=40,
                   active_qubits=len(active), clean_local_count=len(clean), prefix_operations=start,
                   entry_contract_sha256=_digest(dict(registers=layout, width=tape.width,
                       address_interval=[0, ADDRESSES-1], word_inputs='arbitrary', other_qubits='zero')),
                   proof='per-legal-address known-zero and known-one induction on native basis values',
                   numeric_prescan='native baseline prefix only; no candidate or suffix used to derive this projection',
                   scope='every legal parent address, arbitrary 40-bit word and independent outcomes; '
                         'remaining local wires arbitrary except for explicitly proved clean inputs')
    def finish(reason):
        return {'loader_domain': dict(summary, reason=reason)}

    state = [ZERO] * tape.width
    words = [q for name in WORDS for q in layout[name]]
    for q in words:
        state[q] = UNKNOWN
    addresses = np.arange(ADDRESSES)
    sim = km.Simulator(batch_size=ADDRESSES, seed=19471)
    for j, q in enumerate(layout['selection']):
        row = ((addresses >> j) & 1).astype(bool)
        mask = _bits(row)
        state[q] = (ALL ^ mask, mask)
        sim.write_across_shots(km.q(q), row)
    rng = np.random.default_rng(19471)
    for q in words:
        sim.write_across_shots(km.q(q), rng.integers(0, 2, size=ADDRESSES).astype(bool))
    classical, stack, seen, prefix = {}, [], Counter(), hashlib.sha256()
    for first in range(0, start, 65536):
        last = min(start, first + 65536)
        native = tape.span(first, last)
        lines = str(native).splitlines()
        if len(lines) != last-first:
            raise ValueError('Prefix decode did not preserve native operation coordinates')
        for at, line in enumerate(lines, first):
            prefix.update(line.encode() + b'\n')
            item = _parse(line)
            op, qs = item['op'], item['wires']
            seen[op] += 1
            if item['status'] != 'supported' and item.get('reason') != 'unsupported_fractional_phase':
                summary['stopped_at'] = at
                return finish('unsupported_native_prefix')
            cond = classical.get(item['condition'], UNKNOWN) if item['condition'] is not None else ONE
            if op == 'PUSH_CONDITION':
                # Capture the current value. A later write to that classical
                # bit must not retroactively change an enclosing scope.
                stack.append(_and(stack[-1], cond) if stack else cond)
                continue
            if op == 'POP_CONDITION':
                if not stack:
                    return finish('unbalanced_prefix_condition')
                stack.pop()
                continue
            if stack:
                cond = _and(cond, stack[-1])
            if op in {'X', 'CX', 'CCX'}:
                toggle = cond
                for q in qs[:-1]:
                    toggle = _and(toggle, state[q])
                state[qs[-1]] = _xor(state[qs[-1]], toggle)
            elif op in {'SWAP', 'CSWAP'}:
                switch = _and(cond, state[qs[0]]) if op == 'CSWAP' else cond
                a, b = qs[-2:]
                state[a], state[b] = _select(switch, state[b], state[a]), _select(switch, state[a], state[b])
            elif op == 'HMR':
                bit = item['bits'][0]
                state[qs[0]] = _select(cond, ZERO, state[qs[0]])
                classical[bit] = _select(cond, UNKNOWN, classical.get(bit, UNKNOWN))
            elif op == 'R':
                if (ALL ^ cond[0]) & (ALL ^ state[qs[0]][0]):
                    summary['stopped_at'] = at
                    return finish('prefix_reset_zero_unproved')
            elif op in {'BIT_STORE0', 'BIT_STORE1', 'BIT_INVERT'}:
                bit = item['bits'][0]
                old = classical.get(bit, UNKNOWN)
                value = (old[1], old[0]) if op == 'BIT_INVERT' else ONE if op == 'BIT_STORE1' else ZERO
                classical[bit] = _select(cond, value, old)
            elif op not in {'NEG', 'Z', 'CZ', 'CCZ', 'Z_POW'}:
                summary['stopped_at'] = at
                return finish('unsupported_native_prefix')
        sim.do(native)
    summary.update(prefix_canonical_sha256=prefix.hexdigest(), prefix_op_counts=dict(sorted(seen.items())))
    if stack:
        return finish('slice_inherits_parent_condition')
    if any(state[active[i]] != ZERO for i in clean):
        return finish('local_clean_claim_not_proved_by_prefix')
    # Compare every asserted zero/one bit, including facts that hold on only
    # some addresses, to an independently executed SDK prefix.
    checked = 0
    for q in active:
        zero, one = state[q]
        assert not zero & one
        checked += (zero | one).bit_count()
        actual = _bits(sim.read_across_shots(km.q(q)))
        if (actual & zero) or ((ALL ^ actual) & one):
            raise ValueError('Native replay contradicted a proved prefix bit')
    selected = [i for i, q in enumerate(active) if i not in clean and (state[q][0] | state[q][1]) == ALL]
    other = sorted(set(range(len(active))) - set(selected) - set(clean))
    rows = sorted({sum(((state[active[i]][1] >> a) & 1) << j for j, i in enumerate(selected))
                   for a in range(ADDRESSES)})
    summary.update(deterministic_address_bits=len(selected), projected_address_count=len(rows),
                   remaining_arbitrary_qubits=len(other), full_address_count=1 << len(selected),
                   native_cross_check=dict(addresses=ADDRESSES, dirty_vectors=ADDRESSES,
                       asserted_bits_checked=checked, seed=19471, status='passed',
                       scope='one deterministic-seed dirty vector per address; separate from the all-dirty proof'))
    if not selected or len(rows) == 1 << len(selected):
        return finish('projected_address_support_is_unrestricted')
    spec = dict(version=1, type='address_promise', address_wires=selected, clean_wires=clean,
                arbitrary_wires=other, addresses=rows)
    if rows[-1] - rows[0] + 1 == len(rows):
        spec['interval'] = [rows[0], rows[-1]]
        del spec['addresses']
    native_source = km.Circuit.from_file(str(work / 'source.kmx'))
    bound = parse_address_promise(spec, native_source, clean_inputs=clean)
    path.write_text(json.dumps(spec, sort_keys=True) + '\n')
    summary.update(applied=True, domain_hash=bound['domain_hash'], domain_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   admitted_local_assignments=bound['assignment_count'])
    return finish('proper_projected_address_support')
