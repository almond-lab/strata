"""Recover stored products and their consumers without expanding a lookup ANF."""
from bisect import bisect_left
from collections import defaultdict

import kickmix as km

from .semantic import _advance_zero_facts, _parse


def _writes(item):
    op, wires = item['op'], item['wires']
    if op in {'X', 'CX', 'CCX', 'HMR', 'R'}:
        return wires[-1:]
    if op in {'SWAP', 'CSWAP'}:
        return wires[-2:]
    return ()


class _ValueSignatures:
    """Sparse XOR signatures treating products as hash-consed opaque atoms."""

    def __init__(self, clean):
        self.values = dict.fromkeys(clean, frozenset())
        self.products = {}
        self.next_atom = 1  # Atom zero represents the constant one.

    def fresh(self):
        result = frozenset({self.next_atom})
        self.next_atom += 1
        return result

    def value(self, q):
        if q not in self.values:
            self.values[q] = self.fresh()
        return self.values[q]

    def advance(self, item, plain):
        op, wires = item['op'], item.get('wires', ())
        if item['status'] != 'supported':
            self.values.clear()
        elif not plain:
            for q in _writes(item):
                self.values[q] = self.fresh()
        elif op == 'X':
            self.values[wires[0]] = self.value(wires[0]) ^ {0}
        elif op in {'CX', 'CCX'}:
            operands = [self.value(q) for q in wires[:-1]]
            if op == 'CX':
                added = operands[0]
            elif not all(operands):
                added = frozenset()
            elif operands[0] == operands[1]:
                added = operands[0]
            else:
                key = frozenset(operands)
                if key not in self.products:
                    self.products[key] = self.fresh()
                added = self.products[key]
            target = wires[-1]
            value = self.value(target) ^ added
            self.values[target] = value if len(value) <= 128 else self.fresh()
        elif op == 'SWAP':
            a, b = wires
            self.values[a], self.values[b] = self.value(b), self.value(a)
        elif op in {'HMR', 'R'}:
            self.values[wires[0]] = frozenset()
        else:
            for q in _writes(item):
                self.values[q] = self.fresh()


def product_lifetimes(lines, clean_inputs=()):
    """Record syntactically matched CCX/CCX and CCX/HMR product lifetimes."""
    items, uses, lifetimes = [], defaultdict(list), []
    last_write, active = {}, {}
    zeros = {q: {} for q in clean_inputs}
    signatures = _ValueSignatures(zeros)
    depth = 0
    for at, line in enumerate(lines):
        item = _parse(line)
        items.append(item)
        op, wires = item['op'], item.get('wires', ())
        if item['status'] == 'metadata':
            continue
        for q in wires:
            uses[q].append(at)
        if item['status'] != 'supported':
            active.clear()
        plain = item['status'] == 'supported' and not depth and not item.get('condition')
        if op == 'PUSH_CONDITION':
            depth += 1
        elif op == 'POP_CONDITION':
            depth -= 1
        restored = None
        for q in _writes(item):
            previous = active.pop(q, None)
            if plain and previous:
                start, controls, clean, first, saved_values = previous
                unchanged = all(last_write.get(c, -1) < start for c in controls)
                same_values = saved_values == [signatures.value(c) for c in controls]
                inverse = op == 'CCX' and sorted(wires[:2]) == controls
                if same_values and (inverse or op == 'HMR'):
                    lifetimes.append(dict(start=start, stop=at+1, wire=q, controls=controls,
                        consumers=uses[q][first:-1], clean_entry=clean, controls_untouched=unchanged,
                        cleanup=op, outcome=item['bits'][0] if op == 'HMR' else None))
                    if inverse:
                        if clean:
                            restored = q
                        last_write[q] = at
                        continue
            if plain and op == 'CCX':
                controls = sorted(wires[:2])
                active[q] = (at, controls, q in zeros, len(uses[q]), [signatures.value(c) for c in controls])
            last_write[q] = at
        signatures.advance(item, plain)
        _advance_zero_facts(item, zeros, depth, at)
        if restored is not None:
            zeros[restored] = dict(kind='matched_product_restore', origin_at=at, origin_wire=restored)
    return items, lifetimes


def measured_value_surrogate(circuit, clean_inputs, internal_bits, proof_limits):
    """Replace standard, private measured product erasures for proposal generation."""
    from .measurement_equivalence import equivalent_independent_outputs

    lines = str(circuit).splitlines()
    items, lifetimes = product_lifetimes(lines, clean_inputs)
    private = {f'b{b}' for b in internal_bits}
    bit_uses = defaultdict(list)
    for i, item in enumerate(items):
        if item['status'] == 'metadata':
            continue
        for bit in set(item.get('bits', ())) | ({item['condition']} if item.get('condition') else set()):
            bit_uses[bit].append(i)
    edits, matched = {}, []
    for record in lifetimes:
        outcome, after = record['outcome'], record['stop']
        if not record['clean_entry'] or outcome not in private:
            continue
        references = bit_uses[outcome]
        index = bisect_left(references, after-1)
        if index + 1 >= len(references):
            continue
        repair_at = references[index+1]
        if index + 2 < len(references):
            overwrite = items[references[index+2]]
            if (overwrite['op'] not in {'HMR', 'BIT_STORE0', 'BIT_STORE1'} or overwrite['condition']
                    or outcome not in overwrite['bits']):
                continue
        repair = items[repair_at]
        if (repair['status'] != 'supported' or repair['op'] != 'CZ' or repair['condition'] != outcome
                or sorted(repair['wires']) != record['controls']):
            continue
        edits[after-1] = lines[record['start']]
        edits[repair_at] = None
        matched.append(dict(record, phase_repair=repair_at))
    if not matched:
        first = next((i for i, item in enumerate(items)
                      if item['status'] != 'metadata' and (
                          item.get('condition') or item['op'] in {'HMR', 'PUSH_CONDITION', 'POP_CONDITION'}
                          or item['op'].startswith('BIT_'))), None)
        return None, dict(status='not_applicable', blocking_text_line=first,
                          instruction=lines[first] if first is not None else None,
                          reason='no certified private HMR with matching live controls and conditioned CZ')
    body = [edits.get(i, line) for i, line in enumerate(lines)]
    body = [line for line in body if line is not None]
    if any((item := _parse(line))['status'] != 'metadata' and (
            item.get('condition') or item['op'] in {'HMR', 'PUSH_CONDITION', 'POP_CONDITION'}
            or item['op'].startswith('BIT_')) for line in body):
        return None, dict(status='not_applicable', reason='unmatched classical operation remains after HMR normalization',
                          matched=matched)
    proof = equivalent_independent_outputs(lines, body, clean_inputs=clean_inputs, internal_bits=internal_bits,
        max_operations=proof_limits['max_operations'], max_variables=proof_limits['max_variables'],
        max_nodes=4*proof_limits['max_terms'], max_apply_steps=32*proof_limits['max_product_terms'])
    evidence = dict(status='verified' if proof.get('equivalent') is True else 'unknown', matched=matched, proof=proof)
    return (km.Circuit('\n'.join(body)) if proof.get('equivalent') is True else None), evidence


def terminal_cache_fusion(circuit, clean_inputs):
    """Keep read-shared cache products and inline copy-only terminal products."""
    lines = str(circuit).splitlines()
    items, lifetimes = product_lifetimes(lines, clean_inputs)
    edits, fused = {}, []
    for record in lifetimes:
        if not record['clean_entry'] or not record['controls_untouched'] or record['cleanup'] != 'CCX':
            continue
        a, b = record['controls']
        product = record['wire']
        consumers = record['consumers']
        # Up to two direct updates cost at most the stored compute/uncompute.
        if len(consumers) > 2 or any(items[i]['op'] != 'CX' or items[i].get('condition') or
                items[i]['wires'][0] != product or items[i]['wires'][1] in (a, b)
                for i in consumers):
            continue
        edits[record['start']] = None
        edits[record['stop']-1] = None
        for i in consumers:
            edits[i] = f'CCX q{a} q{b} q{items[i]["wires"][1]}'
        fused.append(record)
    if not fused:
        return dict(status='not_applicable', reason='no restored copy-only terminal product of non-increasing paid cost')
    body = [edits.get(i, line) for i, line in enumerate(lines)]
    return dict(status='constructed', candidate='\n'.join(line for line in body if line is not None),
                structural_witness=dict(terminal_products_fused=True, product_lifetimes=fused))
