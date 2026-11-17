"""Count distinct verified, cheaper native designs under a fixed parent contract."""
from collections import defaultdict
import hashlib
import re

import kickmix as km
from .semantic import _parse


def design_key(circuit, parent_bits, private):
    """Ignore display metadata and canonicalize newly allocated private-bit names."""
    private = set(private)
    names = {}
    def rename(match):
        bit = int(match[1])
        if bit >= parent_bits and bit in private:
            return names.setdefault(bit, 'private'+str(len(names)))
        return match[0]
    body = []
    for line in str(circuit).splitlines():
        if line.split()[0] in {'APPEND_TO_REGISTER','REGISTER','INFO_TAG'}:
            continue
        body.append(re.sub(r'\bb(\d+)\b',rename,line))
    content = str(circuit.num_qubits)+'\n'+'\n'.join(body)
    return hashlib.sha256(content.encode()).hexdigest()


def nonlinear_function_key(circuit, clean):
    """Group small architectures by the exact nonlinear functions they actually pay to compute.

    Every signature uses the original arbitrary input coordinates. Free reversible
    adapters, helper numbering, repeat uses and reordering of the same products do
    not split a group. This deliberately coarsens native programs; it is not an
    equivalence test or proof that two algorithms are mathematically different.
    """
    if len(circuit) > 512:
        return None
    ops = [_parse(line) for line in str(circuit).splitlines()]
    ops = [item for item in ops if item['status'] != 'metadata']
    supported = {'X','CX','CCX','SWAP','CSWAP','NEG','Z','CZ','CCZ'}
    if any(item['status'] != 'supported' or item['condition'] or item['op'] not in supported for item in ops):
        return None
    wires = {q for item in ops for q in item['wires']}
    public = sorted(wires - set(clean))
    if len(public) > 10:
        return None
    # Bit i is the value of this wire on basis input i. Include physical input
    # identifiers in each function's support so unused or adapted wires cannot
    # make unrelated functions share a key.
    rows = 1 << len(public)
    ones = (1 << rows) - 1
    values = dict.fromkeys(wires,0)
    low_masks = []
    for j,q in enumerate(public):
        half = 1 << j
        values[q] = (((1 << half) - 1) << half) * (ones // ((1 << (2*half))-1))
        low_masks.append(ones ^ values[q])
    terms = set()
    visited = set()
    def record(role, word):
        if (role,word) in visited:
            return
        visited.add((role,word))
        active = [q for j,q in enumerate(public) if (word ^ (word >> (1 << j))) & low_masks[j]]
        if len(active) >= 2:
            reduced = word
            if len(active) != len(public):
                omitted = sum(1 << j for j,q in enumerate(public) if q not in active)
                indices = (i for i in range(rows) if not i & omitted)
                reduced = sum(((word >> index) & 1) << bit for bit,index in enumerate(indices))
            terms.add((role,tuple(active),reduced))
    for item in ops:
        op, q = item['op'], item['wires']
        if op == 'X':
            values[q[0]] ^= ones
        elif op == 'CX':
            values[q[1]] ^= values[q[0]]
        elif op == 'SWAP':
            values[q[0]],values[q[1]] = values[q[1]],values[q[0]]
        elif op in {'CCX','CCZ'}:
            product = values[q[0]] & values[q[1]]
            if op == 'CCZ':
                record('sign', product & values[q[2]])
            else:
                record('value',product)
                values[q[2]] ^= product
        elif op == 'CSWAP':
            product = values[q[0]] & (values[q[1]] ^ values[q[2]])
            record('value',product)
            values[q[1]] ^= product
            values[q[2]] ^= product
    return hashlib.sha256(repr(sorted(terms)).encode()).hexdigest()


def summarize(states, baseline, witnesses, parent_bits, clean=()):
    designs = {}
    families = defaultdict(set)
    coordinates = ('Q','toffoli','t','rotations','instructions','reaction_depth')
    points = set()
    nonlinear_classes = set()
    unclassified = 0
    for state in states:
        mechanisms = witnesses.get(state['id'],set())
        if state['cost'] >= baseline['cost'] or not mechanisms:
            continue
        circuit = km.Circuit.from_file(state['path'])
        key = design_key(circuit,parent_bits,state.get('private_bits',()))
        if key not in designs:
            function_key = nonlinear_function_key(circuit, clean)
            if function_key is None:
                unclassified += 1
            else:
                nonlinear_classes.add(function_key)
            designs[key] = dict(id=state['id'],cost=state['cost'],qualifications=set(),
                                nonlinear_function_class=function_key)
        row = designs[key]
        row['qualifications'].update(mechanisms)
        points.add(tuple(state['resources'][k] for k in coordinates))
        for kind in mechanisms:
            families[kind].add(key)
    for row in designs.values():
        row['qualifications']=sorted(row['qualifications'])
    frontier = [p for p in sorted(points) if not any(other!=p and all(x<=y for x,y in zip(other,p))
                                                     for other in points)]
    return dict(unique_verified_cheaper_native_designs=len(designs),qualification_family_count=len(families),
                unique_resource_points=len(points),pareto_resource_points=[dict(zip(coordinates,p)) for p in frontier],
                distinct_costs=sorted({row['cost'] for row in designs.values()}),
                distinct_nonlinear_function_sets=len(nonlinear_classes),
                designs_outside_nonlinear_classification=unclassified,
                nonlinear_class_scope='sets of exact paid product/sign functions in public inputs, ignoring clean wire placement, order and repeated uses; bounded to supported unmeasured circuits, 10 active public inputs and 512 operations',
                designs_by_qualification={kind:len(values) for kind,values in sorted(families.items())},
                designs=sorted(designs.values(),key=lambda r:(r['cost'],r['id'])),
                scope='supplied objective and boundary; native bytes modulo display metadata and fresh private-bit naming; equivalent schedules may remain different designs')



def constructor_comparison(attempts, baseline_cost):
    """Compare independently checked native constructions against the supplied cost."""
    families = defaultdict(list)
    for attempt in attempts:
        families[attempt['mechanism']].append(attempt)
    summary = {}
    for mechanism, trials in sorted(families.items()):
        deltas = [trial['cumulative_cost_delta'] for trial in trials if trial['status'] == 'verified']
        summary[mechanism] = dict(
            attempted=len(trials), independently_verified=len(deltas),
            best_verified_cost=baseline_cost + min(deltas) if deltas else None,
            best_delta_from_source=min(deltas) if deltas else None,
            source_cheaper_than_all_verified=bool(deltas) and min(deltas) > 0,
        )
    return dict(source_cost=baseline_cost, constructors=summary, scope='supplied artifact and objective')
