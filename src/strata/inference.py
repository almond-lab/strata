"""Exact, bounded value queries and executable information-compression witnesses."""
from itertools import combinations
from time import monotonic
from .model import StrataError, digest
from .semantic import analyze_window
from .domains import load_domain, window_lines, touched_wires, execute_assignment, _Unknown, WindowLimit


def _term_key(term):
    return len(term), tuple((type(x).__name__, str(x)) for x in sorted(term, key=lambda x: (type(x).__name__, str(x))))


def _poly(row):
    return {frozenset(t) for t in row['monomials']}


def _multiply(a, b, limit):
    if len(a) * len(b) > limit:
        raise StrataError('functional query product budget exceeded')
    result = set()
    for x in a:
        for y in b:
            term = x | y
            if term in result:
                result.remove(term)
            else:
                result.add(term)
    if len(result) > limit:
        raise StrataError('functional query term budget exceeded')
    return result


def _wires(circuit, wires, name='wire set'):
    values = list(wires)
    if len(set(values)) != len(values) or any(type(q) is not int or not 0 <= q < circuit.num_qubits for q in values):
        raise StrataError('invalid ' + name)
    return values


def functions_at(circuit, at=None, *, from_at=0, clean_inputs=(), domain=None,
                 max_variables=128, max_terms=65536, max_operations=10000,
                 max_product_terms=1048576, max_assignments=65536, max_outcomes=1024):
    """Recover values in [from_at, at), treating its initial boundary as arbitrary.

    A finite domain is a complete explicit boundary promise, not an assumption
    about an earlier prefix. Domain rows are independently executed and retained
    for exact finite relation/rank queries even when symbolic limits are reached.
    """
    try:
        lines, at = window_lines(circuit, from_at, at, max_lines=max_operations)
    except WindowLimit as exc:
        return dict(status='unknown', reason=str(exc), from_at=from_at, at=at)
    clean = set(_wires(circuit, clean_inputs, 'clean inputs'))
    base = dict(at=at, from_at=from_at, touched_wires=touched_wires(lines), clean_inputs=sorted(clean),
                window_hash=digest(dict(lines=lines, num_qubits=circuit.num_qubits,
                                        from_at=from_at, at=at, clean_inputs=sorted(clean))),
                scope='all arbitrary window boundary inputs except explicitly declared zero inputs')
    if domain is not None:
        bound = load_domain(domain, circuit, from_at=from_at, at=at,
                            max_assignments=max_assignments, max_operations=max_operations)
        if any(row[f'q{q}'] for row in bound['assignments'] for q in clean):
            raise StrataError('finite domain contradicts declared clean boundary inputs')
        rows = []
        import shlex
        count = sum(bool(w) and w[0] == 'HMR' for w in (shlex.split(line, comments=True) for line in lines))
        if type(max_outcomes) is not int or max_outcomes < 1:
            raise StrataError('max_outcomes must be positive')
        if (1 << count) > max_outcomes:
            return dict(base, status='unknown', reason='measurement_outcome_budget', domain_hash=bound['domain_hash'])
        try:
            for assignment in bound['assignments']:
                for encoded in range(1 << count):
                    outcomes = [(encoded >> i) & 1 for i in range(count)]
                    row = execute_assignment(lines, assignment, circuit.num_qubits, outcomes=outcomes, max_operations=max_operations)
                    rows.append({int(q): value for q, value in row['outputs'].items()})
        except (_Unknown, ValueError) as exc:
            return dict(base, status='unknown', reason=str(exc), domain_hash=bound['domain_hash'])
        base.update(scope=bound['scope'], domain_hash=bound['domain_hash'],
                    assignment_count=bound['assignment_count'], outcome_count=1 << count,
                    finite_values=rows, all_input_equivalence=False,
                    measurement_variables=[f'm{i}' for i in range(count)],
                    classical_inputs=bound['classical_inputs'],
                    is_permutation=not any((words := shlex.split(line, comments=True)) and
                        (words[0] in {'HMR','R','BIT_STORE0','BIT_STORE1','BIT_INVERT','PUSH_CONDITION','POP_CONDITION'}
                         or 'if' in words) for line in lines),
                    semantic_scope='finite native outcome-parametric basis trajectories')
    sem = analyze_window(lines, max_variables=max_variables, max_terms=max_terms,
                         max_operations=max_operations, max_product_terms=max_product_terms,
                         clean_inputs=sorted(clean))
    if sem['status'] != 'exact':
        if domain is not None:
            return dict(base, status='exact', kind='finite_enumeration', symbolic_status=sem['status'])
        return dict(base, status='unknown', reason=sem.get('reason'))
    funcs = {q: _poly(sem['outputs'].get(str(q), {'monomials': [[q]]})) for q in range(circuit.num_qubits)}
    funcs = {q: {term for term in poly if not term & clean} for q, poly in funcs.items()}
    return dict(base, status='exact', functions=funcs,
                phase={t for t in _poly(sem['phase']) if not t & clean},
                is_permutation=sem.get('is_permutation', True),
                semantic_scope=sem.get('semantic_scope', 'exact_permutation_and_sign'),
                classical_inputs=sem.get('classical_inputs', []),
                measurement_variables=sem.get('measurement_variables', []))


def is_function_of(circuit, wire, at, support, *, degree=2, max_forms=4096,
                   timeout=5, clean_inputs=(), from_at=0, domain=None, **budgets):
    support = tuple(support)
    if len(set(support)) != len(support) or wire in support:
        raise StrataError('support must be distinct and exclude target')
    _wires(circuit, (wire, *support), 'query wires')
    if type(degree) is not int or not 0 <= degree <= 3 or type(max_forms) is not int or max_forms < 1 or timeout <= 0:
        raise StrataError('invalid functional-search budget')
    start = monotonic()
    family = dict(kind='Boolean polynomial in current support values', max_degree=degree, support=list(support))
    result = functions_at(circuit, at, from_at=from_at, domain=domain, clean_inputs=clean_inputs, **budgets)
    if result['status'] != 'exact':
        return dict(result, searched_family=family)
    finite = result.get('finite_values')
    target = ({i for i, row in enumerate(finite) if row[wire]} if finite is not None else result['functions'][wire])
    columns, forms = [], []
    try:
        for size in range(degree + 1):
            for form in combinations(support, size):
                if len(forms) >= max_forms or monotonic() - start > timeout:
                    return dict(status='unknown', reason='functional search budget', searched_family=family, forms=len(forms))
                if finite is not None:
                    poly = {i for i, row in enumerate(finite) if all(row[q] for q in form)}
                else:
                    poly = {frozenset()}
                    for q in form:
                        poly = _multiply(poly, result['functions'][q], budgets.get('max_terms', 65536))
                columns.append(poly)
                forms.append(form)
        basis = {}
        pivot_key = (lambda x: x) if finite is not None else _term_key
        def reduce(poly, coeff):
            poly = set(poly)
            while poly:
                pivot = max(poly, key=pivot_key)
                if pivot not in basis:
                    break
                row, mask = basis[pivot]
                poly ^= row
                coeff ^= mask
            return poly, coeff
        for index, poly in enumerate(columns):
            if monotonic() - start > timeout:
                return dict(status='unknown', reason='linear solve budget', searched_family=family)
            row, mask = reduce(poly, 1 << index)
            if row:
                basis[max(row, key=pivot_key)] = row, mask
        remainder, solution = reduce(target, 0)
        evidence = dict(kind='exact_finite' if finite is not None else 'symbolic', at=result['at'],
                        from_at=from_at, clean_inputs=result['clean_inputs'], forms_searched=len(forms),
                        linear_rank=len(basis), scope=result['scope'], window_hash=result['window_hash'])
        evidence['semantic_scope'] = result.get('semantic_scope', 'finite_basis_trajectories')
        if domain is not None:
            evidence.update(domain_hash=result['domain_hash'], assignment_count=result['assignment_count'])
        if remainder:
            return dict(status='no', searched_family=family, evidence=evidence,
                        reason='target is outside the exact span of the searched family')
        selected = [list(form) for i, form in enumerate(forms) if solution >> i & 1]
        check = set()
        for i, poly in enumerate(columns):
            if solution >> i & 1:
                check ^= poly
        if check != target:
            raise StrataError('functional witness failed replay')
        return dict(status='yes', wire=wire, at=result['at'], from_at=from_at,
                    form=dict(monomials=selected, expression=' ^ '.join('*'.join(f'q{q}' for q in t) or '1' for t in selected) or '0'),
                    searched_family=family, evidence=evidence,
                    witness_hash=digest(dict(wire=wire, form=selected, scope=evidence)))
    except StrataError as exc:
        return dict(status='unknown', reason=str(exc), searched_family=family)


def _patterns(result, selected, max_inputs):
    if type(max_inputs) is not int or max_inputs < 0:
        raise StrataError('max_inputs must be a nonnegative integer')
    if 'finite_values' in result:
        return {tuple(row[q] for q in selected) for row in result['finite_values']}
    funcs = [result['functions'][q] for q in selected]
    inputs = sorted({q for poly in funcs for term in poly for q in term}, key=lambda x: (type(x).__name__, str(x)))
    if len(inputs) > max_inputs:
        raise StrataError('distinct-state enumeration budget')
    patterns = set()
    for assignment in range(1 << len(inputs)):
        on = {q for i, q in enumerate(inputs) if assignment >> i & 1}
        patterns.add(tuple(sum(term <= on for term in poly) % 2 for poly in funcs))
    return patterns


def information_slack(circuit, at=None, *, from_at=0, wires=None, clean_inputs=(),
                      domain=None, max_inputs=16, **budgets):
    """Exact joint information bound; the default set is touched window wires."""
    try:
        lines, at = window_lines(circuit, from_at, at, max_lines=budgets.get('max_operations', 10000))
    except WindowLimit as exc:
        return dict(status='unknown', reason=str(exc), from_at=from_at, at=at)
    selected = _wires(circuit, touched_wires(lines) if wires is None else wires, 'slack wire set')
    result = functions_at(circuit, at, from_at=from_at, clean_inputs=clean_inputs, domain=domain, **budgets)
    if result['status'] != 'exact':
        return result
    if type(max_inputs) is not int or max_inputs < 0:
        raise StrataError('max_inputs must be a nonnegative integer')
    funcs = [result.get('functions', {}).get(q, set()) for q in selected]
    if domain is None and all(len(term) <= 1 for poly in funcs for term in poly):
        basis = {}
        key = lambda x: (type(x).__name__, str(x))
        for poly in funcs:
            row = {next(iter(term)) for term in poly if term}
            while row and max(row, key=key) in basis:
                row ^= basis[max(row, key=key)]
            if row:
                basis[max(row, key=key)] = row
        bits = len(basis)
        kind, distinct = 'exact_affine_rank', 1 << bits
    else:
        try:
            patterns = _patterns(result, selected, max_inputs)
        except StrataError as exc:
            return dict(status='unknown', reason=str(exc), from_at=from_at, at=at)
        distinct = len(patterns)
        bits, kind = (distinct - 1).bit_length(), 'exact_finite' if domain is not None else 'exact_full'
    answer = dict(status='exact', kind=kind, from_at=from_at, at=at, wires=selected,
                  distinct_states=distinct, information_bits=bits, slack_bits=len(selected) - bits,
                  clean_inputs=result['clean_inputs'], scope=result['scope'],
                  note='Information compression bound only; does not establish a free clean scratch wire or cheap encoder.')
    if domain is not None:
        answer.update(domain_hash=result['domain_hash'], assignment_count=result['assignment_count'], all_input_equivalence=False)
    return answer


def discover_dependencies(circuit, at=None, *, from_at=0, wires=None, max_support=3,
                          max_queries=256, **budgets):
    """Automatically search current-value linear and nonlinear dependencies."""
    try:
        lines, at = window_lines(circuit, from_at, at, max_lines=budgets.get('max_operations', 10000))
    except WindowLimit as exc:
        return dict(status='unknown', reason=str(exc), from_at=from_at, at=at)
    selected = _wires(circuit, touched_wires(lines) if wires is None else wires)
    if type(max_support) is not int or max_support < 0 or type(max_queries) is not int or max_queries < 1:
        raise StrataError('invalid dependency-search budget')
    relations, searched, bounded = [], 0, False
    for target in selected:
        found = False
        for size in range(min(max_support, len(selected) - 1) + 1):
            for support in combinations([q for q in selected if q != target], size):
                if searched >= max_queries:
                    bounded = True
                    break
                searched += 1
                row = is_function_of(circuit, target, at, support, from_at=from_at, **budgets)
                if row['status'] == 'yes':
                    relations.append(row)
                    found = True
                    break
            if bounded or found:
                break
        if bounded:
            break
    return dict(status='bounded' if bounded else 'complete_in_family', from_at=from_at, at=at,
                relations=relations, queries=searched, max_support=max_support,
                searched_family='automatic minimal-support Boolean polynomial dependencies')


def validate_compression(witness):
    """Check serialized tables and scoped zero codes; source reachability is separate."""
    width = len(witness['wires'])
    if (any(type(q) is not int or q < 0 for q in witness['wires']) or
            len(set(witness['wires'])) != width):
        raise StrataError('invalid compression wire layout')
    bits = witness['information_bits']
    if type(bits) is not int or not 0 <= bits <= width:
        raise StrataError('invalid compressed information width')
    expected = set(range(1 << width))
    for name in ('encoder', 'decoder'):
        table = witness[name]
        if (len(table) != 1 << width or any(type(value) is not int for value in table)
                or set(table) != expected):
            raise StrataError('compression witness is not a complete permutation')
    if any(witness['decoder'][target] != source for source, target in enumerate(witness['encoder'])):
        raise StrataError('compression encoder and decoder are not mutual inverses')
    if any(type(value) is not int or value not in expected or witness['encoder'][value] >> bits
           for value in witness['support']):
        raise StrataError('compression support does not map to the declared zero suffix')
    return dict(status='verified', scope='complete inverse tables and declared support codes; source support must be separately bound')


def execute_compression(witness, values, *, inverse=False):
    """Execute the serialized reversible permutation after checking both tables."""
    validate_compression(witness)
    width = len(witness['wires'])
    table = witness['decoder'] if inverse else witness['encoder']
    if len(values) != width or any(type(v) is not int or v not in (0, 1) for v in values):
        raise StrataError('compression input must be a complete bit vector')
    value = table[sum(bit << i for i, bit in enumerate(values))]
    return [(value >> i) & 1 for i in range(width)]


def discover_compressions(circuit, *, from_at=0, at=None, max_cuts=32, max_support=3,
                          max_queries=128, max_width=8, **budgets):
    """Automatically choose bounded cuts and dependent current-value supports.

    Every emitted witness has its own exact joint support proof and replay.
    Exceeding the cut budget is explicitly partial search, never a no-go claim.
    """
    if type(max_cuts) is not int or max_cuts < 1:
        raise StrataError('max_cuts must be positive')
    try:
        lines, at = window_lines(circuit, from_at, at, max_lines=budgets.get('max_operations', 10000))
    except WindowLimit as exc:
        return dict(status='unknown', reason=str(exc), from_at=from_at, at=at)
    import shlex
    cuts = [from_at + i + 1 for i, line in enumerate(lines)
            if (words := shlex.split(line, comments=True)) and
            words[0] not in {'REGISTER', 'APPEND_TO_REGISTER', 'INFO_TAG', 'DEBUG_PRINT'}]
    if len(cuts) > max_cuts:
        chosen = [cuts[(i * (len(cuts) - 1)) // max(1, max_cuts - 1)] for i in range(max_cuts)]
    else:
        chosen = cuts
    candidates, failures = [], []
    for cut in chosen:
        dependencies = discover_dependencies(circuit, cut, from_at=from_at,
                                            max_support=max_support, max_queries=max_queries, **budgets)
        supports = {tuple(sorted({row['wire'], *row['searched_family']['support']}))
                    for row in dependencies.get('relations', [])}
        active = touched_wires(lines[:cut-from_at])
        if len(active) <= max_width:
            supports.add(tuple(active))
        for support in sorted(supports):
            result = constructive_compression(circuit, cut, from_at=from_at, wires=support,
                                               max_width=max_width, **budgets)
            if result['status'] == 'constructed':
                candidates.append(result)
            elif result['status'] == 'unknown':
                failures.append(dict(at=cut, wires=list(support), reason=result.get('reason')))
    candidates.sort(key=lambda row: (-row['resource_bill']['leased_zero_wires'],
                                    row['resource_bill']['encoder_instructions']
                                    if row['resource_bill']['encoder_instructions'] is not None else float('inf'),
                                    row['witness']['at'], row['witness']['wires']))
    return dict(status='partial' if len(chosen) < len(cuts) else 'complete_in_family',
                cuts_searched=chosen, total_cuts=len(cuts), candidates=candidates,
                unknown=failures, max_support=max_support,
                search_family='bounded cuts, automatic polynomial dependencies, and exact bounded joint support',
                physical_layout_changed=False)


def _native_permutation(table, wires):
    """Synthesize bounded permutations as gray-path adjacent transpositions.

    For <=3 wires every adjacent swap is X, CX, or CCX, with X conjugation
    for zero controls. Wider tables remain executable permutation witnesses,
    but deliberately make no unsupported native gate-cost claim.
    """
    if len(wires) > 3:
        return None
    lines = []
    def adjacent(a, b):
        differing = (a ^ b).bit_length() - 1
        controls = [i for i in range(len(wires)) if i != differing]
        flips = [i for i in controls if not (a >> i) & 1]
        lines.extend(f'X q{wires[i]}' for i in flips)
        op = ('X', 'CX', 'CCX')[len(controls)]
        lines.append(' '.join([op, *[f'q{wires[i]}' for i in controls], f'q{wires[differing]}']))
        lines.extend(f'X q{wires[i]}' for i in reversed(flips))
    def swap(a, b):
        path = [a]
        for i in range(len(wires)):
            if (a ^ b) >> i & 1:
                path.append(path[-1] ^ (1 << i))
        for x, y in zip(path, path[1:]):
            adjacent(x, y)
        for i in range(len(path) - 3, -1, -1):
            adjacent(path[i], path[i + 1])
    visited = set()
    for root in range(len(table)):
        if root in visited:
            continue
        cycle, current = [], root
        while current not in visited:
            visited.add(current)
            cycle.append(current)
            current = table[current]
        for other in cycle[1:]:
            swap(cycle[0], other)
    return lines


def constructive_compression(circuit, at=None, *, from_at=0, wires=None,
                             clean_inputs=(), domain=None, max_inputs=16, max_width=8, **budgets):
    """Build and replay a reversible encoder from the exact *joint* support.

    No independence assumption is made about a tag bit or a separate register.
    The whole selected support is enumerated before assigning compact codes.
    The full permutation also specifies behavior outside the promised support.
    Physical width is unchanged: a zero after encoding is a scoped lease, not
    permission to delete an arbitrary external input or skip decoder costs.
    """
    try:
        lines, at = window_lines(circuit, from_at, at, max_lines=budgets.get('max_operations', 10000))
    except WindowLimit as exc:
        return dict(status='unknown', reason=str(exc), from_at=from_at, at=at)
    selected = _wires(circuit, touched_wires(lines) if wires is None else wires)
    if type(max_width) is not int or max_width < 0:
        raise StrataError('max_width must be a nonnegative integer')
    if len(selected) > max_width:
        return dict(status='unknown', reason='permutation width budget', width=len(selected), max_width=max_width)
    values = functions_at(circuit, at, from_at=from_at, clean_inputs=clean_inputs, domain=domain, **budgets)
    if values['status'] != 'exact':
        return values
    try:
        support = _patterns(values, selected, max_inputs)
    except StrataError as exc:
        return dict(status='unknown', reason=str(exc))
    bits = (len(support) - 1).bit_length()
    if bits >= len(selected):
        return dict(status='no_compression', distinct_states=len(support), information_bits=bits,
                    wires=selected, scope=values['scope'])
    encoded_support = sorted(sum(value << i for i, value in enumerate(row)) for row in support)
    low_codes = [value & ((1 << bits) - 1) for value in encoded_support]
    codes = low_codes if len(set(low_codes)) == len(low_codes) else range(len(encoded_support))
    assigned = dict(zip(encoded_support, codes))
    available = set(range(1 << len(selected))) - set(assigned.values())
    # Preserve unused states wherever possible; e.g. an AND word clears with
    # one CCX instead of an arbitrary completion requiring many transpositions.
    for value in range(1 << len(selected)):
        if value not in assigned and value in available:
            assigned[value] = value
            available.remove(value)
    remaining_source = [x for x in range(1 << len(selected)) if x not in assigned]
    remaining_target = sorted(available)
    assigned.update(zip(remaining_source, remaining_target))
    encoder = [assigned[x] for x in range(1 << len(selected))]
    decoder = [0] * len(encoder)
    for source, target in enumerate(encoder):
        decoder[target] = source
    native = _native_permutation(encoder, selected)
    witness = dict(version=1, kind='exact_joint_support_reversible_permutation', wires=selected,
                   information_bits=bits, distinct_states=len(support), support=encoded_support,
                   encoder=encoder, decoder=decoder, encoded_zero_wires=selected[bits:],
                   scope=values['scope'], domain_hash=values.get('domain_hash'),
                   window_hash=values['window_hash'],
                   from_at=from_at, at=at, clean_inputs=values['clean_inputs'],
                   native_encoder=None if native is None else '\n'.join(native),
                   native_decoder=None if native is None else '\n'.join(reversed(native)))
    for value in range(len(encoder)):
        row = [(value >> i) & 1 for i in range(len(selected))]
        encoded = execute_compression(witness, row)
        if execute_compression(witness, encoded, inverse=True) != row:
            raise StrataError('compression decoder replay failed')
        if value in assigned and value in encoded_support and any(encoded[bits:]):
            raise StrataError('compression clean lease replay failed')
    if native is not None:
        for value in range(len(encoder)):
            assignment = {f'q{q}': 0 for q in range(circuit.num_qubits)}
            assignment.update({f'q{q}': (value >> i) & 1 for i, q in enumerate(selected)})
            replay = execute_assignment(native, assignment, circuit.num_qubits)
            actual = sum(replay['outputs'][str(q)] << i for i, q in enumerate(selected))
            if actual != encoder[value] or replay['phase']:
                raise StrataError('native compression encoder replay failed')
    counts = None if native is None else {
        op: sum(line.split()[0] == op for line in native) for op in ('X', 'CX', 'CCX')}
    bill = dict(physical_width_before=circuit.num_qubits, physical_width_after=circuit.num_qubits,
                leased_zero_wires=len(selected) - bits, encoder_table_entries=len(encoder),
                decoder_table_entries=len(decoder), native_encoder_counts=counts,
                native_decoder_counts=counts, encoder_instructions=None if native is None else len(native),
                decoder_instructions=None if native is None else len(native),
                native_cost_status='exact_emitted' if native is not None else 'not_synthesized',
                width_reduction=0)
    return dict(status='constructed', witness=witness, witness_hash=digest(witness), resource_bill=bill,
                replay=dict(status='verified', assignments=len(encoder), inverse=True,
                            scoped_zero_bits=True, native_encoder=native is not None),
                note='Encoder and decoder must both be priced. Physical wire layout and external inputs are unchanged.')


def eliminate_clean_phase_workspace(circuit, *, clean_inputs=(), drop_clean=False, max_check_inputs=16,
                                    max_variables=128, max_terms=65536, max_operations=10000,
                                    max_product_terms=1048576):
    """Emit an exact native phase circuit when clean materialized words restore.

    This general path recognizes arbitrary ANF compute/consume/uncompute phase
    functions up to degree three, without register names or producer templates.
    Every physical output is proved unchanged under the explicit clean promise.
    """
    clean = set(_wires(circuit, clean_inputs, 'clean inputs'))
    if not clean:
        return dict(status='unknown', reason='explicit clean boundary inputs required')
    state = functions_at(circuit, clean_inputs=clean, max_variables=max_variables, max_terms=max_terms,
                         max_operations=max_operations, max_product_terms=max_product_terms)
    if state['status'] != 'exact':
        return state
    if not state.get('is_permutation', True) or state.get('classical_inputs') or state.get('measurement_variables'):
        return dict(status='unknown', reason='native phase reconstruction requires a reversible quantum-only window')
    for q, poly in state['functions'].items():
        if poly != (set() if q in clean else {frozenset({q})}):
            return dict(status='unknown', reason='nonidentity output outside clean phase reconstruction family')
    phase = state['phase']
    if any(len(term) > 3 or any(type(q) is not int for q in term) for term in phase):
        return dict(status='unknown', reason='phase degree exceeds native synthesis family')
    emitted = []
    for term in sorted(phase, key=_term_key):
        emitted.append(' '.join([('NEG', 'Z', 'CZ', 'CCZ')[len(term)], *[f'q{q}' for q in sorted(term)]]))
    import kickmix as km
    # Public input wire indices remain stable. Only declared-zero wires can be
    # omitted from the physical layout; interior holes alone do not save Q.
    import shlex
    public = sorted(set(range(circuit.num_qubits)) - clean)
    retained = public if drop_clean else list(range(circuit.num_qubits))
    metadata = []
    used_registers = set()
    for line in str(circuit).splitlines():
        words = shlex.split(line, comments=True)
        if words and words[0] in {'REGISTER', 'APPEND_TO_REGISTER', 'INFO_TAG'}:
            used_registers.update(int(w[1:]) for w in words if w.startswith('r') and w[1:].isdigit())
            if words[0] == 'APPEND_TO_REGISTER' and drop_clean and any(w == f'q{q}' for q in clean for w in words):
                continue
            metadata.append(line)
    boundary_id = max(used_registers, default=-1) + 1
    if retained:
        metadata += [f'APPEND_TO_REGISTER q{max(retained)} r{boundary_id}', f'REGISTER r{boundary_id} "strata_boundary"']
    emitted = metadata + emitted
    candidate = km.Circuit('\n'.join(emitted))
    verification = functions_at(candidate, clean_inputs=clean & set(range(candidate.num_qubits)), max_variables=max_variables, max_terms=max_terms,
                                max_operations=max_operations, max_product_terms=max_product_terms)
    expected = {q: value for q, value in state['functions'].items() if q < candidate.num_qubits}
    if verification['status'] != 'exact' or verification['functions'] != expected or verification['phase'] != phase:
        raise StrataError('native clean phase reconstruction replay failed')
    if type(max_check_inputs) is not int or max_check_inputs < 0:
        raise StrataError('max_check_inputs must be a nonnegative integer')
    independent_inputs = sorted(set(state['touched_wires']) - clean)
    independent = dict(status='unknown', reason='independent enumeration budget', input_count=len(independent_inputs))
    if len(independent_inputs) <= max_check_inputs:
        before_lines, after_lines = str(circuit).splitlines(), str(candidate).splitlines()
        for encoded in range(1 << len(independent_inputs)):
            assignment = {f'q{q}': 0 for q in range(circuit.num_qubits)}
            assignment.update({f'q{q}': (encoded >> i) & 1 for i, q in enumerate(independent_inputs)})
            a = execute_assignment(before_lines, assignment, circuit.num_qubits, max_operations=max_operations)
            b = execute_assignment(after_lines, assignment, circuit.num_qubits, max_operations=max_operations)
            if any(a[k] != b[k] for k in ('outputs', 'phase', 'classical_outputs')):
                raise StrataError('independent native phase reconstruction replay failed')
        independent = dict(status='verified', assignments=1 << len(independent_inputs),
                           method='independent_concrete_basis_sign_interpreter',
                           untouched_inputs='arbitrary and unchanged; not enumerated',
                           quantum_scope='exact basis outputs and relative sign with declared zero workspace')
    from .kickmix import Kickmix
    before, after = Kickmix().measured(circuit), Kickmix().measured(candidate)
    return dict(status='constructed', candidate=str(candidate), clean_inputs=sorted(clean),
                witness_hash=digest(dict(source=str(circuit), candidate=str(candidate), clean_inputs=sorted(clean))),
                proof=dict(status='equivalent', scope=state['scope'], outputs='all physical outputs and exact sign',
                           all_input_equivalence=False, method='exact_clean_boundary_ANF_replay'),
                independent_replay=independent,
                layout=dict(public_inputs=public, wire_map={str(q):q for q in retained},
                            omitted_clean_wires=sorted(clean) if drop_clean else [],
                            restored_outputs={str(q):0 for q in clean} if drop_clean else {},
                            external_input_indices_unchanged=True),
                resource_bill=dict(before=before, after=after, native_cost_status='exact_emitted',
                                   width_reduction=before['Q']-after['Q']),
                removed_materialized_wires=sorted(clean & set(state['touched_wires'])),
                note='Only explicitly clean workspace may leave the layout; every external input keeps its physical index.')
