#!/usr/bin/env python3
"""Freeze untagged generated sources without importing any Strata constructor."""
import hashlib
import json
from pathlib import Path
import random
import re

HERE = Path(__file__).resolve().parent
SEED = 914027


def sha(data):
    return hashlib.sha256(data).hexdigest()


def field_mul(a, b, out):
    # GF(4), polynomial basis, z^2 = z + 1. No target is a control.
    return [f'CCX q{a[i]} q{b[j]} q{out[k]}'
            for i, j, k in [(0, 0, 0), (1, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]]


def evaluator_delta(delta):
    # Materialize P(x) and P(x+d), P(t)=a*t^3 + b*t over GF(4).
    x, a, b, y = [0, 1], [2, 3], [4, 5], [6, 7]
    program, clean = [], []
    for evaluation in range(2):
        square = [8 + 8 * evaluation, 9 + 8 * evaluation]
        cube = [10 + 8 * evaluation, 11 + 8 * evaluation]
        high = [12 + 8 * evaluation, 13 + 8 * evaluation]
        low = [14 + 8 * evaluation, 15 + 8 * evaluation]
        clean += square + cube + high + low
        offset = [f'X q{x[i]}' for i in range(2) if evaluation and delta >> i & 1]
        square_gates = [f'CX q{x[0]} q{square[0]}', f'CX q{x[1]} q{square[0]}',
                        f'CX q{x[1]} q{square[1]}']
        compute = square_gates + field_mul(square, x, cube) + field_mul(cube, a, high) + field_mul(x, b, low)
        consume = [f'CX q{q} q{y[i]}' for word in [high, low] for i, q in enumerate(word)]
        program += offset + compute + consume + compute[::-1] + offset[::-1]
    return program, clean, y


def table_bank(table, width):
    # Store every minterm before observing the output word; all input NOTs
    # are restored, and each materialized minterm has its own clean chain.
    n = (len(table) - 1).bit_length()
    targets = list(range(n, n + width))
    next_wire, clean, bank, terminals = n + width, [], [], []
    for word in range(1 << n):
        scratch = list(range(next_wire, next_wire + n - 1))
        next_wire += n - 1
        clean += scratch
        offset = [f'X q{i}' for i in range(n) if not (word >> i & 1)]
        compute = [f'CCX q0 q1 q{scratch[0]}']
        compute += [f'CCX q{scratch[i - 2]} q{i} q{scratch[i - 1]}' for i in range(2, n)]
        bank += offset + compute + offset[::-1]
        terminals.append(scratch[-1])
    observe = [f'CX q{terminals[word]} q{targets[bit]}'
               for word, value in enumerate(table) for bit in range(width) if value >> bit & 1]
    return bank + observe + bank[::-1], clean, targets, terminals


def relayout(lines, clean, targets, seed, holes=0):
    rng = random.Random(seed)
    width = 1 + max(int(q) for line in lines for q in re.findall(r'\bq(\d+)\b', line))
    # Keep the clean bank high enough to permit physical retirement. Random
    # permutations within data and scratch classes are independent of search.
    public = [q for q in range(width) if q not in clean]
    public_places = list(range(len(public) + holes))
    if holes:
        unused = {1, len(public_places) - 2}
        public_places = [q for q in public_places if q not in unused]
    rng.shuffle(public_places)
    clean_places = list(range(len(public) + holes, width + holes))
    rng.shuffle(clean_places)
    mapping = dict(zip(public, public_places)) | dict(zip(clean, clean_places))
    renamed = [re.sub(r'\bq(\d+)\b', lambda m: f'q{mapping[int(m[1])]}', line) for line in lines]
    return renamed, sorted(mapping[q] for q in clean), [mapping[q] for q in targets], mapping


def fixtures():
    rows = []

    def add(name, family, lines, clean, targets, seed, holes=0, role='opportunity'):
        lines, clean, targets, mapping = relayout(lines, clean, targets, seed, holes)
        text = '\n'.join(lines) + '\n'
        row = dict(name=name, family=family, role=role, seed=seed, text=text,
                   sha256=sha(text.encode()), clean_inputs=clean, output_wires=targets,
                   internal_bits=[], layout=dict(untouched_arbitrary_holes=holes))
        rows.append(row)
        return row, mapping

    field_rows = []
    for i, delta in enumerate([1, 3]):
        lines, clean, targets = evaluator_delta(delta)
        row, _ = add(f'field_difference_{i}', 'gf4_polynomial_difference', lines, clean, targets,
                     SEED + i, holes=2 * i)
        field_rows.append(row)

    table_rows = []
    for i, n in enumerate([3, 4]):
        rng = random.Random(SEED + 20 + i)
        modulus = 1 << n
        a, c, delta = rng.randrange(1, modulus, 2), rng.randrange(modulus), rng.randrange(1, modulus, 2)
        values = [(x * x * x + a * x + c) % modulus for x in range(modulus)]
        table = [values[(x + delta) % modulus] ^ values[x] for x in range(modulus)]
        lines, clean, targets, _ = table_bank(table, n)
        row, _ = add(f'modular_difference_{n}', 'modular_word_difference_table', lines, clean, targets,
                     SEED + 20 + i)
        table_rows.append(row)

    for i in range(2):
        rng = random.Random(SEED + 40 + i)
        values = [rng.randrange(8) for _ in range(8)]
        mask = rng.randrange(1, 8)
        # Two complete materialized evaluations feed one dirty result word.
        first, clean, targets, _ = table_bank(values, 3)
        second, clean2, _, _ = table_bank([values[x ^ mask] for x in range(8)], 3)
        remap = {q: q + len(clean) for q in clean2}
        second = [re.sub(r'\bq(\d+)\b', lambda m: f'q{remap.get(int(m[1]), int(m[1]))}', line)
                  for line in second]
        clean += [remap[q] for q in clean2]
        add(f'table_difference_{i}', 'paired_boolean_table_evaluators', first + second, clean, targets,
            SEED + 40 + i, holes=2 * i)

    phase = dict(field_rows[0])
    phase.update(name='phase_obligation', role='phase_guard',
                 text=phase['text'] + f'Z q{phase["output_wires"][0]}\n')
    phase['sha256'] = sha(phase['text'].encode())
    rows.append(phase)

    dirty = dict(table_rows[0])
    dirty.update(name='dirty_workspace_obligation', role='dirty_guard',
                 clean_inputs=dirty['clean_inputs'][1:])
    rows.append(dirty)

    classical = dict(table_rows[0])
    classical.update(name='public_classical_obligation', role='classical_guard',
                     text='BIT_STORE1 b2\n' + classical['text'])
    classical['sha256'] = sha(classical['text'].encode())
    rows.append(classical)
    assert len({r['name'] for r in rows}) == len(rows)
    assert all('INFO_TAG' not in r['text'] and not r['internal_bits'] for r in rows)
    return rows


def main():
    target = HERE / 'fixtures.json'
    if target.exists():
        raise RuntimeError('Frozen fixtures already exist; refusing to regenerate after observing search')
    rows = fixtures()
    target.write_text(json.dumps(rows, indent=2) + '\n')
    manifest = dict(generator_seed=SEED, generator_sha256=sha(Path(__file__).read_bytes()),
                    fixtures_sha256=sha(target.read_bytes()), fixture_count=len(rows),
                    frozen_before_first_search=True, no_replacement_or_region_supplied=True,
                    provenance='Independent review-agent generators; frozen after existing development suites were read, before any holdout search. This is not a blind historical benchmark.',
                    objective='Q*toffoli', budgets=dict(rounds=1, beam=4, max_candidates=12,
                        max_operations=512, max_variables=128, max_terms=65536, max_product_terms=65536),
                    fixture_timeout_seconds=45, native_case_cap=1048576,
                    source_hashes={row['name']: row['sha256'] for row in rows})
    (HERE / 'freeze.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
