"""Independent reference checks, pinned in each manifest at experiment creation."""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import re

p = argparse.ArgumentParser()
p.add_argument('kind', choices=['fit', 'circuit', 'words'])
p.add_argument('artifact')
p.add_argument('--input')
p.add_argument('--min-length', type=int, default=1)
p.add_argument('--width', type=int)
args = p.parse_args()
details = {}
resources = {}
if args.kind == 'fit':
    result = json.loads(Path(args.artifact).read_text())
    rows = list(csv.DictReader(open(args.input)))
    predictions = [result['slope'] * float(row['x']) + result['intercept'] for row in rows]
    mse = math.fsum((guess - float(row['y'])) ** 2 for guess, row in zip(predictions, rows)) / len(rows)
    accepted = len(result['predictions']) == len(rows) and all(math.isclose(a, b) for a, b in zip(result['predictions'], predictions)) and math.isclose(mse, result['mse'], abs_tol=1e-12) and mse < 0.1
    details = dict(independent_mse=mse, acceptance_threshold=0.1, unit='squared_response')
    resources = dict(points=len(rows), mse=mse)
elif args.kind == 'circuit':
    n = args.width
    gates = []
    accepted = type(n) is int and 1 <= n <= 8
    for line in Path(args.artifact).read_text().splitlines():
        tokens = line.split()
        if len(tokens) != 3 or tokens[0] != 'CX' or any(not re.fullmatch('q[0-9]+', q) for q in tokens[1:]):
            accepted = False
            break
        gates.append(['cx', int(tokens[1][1:]), int(tokens[2][1:])])
    checked = 0
    if accepted:
        for bits in itertools.product((0, 1), repeat=n + 1):
            state = list(bits)
            for op, control, target in gates:
                if op != 'cx' or type(control) is not int or type(target) is not int or not (0 <= control <= n and 0 <= target <= n) or control == target:
                    accepted = False
                    break
                state[target] ^= state[control]
            accepted &= state[:n] == list(bits[:n]) and state[n] == (sum(bits[:n]) % 2) ^ bits[n]
            checked += 1
    details = dict(basis_states_checked=checked, declared_input_width=n, scope='all basis states for the frozen declared circuit width')
    resources = dict(gates=len(gates), wires=n + 1)
else:
    rows = [json.loads(line) for line in Path(args.artifact).read_text().splitlines()]
    words = [w for w in re.split('[^a-z]+', Path(args.input).read_text().lower()) if len(w) >= args.min_length and w]
    reference = {w: words.count(w) for w in set(words)}
    accepted = len(rows) == len(reference) and {r['word']: r['count'] for r in rows} == reference
    details = dict(tokens=len(words), vocabulary=len(reference))
    resources = dict(tokens=len(words), vocabulary=len(reference))
print(json.dumps(dict(accepted=bool(accepted), details=details, resources=resources)))
