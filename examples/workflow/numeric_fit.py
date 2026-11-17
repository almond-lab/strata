"""An ordinary stdlib fit script: no Strata imports or adapter protocol."""
import argparse
import csv
import json
from pathlib import Path
import random

p = argparse.ArgumentParser()
p.add_argument('--data', required=True)
p.add_argument('--steps', type=int, required=True)
p.add_argument('--seed', type=int, required=True)
p.add_argument('--checkpoint')
args = p.parse_args()
rows = [(float(r['x']), float(r['y'])) for r in csv.DictReader(open(args.data))]
slope = random.Random(args.seed).uniform(-1, 1)
intercept = 0.0
if args.checkpoint:
    previous = json.loads(Path(args.checkpoint).read_text())
    slope, intercept = previous['slope'], previous['intercept']
with open('curve.csv', 'w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=['step', 'mse'])
    writer.writeheader()
    for step in range(args.steps):
        errors = [slope * x + intercept - y for x, y in rows]
        slope -= 0.1 * sum(e * x for e, (x, _) in zip(errors, rows)) / len(rows)
        intercept -= 0.1 * sum(errors) / len(rows)
        mse = sum((slope * x + intercept - y) ** 2 for x, y in rows) / len(rows)
        writer.writerow(dict(step=step + 1, mse=mse))
result = dict(slope=slope, intercept=intercept, mse=mse, points=len(rows), steps=args.steps,
              predictions=[slope * x + intercept for x, _ in rows])
Path('fit.json').write_text(json.dumps(result))
# Atomic checkpoint replacement is the script's own feature.
Path('checkpoint.partial').write_text(json.dumps(dict(slope=slope, intercept=intercept)))
Path('checkpoint.partial').replace('checkpoint.json')
print('fit completed', args.steps, 'steps')
