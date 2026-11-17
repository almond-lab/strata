"""An unchanged-style circuit generator with its existing JSON output format."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--width', type=int, required=True)
p.add_argument('--seed', type=int, required=True)
args = p.parse_args()
if not 1 <= args.width <= 8:
    raise SystemExit('reference fixture supports one through eight input wires')
gates = [['cx', source, args.width] for source in range(args.width)]
Path('circuit.json').write_text(json.dumps(dict(inputs=args.width, wires=args.width + 1, gates=gates)))
Path('circuit.kmx').write_text(''.join(f'CX q{source} q{args.width}\n' for source in range(args.width)))
Path('resources.json').write_text(json.dumps(dict(gates=len(gates), wires=args.width + 1)))
print('parity circuit generated')
