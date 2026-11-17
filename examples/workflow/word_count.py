"""A non-circuit workload writing ordinary JSON Lines."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

p = argparse.ArgumentParser()
p.add_argument('--input', required=True)
p.add_argument('--min-length', type=int, required=True)
p.add_argument('--seed', type=int, required=True)
args = p.parse_args()
counts = Counter(w for w in re.findall(r'[a-z]+', Path(args.input).read_text().lower())
                 if len(w) >= args.min_length)
with open('counts.jsonl', 'w') as stream:
    for word, count in sorted(counts.items()):
        stream.write(json.dumps(dict(word=word, count=count)) + '\n')
Path('summary.json').write_text(json.dumps(dict(tokens=sum(counts.values()), vocabulary=len(counts))))
print('counted', sum(counts.values()), 'tokens')
