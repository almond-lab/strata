"""Reproducible synthetic repeated-kernel census; no host-performance claim."""
import argparse
import json
from pathlib import Path
import tempfile
import time
import kickmix as km
from strata.streaming import scan_circuit

parser=argparse.ArgumentParser()
parser.add_argument('--operations',type=int,default=500000)
parser.add_argument('--out',required=True)
args=parser.parse_args()
with tempfile.TemporaryDirectory() as temporary:
    root=Path(temporary);source=root/'source.kmx'
    block='CCX q0 q1 q2\nCX q2 q3\nCCX q0 q1 q2\nINFO_TAG "kernel"\n'
    with source.open('w') as f:
        for _ in range(args.operations//4):f.write(block)
    begun=time.monotonic();c=km.Circuit.from_file(source)
    parse_seconds=time.monotonic()-begun
    binary=root/'source.kmb';c.to_file(binary,format='kmb')
    report=scan_circuit(c,root/'scan',source=binary)
    report.update(benchmark='synthetic repeated compute-consume-return kernels',native_parse_seconds=parse_seconds,
                  requested_text_lines=args.operations,binary_bytes=binary.stat().st_size,
                  host_verifier_run=False)
    # Temporary index paths are intentionally not published as durable artifacts.
    report.pop('index',None)
    for group in report['groups']:group.pop('occurrence_index',None)
    Path(args.out).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('scanned_operations','seconds','seconds_per_million_operations','peak_rss_bytes','group_count','occurrences')}))
