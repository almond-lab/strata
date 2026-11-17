#!/usr/bin/env python3
"""Secondary objective check for the selected inverse endpoint in the frozen archive."""
import argparse
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import tempfile
import time
import zipfile

import kickmix as km
from strata.cli import main
from native_spans import NativeSpans
from run import ARCHIVE_SHA, HERE, OPTIONS, build_id, file_sha, make_source, sha
from replay import independent_check


def run(archive, scratch, output):
    reference = json.loads((HERE/'release-promises.json').read_text())
    if file_sha(archive) != ARCHIVE_SHA or build_id() != reference['package_python_sha256']:
        raise ValueError('Use the archive and installed release sources from the primary replay')
    case = next(r for r in reference['results'] if r['case']=='inverse-fractional-endpoint')
    epsilon = 1.2739997691835382e-13
    rotation_t = 0.56 * math.log2(1/epsilon) + 5.3
    rows = []
    with tempfile.TemporaryDirectory(dir=scratch) as root, zipfile.ZipFile(archive) as z:
        root = Path(root)
        with tempfile.TemporaryFile(dir=root) as binary, z.open(case['member']) as src:
            shutil.copyfileobj(src, binary)
            binary.seek(0)
            if hashlib.file_digest(binary, 'sha256').hexdigest() != reference['parent_members'][case['member']]['sha256']:
                raise ValueError('Parent binary mismatch')
            tape = NativeSpans(binary)
            try:
                episode = make_source(tape, case['member'], case['start'], case['stop'], root, True)
                assert episode['source_native_sha256']==case['source_native_sha256'] and tape.width==44
            finally:
                tape.close()
        before = km.Circuit.from_file(root/'source.kmx')
        for label, formula in [('fixed_native_t', 'Q*(toffoli+t/4)'),
                               ('rotation_sensitivity', f'Q*(toffoli+(t+{rotation_t!r}*rotations)/4)')]:
            out = root/label
            options = list(OPTIONS)
            options[options.index('--cost')+1] = formula
            started = time.monotonic()
            with contextlib.redirect_stdout(io.StringIO()):
                status = main(['search', str(root/'source.kmx'), '--out', str(out),
                               '--width', '44', '--clean-inputs', ','.join(map(str,episode['clean_local'])), *options])
            if status:
                raise ValueError('Search failed')
            receipt = json.loads((out/'restructure.json').read_text())
            def stats(state):
                return {'cost':state['cost'], **{k:state['resources'][k] for k in
                        ('Q','toffoli','t','rotations','instructions')}}
            candidate = km.Circuit.from_file(receipt['best']['path'])
            bare = km.Circuit('\n'.join(line for line in str(candidate).splitlines()
                               if line.split()[0] not in ('REGISTER','APPEND_TO_REGISTER')))
            check = independent_check(before, bare, episode['clean_local'], receipt['best']['private_bits'], None, 65536)
            trials = [dict(status=a['status'], delta=a.get('cost_delta'), reason=a.get('construction',{}).get('reason'))
                      for a in receipt['attempts'] if a['mechanism']=='fractional_observable_search']
            rows.append(dict(label=label, formula=formula, status=receipt['status'], before=stats(receipt['baseline']),
                             after=stats(receipt['best']), check=check, fractional_trials=trials,
                             seconds=round(time.monotonic()-started,3), timed_out=receipt['search_control']['timed_out']))
    result = dict(scope='secondary targeted objective recheck after primary sweep', case=case['case'],
                  archive_sha256=ARCHIVE_SHA, parent_member_sha256=reference['parent_members'][case['member']]['sha256'],
                  start=case['start'], stop=case['stop'], source_native_sha256=case['source_native_sha256'],
                  wheel_sha256=reference['installed_wheel_sha256'], package_python_sha256=build_id(),
                  script_sha256=file_sha(__file__), primary_receipt_sha256=file_sha(HERE/'release-promises.json'),
                  parent_synthesis_epsilon=epsilon, fixed_t_per_fine_rotation=rotation_t,
                  rate_provenance='archived default-salt c1 inverse receipt, matching member hash; '
                                  '0.56 log2(1/epsilon) + 5.3. Held fixed for sensitivity.',
                  official_verifier_run=False, runs=rows)
    with output.open('x') as file:
        json.dump(result, file, indent=2)
        file.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--scratch', required=True, type=Path)
    args = parser.parse_args()
    if args.scratch.resolve().is_relative_to(HERE.parent.parent) or args.out.exists():
        parser.error('choose private scratch outside the checkout and a fresh output path')
    run(args.archive, args.scratch, args.out)
