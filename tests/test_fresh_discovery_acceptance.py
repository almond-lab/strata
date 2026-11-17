"""Fresh, untagged file: automatic proposal, exact lab, price, saved report.

This is a public-safe eight-bit fixture, not a historical competition win.
No task adapter, move list, implementation flag or generator tag is supplied.
"""
from dataclasses import asdict
import json
import kickmix as km
from strata.file_workflow import optimize_file
from strata.lab import exhaustive
from strata.model import Contract, Register
from strata.reports import report, markdown_report
from strata.store import Store


def test_fresh_file_discovers_a_measured_lab_checked_move(tmp_path):
    source = tmp_path/'fresh.kmx'
    source.write_text('CCX q0 q2 q6\nCX q6 q4\nCCX q0 q2 q6\n'
                      'CCX q1 q3 q7\nCX q7 q5\nCCX q1 q3 q7')
    out = tmp_path/'run'
    result = optimize_file(source, out)
    assert result['status'] == 'improved' and result['replay']
    assert result['baseline']['cost'] == 32 and result['optimized']['cost'] == 16
    # Independently specify the external word function, including dirty temps.
    def reference(values, _):
        x = values['state']
        for a, b, y, t in [(0,2,4,6),(1,3,5,7)]:
            x ^= ((((values['state']>>a)&1) & ((values['state']>>b)&1))
                  ^ ((values['state']>>t)&1)) << y
        return {'state': x}, 0
    candidate = km.Circuit.from_file(result['candidate'])
    contract = Contract('external-eight-bit-function', (Register('state', tuple(range(8))),), 'fixture:two-products:v1')
    lab = exhaustive(candidate, contract, reference)
    assert lab.passed and lab.exhaustive and lab.cases == 256
    store = Store(out/'.strata')
    store.put('file_acceptance_lab', dict(optimization=result['record'], receipt=asdict(lab)))
    source.unlink()
    saved = report(store, handle=result['record'])
    attempt = saved['file_optimizations'][0]
    assert attempt['status'] == 'improved' and attempt['replay']['equivalent']
    assert attempt['regions'] and all(row['explanation'] for row in attempt['regions'])
    rendered = markdown_report(saved)
    assert '32' in rendered and '16' in rendered
    assert 'file' in rendered.lower()
    assert json.loads(json.dumps(saved)) == saved
