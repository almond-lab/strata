"""Probability policy is enforced through real Line gates and certificate replay."""
import json
import pytest

from strata.bundle import Bundle, BundleFrontend, Component
from strata.callbacks import CallbackAdapter, PinnedCallback, BuildReceipt, CheckReceipt
from strata.model import StrataError
from strata.project import Project, Design


FRONTEND = BundleFrontend([Component('numbers', 'numbers.json')])


def build(request):
    artifact = Bundle({'numbers.json': json.dumps({'rank': request.implementation.get('rank', 4)}).encode()})
    return BuildReceipt(artifact, FRONTEND.tree(artifact, 'probability-fixture:v1'))


def check(request):
    rank = json.loads(request.artifact.files['numbers.json'])['rank']
    draws = [dict(seed=s, resources=dict(Q=0, component_widths={}, rank=rank*k, fail_rate=p))
             for s, k, p in [(11, 1, 0), (22, 2, 0.2)]]
    return CheckReceipt.for_request(request, passed=rank>0,
        resources=dict(Q=0, component_widths={}, rank=rank*1.5, fail_rate=0.1),
        details=dict(seeds=[11, 22], draws=draws, selection={'mode': 'precommitted'}))


def lucky_check(request):
    receipt = check(request)
    from dataclasses import replace
    return replace(receipt, details={**receipt.details, 'selected_seed': 11})


def make_project(tmp_path, checker=check):
    base = tmp_path/'base';base.mkdir()
    (base/'numbers.json').write_text('{"rank": 4}')
    path = tmp_path/'strata.toml'
    path.write_text('''[project]
store = ".strata"
[score]
formula = "geomean(task.base_cost / task.cost)"
[[task]]
id = "numeric"
adapter = "external"
base = "base"
cost = "R(fail_rate) * rank"
[task.domain]
seeds = [11, 22]
[task.probability]
seeds = [11, 22]
compare = "expected"
max_failure = 0.11
penalty = "1/(1-fail_rate)"
''')
    adapter = CallbackAdapter(frontend=FRONTEND, build=PinnedCallback(build),
        lab=PinnedCallback(checker), measure=PinnedCallback(checker),
        oracle=PinnedCallback(checker), verify=PinnedCallback(checker))
    return Project(path, {'external': adapter})


def test_nonlinear_draw_pricing_survives_line_and_replay(tmp_path):
    project = make_project(tmp_path)
    parent = Design(project, project.init()['numeric'])
    # E[R(p)*rank] = (4 + 8/0.8)/2, not R(E[p])*E[rank].
    assert parent.record['cost'] == 7
    line = project.open_line(parent, 'co-design-numerics', ['numbers'], implementation={'rank': 2})
    assert line.lab().passed
    assert line.build()['measured_cost'] == 3.5
    design = line.verify()
    assert design.record['cost'] == 3.5 and project.replay(design.id)['passed']
    certificate = project.store.get(design.record['certificate'])
    assert certificate['full']['probability']['per_draw_cost'] == [2, 5]
    certificate['full']['probability']['cost'] = 2
    forged = project.store.put('certificate', certificate)
    forged_design = project.store.put('design', {**design.record, 'certificate': forged})
    with pytest.raises(StrataError):Design(project, forged_design).validate()
    project.path.write_text(project.path.read_text().replace('max_failure = 0.11', 'max_failure = 0.2'))
    changed = Project(project.path, project.adapters)
    with pytest.raises(StrataError):changed.replay(design.id)


def test_candidate_selected_draw_cannot_be_ingested(tmp_path):
    project = make_project(tmp_path, lucky_check)
    with pytest.raises(StrataError, match='favourable draw'):project.init()
    assert not project.store.all('design')
