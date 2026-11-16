import json
from pathlib import Path
import kickmix as km
from strata.cli import main
from strata.kickmix import Kickmix


def test_actual_sdk_analysis_and_honest_opaque_lift(tmp_path):
    c=km.Circuit('CCX q0 q1 q2\nHMR q2 b0\nCZ q0 q1 if b0')
    pack=Kickmix().analyze(c,tmp_path)
    assert pack['metrics']['toffoli']==1
    assert pack['debts'][0]['status']=='repayment_unresolved'
    assert (tmp_path/'flame.svg').exists() and (tmp_path/'diagram.html').exists()
    tree,coverage=Kickmix().lift(c)
    assert coverage['structural_typed_coverage']==0
    assert not coverage['confirmed']
    assert any(r.kind=='opaque' for r in tree.children)


def test_cli_end_to_end(tmp_path,capsys):
    src=Path(__file__).parents[1]/'examples/and_fusion'
    for name in ['base.kmx','strata.toml']:(tmp_path/name).write_bytes((src/name).read_bytes())
    args=['--project',str(tmp_path/'strata.toml')]
    assert main(args+['init'])==0
    base=json.loads(capsys.readouterr().out)['and']
    assert main(args+['line','open','--parent',base,'--move','fuse','--regions','lane-0,lane-1,lane-2','--implementation','{"fused":true}'])==0
    line=json.loads(capsys.readouterr().out)['line']
    assert main(args+['line','run',line])==0
    design=json.loads(capsys.readouterr().out)['design']
    assert main(args+['replay',design])==0
    assert json.loads(capsys.readouterr().out)['passed']
    assert main(args+['best'])==0
    assert json.loads(capsys.readouterr().out)['score']>1


def test_research_cli_checks_structure_and_continues_from_retained_best(tmp_path,capsys):
    src=Path(__file__).parents[1]/'examples/and_fusion'
    for name in ['base.kmx','strata.toml']:(tmp_path/name).write_bytes((src/name).read_bytes())
    args=['--project',str(tmp_path/'strata.toml')]
    assert main(args+['init'])==0
    base=json.loads(capsys.readouterr().out)['and']
    assert main(args+['research','and','--run'])==0
    result=json.loads(capsys.readouterr().out)
    assert result['qualified_gains']==1
    outcome=result['outcomes'][0]
    assert outcome['structure']['deep'] and outcome['replay']
    assert (outcome['cost_before'],outcome['cost_after'])==(72,27)
    assert len(outcome['labs'])==2
    assert main(args+['research','and'])==0
    assert json.loads(capsys.readouterr().out)['parent']==outcome['design']
    assert main(args+['research','and','--parent',base])==0
    assert json.loads(capsys.readouterr().out)['parent']==base
