import json
from dataclasses import asdict, replace
from pathlib import Path
import pytest
from strata import open as open_project
from strata.model import Contract, Register, Region, Floor, StrataError, region_from_dict
from strata.store import Store
from strata.cost import Formula, portfolio, rank_gaps
from strata.engine import Patch, compose
from strata.demo import AndFusion
from strata.project import Design


@pytest.fixture
def project(tmp_path):
    example=Path(__file__).parents[1]/'examples/and_fusion'
    for name in ('strata.toml','base.kmx'):
        (tmp_path/name).write_bytes((example/name).read_bytes())
    return open_project(tmp_path/'strata.toml')


def test_formula_not_python_and_exact_contest_arithmetic():
    assert Formula('R(fail_rate) * Q^2 * toffoli',{'R':lambda p:1/(1-p)})({'fail_rate':0,'Q':889,'toffoli':100})==889**2*100
    for expr in ["__import__('os').system('id')",'Q.__class__','[x for x in Q]','2**1000000']:
        with pytest.raises(StrataError): Formula(expr)({'Q':2})
    with pytest.raises(StrataError): Formula('R(fail_rate)')({'fail_rate':0})
    assert portfolio('geomean(task.base_cost / task.cost)',[(8,4),(3,6)])==1


def test_contract_semantics_survive_wire_relabel():
    a=Contract('foo',(Register('x',(0,1)),),'id')
    b=Contract('bar',(Register('y',(7,8)),),'id')
    assert a.fingerprint==b.fingerprint
    assert replace(b,reference='different').fingerprint!=a.fingerprint
    with pytest.raises(StrataError):Contract('bad',(Register('a',(0,)),Register('b',(0,))),'id')


def test_store_integrity_negative_scope_and_no_overwrite(tmp_path):
    s=Store(tmp_path)
    h=s.blob(b'abc');assert s.read_blob(h)==b'abc'
    (tmp_path/'objects'/h).write_bytes(b'wrong')
    with pytest.raises(StrataError):s.read_blob(h)
    for kind,status in [('estimate','priced'),('build','closed'),('exact_small','closed_at_size')]:
        h=s.negative('contract','fuse',kind,{'number':12},'new consumer',{'bits':4} if kind=='exact_small' else None)
        assert s.get(h)['status']==status
    assert len(s.tried('contract','fuse'))==3
    with pytest.raises(StrataError):s.negative('contract','fuse','exact_small',{'pass':False},'more bits')


def test_real_structural_pipeline_replay_and_exact_price(project):
    base=project.init()['and'];d0=Design(project,base)
    assert d0.record['cost']==72
    assert d0.gaps()[0]['weighted_gap'] is not None
    price=d0.price('root','wire',-3)
    assert price['delta']==-45 and price['kind']=='measured_build'
    line=project.open_line(d0,'replace-stages',['root'],implementation={'fused':True})
    with pytest.raises(StrataError):line.verify()
    with pytest.raises(StrataError):line.build()
    assert line.lab(bits=2).passed
    line.build();d1=line.verify()
    assert d1.record['cost']==27 and d1.record['depth']=='L4'
    assert project.best()['score']==pytest.approx(72/27)
    assert project.replay(d1.id)['passed']
    assert d1.record['parent']==base
    assert len(project.store.all('design'))==2
    assert project.store.get(d1.record['certificate'])['full']['official']['authority'].startswith('Strata demo')


def test_floors_do_not_launder_library_or_small_exact_bounds():
    _,tree=AndFusion().build({'bits':1},{})
    fs=[Floor(tree.contract.fingerprint,'toffoli',1,k,'evidence',{'bits':1} if k=='exact_small' else {}) for k in ['library','exact_small','proof']]
    rows=rank_gaps(tree,{'root':{'toffoli':2}},fs,Formula('Q*toffoli'),{'Q':4,'toffoli':2})
    assert rows[0]['weighted_gap']==4
    assert all(r['weighted_gap'] is None for r in rows if r['floor_kind']!='proof')


def test_disjoint_composition_and_ancestor_conflicts():
    _,tree=AndFusion().build({'bits':2},{})
    def patch(name,**kw):
        r=tree.find(name)
        return Patch('re-synthesise',(name,),(r.contract.fingerprint,),(r.occurrence,),{},**kw)
    assert len(compose(patch('lane-0'),patch('lane-1'),tree))==2
    for a,b in [(patch('root'),patch('lane-1')),(patch('lane-0'),patch('lane-0')),
                (patch('lane-0',writes=('debt',)),patch('lane-1',reads=('debt',)))]:
        with pytest.raises(StrataError):compose(a,b,tree)
    assert not replace(patch('lane-0'),contracts=('bad',)).applies(tree)


def test_changed_base_and_draws_rejected(project):
    (project.root/'base.kmx').write_text('X q0')
    with pytest.raises(StrataError,match='reproduce'):project.init()
    cfg=(project.root/'strata.toml').read_text().replace('kind = "exhaustive"','kind = "sampled"\nseeds = [7]\ncompare = "expected"')
    (project.root/'strata.toml').write_text(cfg)
    with pytest.raises(StrataError,match='precommitted'):open_project(project.path)


def test_roundtrip_tree_and_trial_park(project):
    _,tree=AndFusion().build({'bits':2},{})
    assert region_from_dict(json.loads(json.dumps(asdict(tree))))==tree
    project.init()
    line=project.open_line(project.task('and').base_design(),'fuse',['lane-0','lane-1'])
    line.park('paper price too high','different boundary')
    assert all(x['status']=='priced' for _,x in project.store.all('negative'))
