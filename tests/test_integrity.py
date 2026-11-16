from pathlib import Path
import json
import numpy as np
import kickmix as km
import pytest
from strata.artifacts import manifest,verify_manifest
from strata.model import StrataError
from strata.memory import import_attempts
from strata.store import Store
from strata.trace import execution_profile
from strata.blocks import Block,BlockLibrary,product_xor_block
from strata.demo import AndFusion
from strata import open as open_project


def test_bundle_integrity_and_traversal(tmp_path):
    (tmp_path/'loader.kmx').write_text('X q0')
    (tmp_path/'factorization.dat').write_bytes(b'123')
    m=manifest(tmp_path);assert len(m['files'])==2
    assert verify_manifest(tmp_path,m)
    (tmp_path/'loader.kmx').write_text('X q1')
    with pytest.raises(StrataError):verify_manifest(tmp_path,m)
    (tmp_path/'outside').symlink_to('/etc/passwd')
    with pytest.raises(StrataError):manifest(tmp_path)


def test_322_legacy_estimates_never_become_closed(tmp_path):
    rows=[dict(id=str(i),mechanism='fuse',status='closed',evidence='paper estimate proves impossible',reopen_if='new representation') for i in range(322)]
    file=tmp_path/'attempts.jsonl';file.write_text('\n'.join(json.dumps(r) for r in rows))
    s=Store(tmp_path/'store')
    counts=import_attempts(s,file,{str(i):'fp'+str(i) for i in range(322)})
    assert counts['priced']==322 and counts['closed']==0
    assert all(v['status']=='priced' for _,v in s.all('negative'))


def test_trace_measures_conditions_and_does_not_assert_slack():
    c=km.Circuit('CCX q0 q1 q2\nHMR q2 b0\nCZ q0 q1 if b0')
    initial=np.zeros((3,64),dtype=bool);initial[:2]=True
    r=execution_profile(c,initial,seed=10)
    assert r['executed_cost']['mean_toffoli']==1
    assert r['conditionals'][0]['last_assignment']==1
    assert r['slack']['status']=='unknown' and not r['fingerprints']['proof']


def test_tag_or_pattern_without_proof_is_opaque():
    lib=BlockLibrary();lib.register(Block('fake',lambda x:{},lambda x,b:{'passed':True,'kind':'sampled'}))
    assert lib.recognize([])['kind']=='opaque'
    lib.register(product_xor_block())
    ops=[('CCX',(2,4,6)),('CX',(6,8)),('CCX',(2,4,6))]
    r=lib.recognize(ops)
    assert r['kind']=='materialized_product_xor' and r['binding']['promise']=='scratch=0'
    assert lib.recognize(ops[:2])['kind']=='opaque'


def test_declared_region_scope_is_enforced(tmp_path):
    example=Path(__file__).parents[1]/'examples/and_fusion'
    for n in ['strata.toml','base.kmx']:(tmp_path/n).write_bytes((example/n).read_bytes())
    p=open_project(tmp_path/'strata.toml');p.init()
    line=p.open_line(p.task('and').base_design(),'fuse',['lane-0'],implementation={'fused':True})
    # Small bits=1 has only this lane, but full build changes unlisted lanes.
    assert line.lab(bits=1).passed
    with pytest.raises(StrataError,match='outside declared regions'):line.build()


def test_stale_artifact_certificate_cannot_win_best(tmp_path):
    example=Path(__file__).parents[1]/'examples/and_fusion'
    for n in ['strata.toml','base.kmx']:(tmp_path/n).write_bytes((example/n).read_bytes())
    p=open_project(tmp_path/'strata.toml');p.init();d=p.task('and').base_design()
    (p.store.path/'objects'/d.record['artifact']).write_bytes(b'forged')
    with pytest.raises(StrataError,match='integrity'):p.best()
