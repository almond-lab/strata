from pathlib import Path
import json
import pytest
from strata.file_workflow import optimize_file
from strata.discovery import discovery_brief
from strata.model import StrataError
from strata.store import Store


def test_optimized_file_can_become_discovery_parent_in_same_run(tmp_path):
    source=tmp_path/'base.kmx'
    source.write_text('CCX q0 q1 q2\nCX q2 q3\nCCX q0 q1 q2\nX q4')
    out=tmp_path/'run'
    result=optimize_file(source,out)
    assert result['status']=='improved'
    candidate=Path(result['candidate'])
    contents=candidate.read_bytes()
    brief=discovery_brief(candidate,out)
    assert brief['parent_sha256']==result['optimized']['sha256']
    assert candidate.read_bytes()==contents
    assert brief['history_total']>=1
    with pytest.raises(StrataError,match='overwrite'):
        optimize_file(candidate,out)
    assert candidate.read_bytes()==contents


def test_equivalence_sources_are_retained_in_content_store(tmp_path):
    source=tmp_path/'base.kmx'
    source.write_text('CCX q0 q1 q2\nCX q2 q3\nCCX q0 q1 q2\nX q4')
    out=tmp_path/'run'
    result=optimize_file(source,out)
    store=Store(out/'.strata')
    cert=store.get(result['equivalence_certificate'],'file_equivalence')
    for file,sha in cert['code'].items():
        assert store.read_blob(sha)==(Path(__file__).parents[1]/'src/strata'/file).read_bytes()
    assert json.loads((out/'profile.json').read_text())['candidate_order']
