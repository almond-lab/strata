import pytest
from strata.acceptance import check_acceptance
from strata.model import StrataError
from strata.store import Store
from strata.memory import import_attempts
import json


def test_host_constraints_cannot_be_ignored():
    rules={'kind':'chemistry','error_1':'<= 0.020','E_gap':'> 3.914898'}
    check_acceptance(rules,{'error_1':0.019,'E_gap':3.915})
    with pytest.raises(StrataError):check_acceptance(rules,{'error_1':0.021,'E_gap':3.915})
    with pytest.raises(StrataError):check_acceptance(rules,{'error_1':0.019,'E_gap':3.914898})
    with pytest.raises(StrataError):check_acceptance(rules,{'error_1':0.019})
    check_acceptance({'max_fail':'1/16'},{'fail_rate':0.05})
    with pytest.raises(StrataError):check_acceptance({'max_fail':'1/16'},{'fail_rate':0.1})


def test_legacy_proof_label_without_artifact_is_not_closed(tmp_path):
    p=tmp_path/'ledger.jsonl'
    p.write_text(json.dumps({'id':'x','move':'fuse','evidence':{'kind':'proof','artifact_hash':'0'*64},'reopen':'new premise'})+'\n')
    s=Store(tmp_path/'store');r=import_attempts(s,p,{'x':'contract'})
    assert r['priced']==1 and r['closed']==0
