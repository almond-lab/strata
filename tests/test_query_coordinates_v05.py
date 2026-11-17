import json

import kickmix as km
import pytest

from strata.cli import main
from strata.domains import query_text_coordinates
from strata.model import StrataError


@pytest.fixture
def source(tmp_path):
    circuit=km.Circuit('REGISTER r0 "data"\nAPPEND_TO_REGISTER q0 r0\nAPPEND_TO_REGISTER q1 r0\nCX q0 q1\nX q0')
    path=tmp_path/'source.kmb';circuit.to_file(path)
    return path,circuit


@pytest.mark.parametrize('command,extras',[
    ('floors',[]),('slack',['--wires','0,1']),
    ('function-of',['--wire','1','--support','0','--clean-inputs','1'])])
def test_query_and_discovery_operation_units_match(source,capsys,command,extras):
    path,circuit=source
    a,b=query_text_coordinates(circuit,0,1)
    assert a>0 and b==a+1
    assert main([command,str(path),'--from','0','--at','1',*extras])==0
    native=json.loads(capsys.readouterr().out)
    assert main([command,str(path),'--from',str(a),'--at',str(b),'--range-unit','text-line',*extras])==0
    text=json.loads(capsys.readouterr().out)
    assert native['coordinate_unit']=='operation' and native['at']==1 and native['from_at']==0
    assert native['canonical_text_window']==dict(from_at=a,at=b,coordinate_unit='text-line')
    assert text['coordinate_unit']=='text-line'
    assert native['status']==text['status']
    for key in ('value','rank','is_function','window_hash'):
        if key in native:assert native[key]==text[key]


def test_coordinate_translation_counts_tags_and_checks_eof(source):
    _,circuit=source
    total=len(str(circuit).splitlines())
    assert query_text_coordinates(circuit,2,2)==(total,total)
    with pytest.raises(StrataError,match='exceeds'):
        query_text_coordinates(circuit,1,3)
    tagged=km.Circuit('INFO_TAG "x"\nX q0')
    assert query_text_coordinates(tagged,1,2)==(1,2)


def test_query_unspecified_end_reports_resolved_operation_cut(source,capsys):
    path,_=source
    assert main(['floors',str(path)])==0
    result=json.loads(capsys.readouterr().out)
    assert result['coordinate_unit']=='operation' and result['at']==2
