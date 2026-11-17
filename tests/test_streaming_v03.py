import json
import sqlite3

import kickmix as km

from strata.streaming import scan_circuit
from strata.discovery import discovery_brief
from strata.cli import main


def test_repeated_large_kernels_group_across_wire_renaming(tmp_path):
    # Delimiters are observations; the entire 1,330-op run is the recovery target.
    def kernel(a,b,c):
        return '\n'.join([f'CX q{a} q{b}',f'CX q{a} q{b}']*664+[f'CCX q{a} q{b} q{c}',f'CCX q{a} q{b} q{c}'])
    text=kernel(0,1,2)+'\nINFO_TAG "cut"\n'+kernel(3,4,5)
    c=km.Circuit(text)
    result=scan_circuit(c,tmp_path,max_window=8192)
    group=next(g for g in result['groups'] if g['operations_per_occurrence']==1330)
    assert group['occurrences']==2 and group['aggregate_static_toffoli']==4
    assert group['semantics']['status']=='exact'
    assert group['examples'][0]['wire_mapping'] != group['examples'][1]['wire_mapping']
    with sqlite3.connect(result['index']) as db:
        assert db.execute('select count(*) from occurrences where fingerprint=?',(group['fingerprint'],)).fetchone()[0]==2


def test_context_cut_never_turns_conditional_body_unconditional(tmp_path):
    c=km.Circuit('PUSH_CONDITION if b0\nX q0\nX q1\nPOP_CONDITION\nCCX q0 q1 q2')
    r=scan_circuit(c,tmp_path,start=1,stop=4,max_window=1)
    assert r['scanned_operations']==3
    assert all(g['semantics']['status']=='unknown' for g in r['groups'])


def test_large_discover_avoids_full_string_profile_and_emits_progress(tmp_path,monkeypatch):
    path=tmp_path/'repeated.kmb'
    km.Circuit(('CCX q0 q1 q2\nCX q2 q3\nCCX q0 q1 q2\nINFO_TAG "boundary"\n')*100).to_file(path,format='kmb')
    def fail(*a,**kw):raise AssertionError('large path must not enter whole-file Python profile')
    monkeypatch.setattr('strata.file_workflow.inspect_file',fail)
    updates=[]
    result=discovery_brief(path,tmp_path/'report',streaming_threshold=1,progress=updates.append)
    assert result['kind']=='streaming_discovery_brief'
    assert result['targets'][0]['occurrences']==100
    assert updates[0]['stage']=='native_parse' and updates[-1]['stage']=='scan_complete'
    assert result['scan']['seconds_per_million_operations']>0
    assert result['scan']['memory_model'].startswith('one native circuit')


def test_cli_range_produces_bounded_report(tmp_path,capsys):
    path=tmp_path/'source.kmx';path.write_text('X q0\nCCX q0 q1 q2\nX q2')
    assert main(['discover',str(path),'--out',str(tmp_path/'out'),'--range','1:3'])==0
    captured=capsys.readouterr()
    result=json.loads(captured.out)
    assert result['scan']['range']=={'start':1,'stop':3}
    assert json.loads(captured.err.splitlines()[-1])['stage']=='scan_complete'
