import json
from pathlib import Path

import kickmix as km

from strata.restructure import restructure_file,ConstructorRegistry
from strata.cli import main


def test_generic_phase_and_data_families_construct_without_user_hypothesis(tmp_path):
    phase=tmp_path/'phase.kmx';phase.write_text('CCX q0 q1 q2\nZ q2\nCCX q0 q1 q2')
    result=restructure_file(phase,tmp_path/'phase-run',clean_inputs=(2,))
    assert result['status']=='verified_improvement'
    assert result['best']['resources']['Q']==2
    assert result['best']['resources']['toffoli']==0
    assert result['learned_template_count']>=1
    data=tmp_path/'data.kmx';data.write_text('CCX q0 q1 q2\nCX q2 q3\nCCX q0 q1 q2')
    result=restructure_file(data,tmp_path/'data-run')
    assert result['best']['resources']['toffoli']==1
    assert result['baseline']['resources']['toffoli']==2


def test_bilinear_constructor_discovers_shared_products_with_priced_helpers(tmp_path):
    source=tmp_path/'field.kmx'
    source.write_text('APPEND_TO_REGISTER q7 r0\nREGISTER r0 "workspace"\nCCX q0 q2 q4\nCCX q1 q3 q4\nCCX q0 q3 q5\nCCX q1 q2 q5\nCCX q1 q3 q5')
    result=restructure_file(source,tmp_path/'run',clean_inputs=(6,7),cost='toffoli')
    attempts=[a for a in result['attempts'] if a['mechanism']=='bilinear_tensor_decomposition']
    assert any(a['status']=='verified' and a['cost_delta']==-2 for a in attempts)
    assert result['best']['resources']['toffoli']==3


def test_nonwinning_intermediate_is_explored_and_frozen_boundary_catches_cheating(tmp_path):
    source=tmp_path/'base.kmx';source.write_text('X q0\nX q0')
    registry=ConstructorRegistry()
    def expand(c,ctx):
        return dict(status='constructed',candidate=str(c)+'\nX q0\nX q0')
    def collapse(c,ctx):
        if len(c)==4:return dict(status='constructed',candidate='APPEND_TO_REGISTER q0 r0\nREGISTER r0 "public"')
        return dict(status='unknown',reason='alternate representation not yet available')
    registry.register('expand',expand,'Test valid nonwinning intermediate')
    registry.register('collapse',collapse,'Consumer of alternative representation')
    registry.register('invalid',lambda c,ctx:dict(status='constructed',candidate='X q0'),'Incorrect change')
    result=restructure_file(source,tmp_path/'run',registry=registry,cost='instructions',rounds=2)
    assert any(a.get('cost_delta',0)>0 for a in result['attempts'])
    assert any(a['status']=='rejected' and a['check'].get('counterexample') for a in result['attempts'])
    assert result['best']['resources']['instructions']==0
    assert len(result['states'])>=3


def test_cli_clean_phase_and_domain_paths(tmp_path,capsys):
    path=tmp_path/'c.kmx';path.write_text('CCX q0 q1 q2\nZ q2\nCCX q0 q1 q2')
    assert main(['restructure',str(path),'--out',str(tmp_path/'out'),'--clean-inputs','2'])==0
    assert json.loads(capsys.readouterr().out)['status']=='verified_improvement'
