import json

from strata.cli import main
from strata.store import Store


def invoke(tmp_path,capsys,command,spec):
    source=tmp_path/(command+'.json');source.write_text(json.dumps(spec))
    output=tmp_path/(command+'-result.json')
    assert main([command,str(source),'--out',str(output)])==0
    result=json.loads(capsys.readouterr().out)
    assert result==json.loads(output.read_text())
    return result


def test_numeric_cli_keeps_scoped_budgets_floors_and_unknown_forecasts(tmp_path,capsys):
    budget=invoke(tmp_path,capsys,'numeric-budget',dict(unit='error_squared',limit=10,
        terms=[dict(id='fit',evidence=dict(value=3,unit='error_squared',evidence_kind='model',source_hash='a'*64))],
        groups={'stage':['fit'],'block':['fit']},roots=['stage','block']))
    assert budget['total']==3 and budget['remaining']==7
    floor=invoke(tmp_path,capsys,'numeric-floor',dict(hessian=[[2,1],[1,2]],target=['1/2',0],grids={'0':[0,1]},continuous=[1]))
    assert floor['minimum']=='3/8'
    assert floor['certificate_scope']=='supplied finite rational model only'
    run=dict(run_id='one',independent=True,checkpoint='fixed',objective='error',unit='absolute',
             points=[dict(step=i,value=1/(i+1)) for i in range(5)])
    forecast=invoke(tmp_path,capsys,'numeric-forecast',dict(runs=[run],checkpoint='fixed',objective='error',unit='absolute',target=.1,horizon=10))
    assert forecast['status']=='insufficient_evidence'


def test_numeric_bets_and_saved_reports_do_not_claim_certification(tmp_path,capsys):
    result=invoke(tmp_path,capsys,'numeric-bets',dict(current_costs={'a':10,'b':10},baseline_costs={'a':100,'b':100},
        bets=[dict(id='trial',costs={'a':5},probability=dict(value=.25,evidence_kind='judgment'))]))
    assert result['bets'][0]['evidence_kind']=='judgment'
    assert result['bets'][0]['expected_score']>result['safe_score']
    store=Store(tmp_path/'records')
    handle=store.put('numeric_score',dict(candidate='sample',cost=12,cost_unit='operations',evidence_kind='model',eligible=False,frozen_parameters={}))
    store.db.close()
    assert main(['report','--store',str(tmp_path/'records'),'--handle',handle,'--format','markdown'])==0
    report=capsys.readouterr().out
    assert 'Numeric candidates and experiments' in report and 'sample' in report
