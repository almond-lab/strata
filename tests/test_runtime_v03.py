"""Regression coverage for binary bundles, external outcomes and host pricing."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from dataclasses import replace

import kickmix as km
import pytest

from strata.bundle import Bundle, BundleFrontend, Component
from strata.callbacks import CheckReceipt, ExternalCheckError, PinnedCallback
from strata.cost import Formula
from strata.declarative import CommandAdapter
from strata.model import StrataError
from strata.probability import ProbabilityPolicy
from strata import sandbox
from test_task_issues import make_callback_project, callback_check, command_project


def test_binary_bundle_preserves_bytes_and_enforces_width(tmp_path):
    buffer=io.BytesIO()
    km.Circuit('X q3\nX q3').to_file(buffer,format='kmb')
    binary=buffer.getvalue()
    front=BundleFrontend([Component('loader','loader.kmb','kickmix',max_qubits=4)])
    bundle=Bundle({'loader.kmb':binary})
    assert front.measured(bundle)['component_widths']=={'loader':4}
    assert list(front.operations(bundle))==['X','X']
    assert front.parse(front.serialize(bundle)).files['loader.kmb']==binary
    front.write(bundle,tmp_path)
    assert front.load(tmp_path).files['loader.kmb']==binary
    assert front.analyze(bundle,tmp_path/'report')['metrics']['Q']==4
    with pytest.raises(StrataError,match='width cap'):
        BundleFrontend([Component('loader','loader.kmb','kickmix',max_qubits=3)]).validate(bundle)


def reject_check(request):
    return CheckReceipt.for_request(request,passed=False,details={'condition':'missing correction','count':2})


def wrong_scope_check(request):
    if json.loads(request.artifact.files['numbers.json'])['rank']==4:return callback_check(request)
    return replace(reject_check(request),domain_hash='0'*64)


def reject_lab(request):
    if request.stage=='lab' and json.loads(request.artifact.files['numbers.json'])['rank']==2:return reject_check(request)
    return callback_check(request)


def reject_measure(request):
    if request.stage=='build_price' and json.loads(request.artifact.files['numbers.json'])['rank']==2:return reject_check(request)
    return callback_check(request)


def reject_verify(request):
    if request.stage=='official_verifier' and json.loads(request.artifact.files['numbers.json'])['rank']==2:return reject_check(request)
    return callback_check(request)


@pytest.mark.parametrize('gate,callback_name,expected_stage',[
    ('lab','lab','lab'),('build','measure','build_price'),('verify','verify','verified')])
def test_line_retains_scoped_rejection_at_every_gate(tmp_path,gate,callback_name,expected_stage):
    project=make_callback_project(tmp_path,{'lab':reject_lab,'measure':reject_measure,'verify':reject_verify}[callback_name])
    base=project.init()['numeric']
    line=project.open_line(base,'co-design-numerics',['numbers'],implementation={'rank':2})
    if gate!='lab':line.lab()
    if gate=='verify':line.build()
    with pytest.raises(ExternalCheckError,match='missing correction'):
        getattr(line,gate)()
    event=project.store.events(line.id)[-1]
    assert event['stage']==expected_stage and event['evidence_kind']=='measured_rejection'
    assert event['receipt']['passed'] is False and event['scope_valid']
    stored=json.loads(project.store.read_blob(event['receipt_artifact']))
    assert stored['receipt']['details']['count']==2
    assert hashlib.sha256(project.store.read_blob(event['artifact'])).hexdigest()==event['artifact']
    assert project.store.all('negative')[-1][1]['evidence_kind']=='build'


def test_scope_mismatch_recorded_without_measured_negative(tmp_path):
    project=make_callback_project(tmp_path,wrong_scope_check)
    base=project.init()['numeric']
    line=project.open_line(base,'co-design-numerics',['numbers'],implementation={'rank':2})
    with pytest.raises(ExternalCheckError,match='scope/hash mismatch'):
        line.lab()
    event=project.store.events(line.id)[-1]
    assert event['evidence_kind']=='invalid_external_evidence'
    assert not project.store.all('negative')


def test_failed_command_receipt_retains_diagnostics(command_project):
    project,_=command_project
    base=project.init()['numeric']
    candidate=project.root/'bad.json';candidate.write_text('{"rank":2,"error":0.2}')
    line=project.open_line(base,'co-design-numerics',['numbers'],implementation={'files':{'numbers.json':str(candidate)}})
    with pytest.raises(ExternalCheckError,match='checked_files'):line.lab()
    event=project.store.events(line.id)[-1]
    assert event['diagnostics']['limits']['memory_bytes']==2*1024**3
    assert event['diagnostics']['verifier_hash']


def test_virtualenv_invocation_and_environment_identity(tmp_path,monkeypatch):
    env=tmp_path/'venv'
    subprocess.run([sys.executable,'-m','venv','--without-pip',str(env)],check=True)
    python=env/'bin/python'
    site=next(env.glob('lib/python*/site-packages'))
    (site/'only_in_test_venv.py').write_text('VALUE=7\n')
    metadata=site/'fixture-1.dist-info';metadata.mkdir()
    (metadata/'METADATA').write_text('Name: fixture\nVersion: 1\n')
    base=tmp_path/'data.json';base.write_text('{}')
    verifier=tmp_path/'verify.py'
    verifier.write_text('''import json,sys,hashlib
from pathlib import Path
from only_in_test_venv import VALUE
r=json.loads(Path(sys.argv[1]).read_text())
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
print(json.dumps(dict(stage=r['stage'],passed=True,artifact_hash=r['artifact_hash'],parameters_hash=digest(r['parameters']),domain_hash=digest(r['domain']),scope=r['scope'],resources={'value':VALUE},details={})))
''')
    config=dict(base='data.json',frontend='data',verifier=dict(command=[str(python),str(verifier),'{request}'],
        sha256=hashlib.sha256(python.read_bytes()).hexdigest(),
        dependencies={str(verifier):hashlib.sha256(verifier.read_bytes()).hexdigest()},
        memory_bytes=3*1024**3,max_file_bytes=4*1024**2,max_output=8192,cpu_seconds=5))
    if os.environ.get('STRATA_REAL_SANDBOX')!='1':
        def transport(argv,workdir,**limits):
            result=subprocess.run(argv,cwd=workdir,capture_output=True,text=True,check=True)
            return dict(output=result.stdout,sandbox='test-transport',limits=limits)
        monkeypatch.setattr(sandbox,'run',transport)
    adapter=CommandAdapter(tmp_path,config)
    assert adapter.command[0]==str(python)
    artifact,_=adapter.build({},{})
    receipt=adapter.verify(artifact,{}, {})
    assert receipt['resources']['value']==7
    assert receipt['sandbox_limits']['max_file_bytes']==4*1024**2
    assert receipt['runtime_environment']['kind']=='python_virtualenv'
    before=adapter.identity_hash()
    (metadata/'METADATA').write_text('Name: fixture\nVersion: 2\n')
    with pytest.raises(StrataError,match='environment changed'):adapter.identity_hash()


def test_restricted_math_and_count_aware_confidence_penalty():
    policy=ProbabilityPolicy((17,29),penalty='log(2)/-ln(0.5+2*sqrt(eps_plus)) + fail_count/shots')
    draws=[dict(seed=seed,resources=dict(Q=2,toffoli=3,fail_rate=0,fail_count=0,shots=1000,eps_plus=.001),
        confidence_bounds=dict(eps_plus=dict(kind='upper',method='host exact binomial',confidence=.95,fail_count=0,shots=1000))) for seed in policy.seeds]
    result=policy.evaluate(dict(seeds=[17,29],draws=draws),'R(fail_rate)*Q^2*toffoli')
    assert result['cost']>12
    assert result['failure_inputs'][0]['empirical_rate']==0
    assert result['failure_inputs'][0]['confidence_bounds']['eps_plus']['value']==.001
    assert Formula('max(1,min(3,exp(ln(2))))')({})==2
    for expression in ['__import__("os")','math.log(2)','sqrt(-1)','exp(10000)','log(0)','1/0']:
        with pytest.raises(StrataError):Formula(expression)({})
    draws[0]['resources']['fail_count']=2
    with pytest.raises(StrataError,match='disagrees'):policy.evaluate(dict(seeds=[17,29],draws=draws),'Q')


@pytest.mark.parametrize('values',[{'memory_bytes':0},{'cpu_seconds':False},{'max_file_bytes':-1},{'max_output':1.1},{'timeout':float('inf')}])
def test_sandbox_limits_are_strict(values):
    with pytest.raises(StrataError):sandbox.effective_limits(**values)


def test_capture_limit_independent_of_file_limit(tmp_path,monkeypatch):
    original=sandbox.subprocess.Popen
    monkeypatch.setattr(sandbox.shutil,'which',lambda _: '/usr/bin/bwrap')
    def transport(command,**kwargs):
        return original(command[command.index('--')+1:],cwd=tmp_path,**kwargs)
    monkeypatch.setattr(sandbox.subprocess,'Popen',transport)
    result=sandbox.run([sys.executable,'-c',"open('large','wb').write(b'x'*2000000); print('ok')"],tmp_path,max_output=100,max_file_bytes=3000000)
    assert result['output']=='ok\n' and (tmp_path/'large').stat().st_size==2000000
    with pytest.raises(StrataError,match='captured output'):
        sandbox.run([sys.executable,'-c',"print('x'*1000)"],tmp_path,max_output=100)


def binary_command_project(command_project, *, declare_data=False):
    from strata import open as open_project
    project,calls=command_project
    base=project.root/'base'
    circuit=km.Circuit.from_file(base/'loader.kmx')
    (base/'loader.kmx').unlink()
    circuit.to_file(base/'loader.kmb',format='kmb')
    verifier=project.root/'verifier.py'
    text=verifier.read_text()
    text=text.replace("if 'loader.kmx' in files:","if 'loader.kmx' in files:")
    text=text.replace("resources={'Q':", "import kickmix as km\nfor name in files:\n    if name.endswith('.kmb'):\n        widths[name.split('.')[0]]=km.Circuit.from_file(p/name).num_qubits\nresources={'Q':")
    old=hashlib.sha256(verifier.read_bytes()).hexdigest()
    verifier.write_text(text)
    config=project.path.read_text().replace(old,hashlib.sha256(verifier.read_bytes()).hexdigest()).replace('loader.kmx','loader.kmb')
    if declare_data:
        config=config.replace('path = "loader.kmb"\nkind = "kickmix"\nmax_qubits = 1','path = "loader.kmb"\nkind = "data"')
    project.path.write_text(config)
    return open_project(project.path),calls


def test_native_binary_bundle_config_ingest_and_replay(command_project):
    project,calls=binary_command_project(command_project)
    design=project.init()['numeric']
    stored=project.store.get(design,'design')
    assert stored['resources']['component_widths']['loader']==1
    assert stored['resources']['Q']==2
    assert project.replay(design)['passed']
    assert len(calls)>0


def test_binary_data_declaration_fails_before_external_run(command_project):
    project,calls=binary_command_project(command_project,declare_data=True)
    with pytest.raises(StrataError,match='declare kind="kickmix"'):project.init()
    assert not calls


def wrong_width_measure(request):
    receipt=callback_check(request)
    rank=json.loads(request.artifact.files['numbers.json'])['rank']
    if request.stage=='build_price' and rank==2:
        return replace(receipt,resources={**receipt.resources,'Q':889})
    return receipt


def wrong_width_verify(request):
    receipt=callback_check(request)
    rank=json.loads(request.artifact.files['numbers.json'])['rank']
    if request.stage=='official_verifier' and rank==2:
        return replace(receipt,resources={**receipt.resources,'Q':889})
    return receipt


@pytest.mark.parametrize('check,gate',[(wrong_width_measure,'build'),(wrong_width_verify,'verify')])
def test_completed_receipt_survives_postcheck_resource_rejection(tmp_path,check,gate):
    project=make_callback_project(tmp_path,check)
    base=project.init()['numeric']
    line=project.open_line(base,'co-design-numerics',['numbers'],implementation={'rank':2})
    line.lab()
    if gate=='verify':line.build()
    with pytest.raises(ExternalCheckError,match='width disagrees'):getattr(line,gate)()
    event=project.store.events(line.id)[-1]
    assert event['receipt']['passed'] is True
    assert event['receipt']['resources']['Q']==889
    assert event['evidence_kind']=='postcheck_rejection'
    assert project.store.all('external_check_failure')
    assert not project.store.all('negative')


def always_wrong_width_measure(request):
    receipt=callback_check(request)
    if request.stage=='build_price':return replace(receipt,resources={**receipt.resources,'Q':889})
    return receipt


def test_ingest_retains_completed_receipt_without_a_line(tmp_path):
    project=make_callback_project(tmp_path,always_wrong_width_measure)
    with pytest.raises(ExternalCheckError,match='width disagrees'):project.init()
    record=project.store.all('external_check_failure')[-1][1]
    assert record['receipt']['resources']['Q']==889
    assert json.loads(project.store.read_blob(record['receipt_artifact']))==record['receipt']
