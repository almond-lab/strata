"""Bundle/numeric/configuration and external-callback regressions.

The command transport test executes the real fixture verifier. Only the OS
namespace boundary is replaced because CI may disable unprivileged userns;
test_sandbox separately verifies that production never makes that fallback.
"""
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
import pytest
from strata import open as open_project
from strata.project import Project, Design
from strata.model import StrataError, digest
from strata.bundle import Bundle, BundleFrontend, Component
from strata.callbacks import CallbackAdapter, PinnedCallback, BuildReceipt, CheckReceipt
from strata.research import Hypothesis, ResearchRound, ResearchPolicy


VERIFIER = '''import json, sys, hashlib
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text())
p=Path(r['artifact'])
files={str(x.relative_to(p)):x.read_bytes() for x in p.rglob('*') if x.is_file()}
data=json.loads(files['numbers.json'])
passed=(0 < data['rank'] <= 4 and data['error'] <= 0.02)
if 'loader.kmx' in files:
    gates=files['loader.kmx'].decode().splitlines()
    passed=passed and bool(gates) and len(gates)%2==0 and all(gate=='X q0' for gate in gates)
if 'frozen.kmx' in files:
    passed=passed and files['frozen.kmx']==b'X q1\\nX q1\\n'
widths={}
for name, content in files.items():
    if name.endswith('.kmx'):
        import re
        widths[name.split('.')[0]]=max(map(int,re.findall(rb'q(\\d+)',content)),default=-1)+1
resources={'Q':max(widths.values(),default=0),'Q_total':32,'component_widths':widths,'rank':data['rank'],'error':data['error']}
def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
print(json.dumps(dict(stage=r['stage'],passed=passed,artifact_hash=r['artifact_hash'],parameters_hash=digest(r['parameters']),domain_hash=digest(r['domain']),scope=r['scope'],resources=resources,details={'checked_files':sorted(files)})))
'''


@pytest.fixture
def command_project(tmp_path, monkeypatch):
    from strata import sandbox
    calls = []
    original_transport = sandbox.run
    def trusted_test_transport(argv, workdir, **limits):
        request = json.loads(Path(argv[-1]).read_text())
        calls.append(request)
        import os
        if os.environ.get('STRATA_REAL_SANDBOX') == '1':
            return original_transport(argv, workdir, **limits)
        result = subprocess.run(argv, cwd=workdir, capture_output=True, text=True, check=True, timeout=5)
        return dict(argv=argv, sandbox='test-transport-real-verifier', output=result.stdout, returncode=result.returncode)
    monkeypatch.setattr(sandbox, 'run', trusted_test_transport)
    base=tmp_path/'base';base.mkdir()
    (base/'loader.kmx').write_bytes(b'X q0\nX q0\n')
    (base/'frozen.kmx').write_bytes(b'X q1\nX q1\n')
    (base/'numbers.json').write_text('{"rank":4,"error":0.01}')
    verifier=tmp_path/'verifier.py';verifier.write_text(VERIFIER)
    config=f'''[project]
store = ".strata"
[score]
formula = "geomean(task.base_cost / task.cost)"
[[task]]
id = "numeric"
base = "base"
frontend = "bundle"
cost = "Q_total * rank"
max_qubits = 2
[task.domain]
kind = "finite numerical contract"
[task.accept]
error = "<= 0.02"
[[task.components]]
id = "loader"
path = "loader.kmx"
kind = "kickmix"
max_qubits = 1
[[task.components]]
id = "frozen"
path = "frozen.kmx"
kind = "kickmix"
frozen = true
sha256 = "{hashlib.sha256((base/'frozen.kmx').read_bytes()).hexdigest()}"
max_qubits = 2
[[task.components]]
id = "numbers"
path = "numbers.json"
kind = "data"
[task.verifier]
command = [{json.dumps(sys.executable)}, {json.dumps(str(verifier))}, "{{request}}"]
sha256 = "{hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()}"
[task.verifier.dependencies]
{json.dumps(str(verifier))} = "{hashlib.sha256(verifier.read_bytes()).hexdigest()}"
'''
    path=tmp_path/'strata.toml';path.write_text(config)
    return open_project(path),calls


def test_config_only_bundle_full_pipeline_freezes_all_bytes(command_project):
    project,calls=command_project
    base=Design(project,project.init()['numeric'])
    assert base.record['resources']['Q']==2 and base.record['resources']['Q_total']==32
    assert base.tree.find('frozen').kind=='opaque'
    cert=project.store.get(base.record['certificate'])
    assert cert['lab']['evidence_kind']=='full_artifact' and not cert['lab']['exhaustive']
    candidate=project.root/'candidate.json';candidate.write_text('{"rank":2,"error":0.015}')
    line=project.open_line(base,'co-design-numerics',['numbers'],implementation={'files':{'numbers.json':str(candidate)}})
    candidate.write_text('this source path can change after open; its original bytes were pinned')
    with pytest.raises(StrataError,match='gate'):line.build()
    assert line.lab().passed
    built=line.build();assert built['measured_cost']==64
    design=line.verify();assert design.validate() and project.replay(design.id)['passed']
    assert project.best()['score']==2
    assert [x['stage'] for x in calls][-6:]==['lab','build_price','oracle_full','official_verifier','oracle_full','official_verifier']
    assert all(set(x['artifact_files']['files'])=={'loader.kmx','frozen.kmx','numbers.json'} for x in calls)


def test_config_only_rejects_frozen_undeclared_bad_numeric_and_width_mutations(command_project):
    project,_=command_project;project.init();base=project.task('numeric').base_design()
    for target,file,content,error in [
        ('root','frozen.kmx','X q0','frozen'),
        ('loader','numbers.json','{"rank":2,"error":0.01}','undeclared'),
        ('root','loader.kmx','X q3','width'),
        ('loader','loader.kmx','X q0','receipt failed'),
        ('numbers','numbers.json','{"rank":2,"error":0.2}','receipt failed'),
    ]:
        path=project.root/'bad';path.write_text(content)
        line=project.open_line(base,'co-design-numerics',[target],implementation={'files':{file:str(path)}})
        with pytest.raises(StrataError,match=error):line.lab()
    assert len(project.store.all('design'))==1


def test_verifier_dependency_and_bundle_bytes_cannot_change_on_replay(command_project):
    project,_=command_project;design=project.init()['numeric']
    verifier=project.root/'verifier.py';verifier.write_text(VERIFIER+'\n# changed\n')
    with pytest.raises(StrataError,match='dependency changed'):project.replay(design)


def test_full_artifact_receipt_cannot_be_rebound_to_other_artifact(command_project):
    project,_=command_project;design=Design(project,project.init()['numeric'])
    cert=project.store.get(design.record['certificate']);cert['lab']['receipt']['domain_hash']='0'*64
    forged=project.store.put('certificate',cert)
    record={**design.record,'certificate':forged}
    with pytest.raises(StrataError,match='full-artifact lab receipt'):
        Design(project,project.store.put('design',record)).validate()


CALLBACK_FRONTEND=BundleFrontend([Component('numbers','numbers.json')])

def callback_build(request):
    artifact=Bundle({'numbers.json':json.dumps({'rank':request.implementation.get('rank',4)}).encode()})
    return BuildReceipt(artifact,CALLBACK_FRONTEND.tree(artifact,'test:numeric:v1'))

def callback_check(request):
    rank=json.loads(request.artifact.files['numbers.json'])['rank']
    return CheckReceipt.for_request(request,passed=rank>0,resources={'Q':0,'component_widths':{},'rank':rank})

def raw_success(request):
    return {'passed':True}


def make_callback_project(tmp_path,check=callback_check):
    base=tmp_path/'base';base.mkdir();(base/'numbers.json').write_text('{"rank": 4}')
    path=tmp_path/'strata.toml';path.write_text('''[project]
store = ".strata"
[score]
formula = "geomean(task.base_cost / task.cost)"
[[task]]
id = "numeric"
adapter = "external"
base = "base"
cost = "rank"
''')
    adapter=CallbackAdapter(frontend=CALLBACK_FRONTEND,build=PinnedCallback(callback_build),lab=PinnedCallback(check),measure=PinnedCallback(check),oracle=PinnedCallback(check),verify=PinnedCallback(check))
    return Project(path,{'external':adapter})


def test_typed_external_callbacks_use_real_lines_and_research(tmp_path):
    project=make_callback_project(tmp_path);base=project.init()['numeric']
    hypothesis=Hypothesis('co-design-numerics',('numbers',),'numeric-rank','rank four is needed','rank two meets bound','full numeric verifier',implementation={'rank':2})
    research=ResearchRound(project,base,ResearchPolicy(deep_only=False))
    result=research.run(research.plan([hypothesis])['handle'])
    outcome=result['outcomes'][0]
    assert outcome['status']=='VERIFIED_LOCAL_GAIN' and outcome['replay']
    assert Design(project,outcome['design']).validate()


def test_raw_callback_success_is_not_verification(tmp_path):
    project=make_callback_project(tmp_path,raw_success)
    with pytest.raises(StrataError,match='typed CheckReceipt'):project.init()


def test_data_only_configuration_uses_full_numeric_gates(command_project):
    import re
    old,_=command_project
    text=old.path.read_text().replace('base = "base"','base = "numbers.json"').replace('frontend = "bundle"','frontend = "data"')
    text=re.sub(r'\[\[task\.components\]\].*?(?=\[\[task\.components\]\]|\[task\.verifier\])','',text,flags=re.S)
    old.path.write_text(text)
    (old.root/'numbers.json').write_text('{"rank":4,"error":0.01}')
    project=open_project(old.path);project.init();base=project.task('numeric').base_design()
    assert base.record['resources']['Q']==0
    candidate=project.root/'new.json';candidate.write_text('{"rank":1,"error":0.02}')
    line=project.open_line(base,'co-design-numerics',['data'],implementation={'files':{'numbers.json':str(candidate)}})
    line.lab();line.build();design=line.verify()
    assert design.record['cost']==32 and project.replay(design.id)['passed']


def test_command_frontend_rejects_unpinned_dependency_and_shell(command_project):
    project,_=command_project
    cfg=project.path.read_text()
    cfg=cfg[:cfg.index('[task.verifier.dependencies]')]
    project.path.write_text(cfg)
    with pytest.raises(StrataError,match='unpinned dependency'):open_project(project.path).init()
    import tomllib
    spec=tomllib.loads(cfg)['task'][0]
    spec['verifier']={'command':['/bin/sh','-c','echo bad','{request}'],'sha256':hashlib.sha256(Path('/bin/sh').read_bytes()).hexdigest()}
    from strata.declarative import CommandAdapter
    with pytest.raises(StrataError,match='shell verifier'):CommandAdapter(project.root,spec)


class ScaledBundleAdapter:
    def __init__(self):
        self.frozen=b'opaque precomputed host constants; no generator source'
        self.frontend=BundleFrontend([Component('mutable','mutable.kmx','kickmix'),Component('host','host.bin','opaque',True,hashlib.sha256(self.frozen).hexdigest())])

    def build(self, parameters, implementation):
        from strata.demo import AndFusion
        circuit,_=AndFusion().build(parameters,implementation)
        artifact=Bundle({'mutable.kmx':str(circuit).encode(),'host.bin':self.frozen})
        return artifact,self.frontend.tree(artifact,'scaled-bundle:v1')

    def lab_components(self, parameters, artifact):
        from strata.demo import AndFusion
        from strata.kickmix import Kickmix
        from strata.component_lab import ComponentLab
        circuit=Kickmix().parse(artifact.files['mutable.kmx'].decode())
        _,tree=AndFusion().build(parameters,{'fused':circuit.num_qubits==3*parameters['bits']})
        return {'mutable':ComponentLab(tree.contract,AndFusion.reference)}

    def oracle(self, artifact, parameters, domain):
        from strata.demo import AndFusion
        from strata.kickmix import Kickmix
        self.frontend.validate(artifact)
        result=AndFusion().oracle(Kickmix().parse(artifact.files['mutable.kmx'].decode()),parameters,domain)
        result['resources']['component_widths']=self.frontend.measured(artifact)['component_widths']
        result['checked_files']=sorted(artifact.files)
        return result

    def measure(self, artifact, parameters, domain):
        return self.oracle(artifact,parameters,domain)['resources']

    def verify(self, artifact, parameters, domain):
        return {**self.oracle(artifact,parameters,domain),'verifier_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def test_scaled_bundle_labs_exclude_only_frozen_and_full_gates_include_it(tmp_path):
    adapter=ScaledBundleAdapter()
    artifact,_=adapter.build({'bits':2},{})
    base=tmp_path/'base';adapter.frontend.write(artifact,base)
    path=tmp_path/'strata.toml';path.write_text('''[project]
store = ".strata"
[score]
formula = "geomean(task.base_cost / task.cost)"
[[task]]
id = "scaled"
adapter = "scaled"
base = "base"
cost = "Q * toffoli"
[task.scale.bits]
full = 2
small = [1]
''')
    project=Project(path,{'scaled':adapter});project.init();base=project.task('scaled').base_design()
    line=project.open_line(base,'replace-stages',['mutable'],implementation={'fused':True})
    lab=line.lab(bits=1)
    assert set(lab.components)=={'mutable'} and lab.frozen=={'host':hashlib.sha256(adapter.frozen).hexdigest()}
    line.build();design=line.verify();cert=project.store.get(design.record['certificate'])
    assert cert['full']['official']['checked_files']==['host.bin','mutable.kmx']
    assert project.replay(design.id)['passed']


def test_joint_numeric_and_circuit_edit_binds_both_components(command_project):
    project,_=command_project;project.init();base=project.task('numeric').base_design()
    numbers=project.root/'new.json';numbers.write_text('{"rank":2,"error":0.01}')
    loader=project.root/'new.kmx';loader.write_text('X q0\nX q0\nX q0\nX q0\n')
    line=project.open_line(base,'co-design-numerics',['root'],implementation={'files':{'numbers.json':str(numbers),'loader.kmx':str(loader)}})
    line.lab();line.build();design=line.verify()
    artifact=project.task('numeric').parse_artifact(project.store.read_blob(design.record['artifact']))
    assert artifact.files['numbers.json']==numbers.read_bytes() and artifact.files['loader.kmx']==loader.read_bytes()
    assert project.replay(design.id)['passed']


def test_later_component_edit_preserves_parent_numeric_change(command_project):
    project,_=command_project;project.init();base=project.task('numeric').base_design()
    numbers=project.root/'new.json';numbers.write_text('{"rank":2,"error":0.01}')
    first=project.open_line(base,'co-design-numerics',['numbers'],implementation={'files':{'numbers.json':str(numbers)}})
    first.lab();first.build();parent=first.verify()
    loader=project.root/'new.kmx';loader.write_text('X q0\nX q0\nX q0\nX q0\n')
    second=project.open_line(parent,'re-synthesise',['loader'],implementation={'files':{'loader.kmx':str(loader)}})
    second.lab();second.build();design=second.verify()
    assert design.record['resources']['rank']==2 and design.validate()


def test_bundle_rejects_path_aliases_and_traversal():
    for path in ('../outside','/absolute','a//b','a/./b','a/../b','a/'):
        with pytest.raises(StrataError,match='component path'):
            Bundle({path:b'bytes'})


def circuit_callback_build(request):
    from strata.demo import AndFusion
    return BuildReceipt(*AndFusion().build(request.parameters,request.implementation))


def circuit_callback_check(request):
    from strata.demo import AndFusion
    checked=AndFusion().oracle(request.artifact,request.parameters,request.domain)
    return CheckReceipt.for_request(request,passed=checked['passed'],resources=checked['resources'])


def circuit_callback_audit(request):
    from strata.demo import AndFusion
    from strata.callbacks import StructuralReceipt
    witness=AndFusion().audit_restructure(request.before,request.old_tree,request.after,request.new_tree)
    return StructuralReceipt(witness['passed'],witness['mechanism'],witness['checks'],request.before_artifact,request.after_artifact)


def test_pinned_external_callbacks_qualify_deep_structural_research(tmp_path):
    from strata.kickmix import Kickmix
    source=Path(__file__).parents[1]/'examples'/'and_fusion'
    (tmp_path/'base.kmx').write_bytes((source/'base.kmx').read_bytes())
    (tmp_path/'strata.toml').write_text((source/'strata.toml').read_text().replace('and_fusion','external'))
    adapter=CallbackAdapter(frontend=Kickmix(),build=PinnedCallback(circuit_callback_build),
                            lab=PinnedCallback(circuit_callback_check),measure=PinnedCallback(circuit_callback_check),
                            oracle=PinnedCallback(circuit_callback_check),verify=PinnedCallback(circuit_callback_check),
                            audit_restructure=PinnedCallback(circuit_callback_audit))
    project=Project(tmp_path/'strata.toml',{'external':adapter});base=project.init()['and']
    hypothesis=Hypothesis('replace-stages',('root',),'materialization','product word is needed','direct AND into output','complete boundary tests',implementation={'fused':True})
    research=ResearchRound(project,base)
    result=research.run(research.plan([hypothesis])['handle'])
    assert result['qualified_gains']==1 and result['outcomes'][0]['replay']
    witness=result['outcomes'][0]['structure']['structural_witness']
    assert witness['verifier_hash']==adapter.callbacks['audit_restructure'].identity_hash()


def test_merge_distinct_component_files_keeps_existing_parent_bytes(command_project):
    original,_=command_project
    (original.root/'base'/'aux.dat').write_bytes(b'original auxiliary data')
    original.path.write_text(original.path.read_text()+'\n[[task.components]]\nid = "aux"\npath = "aux.dat"\nkind = "data"\n')
    project=open_project(original.path);project.init();base=project.task('numeric').base_design()
    aux=project.root/'updated.dat';aux.write_bytes(b'parent auxiliary data retained')
    ancestry=project.open_line(base,'co-design-numerics',['aux'],implementation={'files':{'aux.dat':str(aux)}})
    ancestry.lab();ancestry.build();parent=ancestry.verify()
    loader=project.root/'new.kmx';loader.write_text('X q0\nX q0\nX q0\nX q0\n')
    numbers=project.root/'new.json';numbers.write_text('{"rank":2,"error":0.01}')
    first=project.open_line(parent,'re-synthesise',['loader'],implementation={'files':{'loader.kmx':str(loader)}})
    second=project.open_line(parent,'co-design-numerics',['numbers'],implementation={'files':{'numbers.json':str(numbers)}})
    for line in (first,second):line.lab();line.build();line.verify()
    merged=project.merge(first.id,second.id)
    with pytest.raises(StrataError,match='gate'):merged.verify()
    merged.lab();merged.build();design=merged.verify()
    artifact=project.task('numeric').parse_artifact(project.store.read_blob(design.record['artifact']))
    assert artifact.files['aux.dat']==aux.read_bytes()
    assert artifact.files['loader.kmx']==loader.read_bytes()
    assert artifact.files['numbers.json']==numbers.read_bytes()
    assert design.validate() and project.replay(design.id)['passed']


def test_merge_rejects_conflicting_component_bytes(command_project):
    project,_=command_project;project.init();parent=project.task('numeric').base_design()
    lines=[]
    for rank in (2,3):
        candidate=project.root/f'rank-{rank}.json';candidate.write_text(json.dumps({'rank':rank,'error':0.01}))
        line=project.open_line(parent,'co-design-numerics',['numbers'],implementation={'files':{'numbers.json':str(candidate)}})
        line.lab();line.build();line.verify();lines.append(line)
    with pytest.raises(StrataError,match='conflict|overlapping'):
        project.merge(lines[0].id,lines[1].id)


def test_discovery_accepts_raw_operation_streams(tmp_path):
    import kickmix as km
    from strata.frontends import OperationStream
    from strata.discovery import discovery_brief, evaluate_candidate, record_research_note
    codec=OperationStream()
    parent=tmp_path/'parent.ops';candidate=tmp_path/'candidate.ops'
    parent.write_text(codec.serialize(km.Circuit('CCX q0 q1 q2\nCCX q0 q1 q2\nX q2')))
    candidate.write_text(codec.serialize(km.Circuit('X q2')))
    report=evaluate_candidate(parent,candidate,tmp_path/'evaluated')
    assert report['checks']['equivalent'] is True and report['cost_delta']<0
    brief=discovery_brief(parent,tmp_path/'brief')
    note=record_research_note(parent,tmp_path/'notes',{'note_class':'observation','observation':'raw operation frontend'})
    assert brief['parent_sha256']==report['parent_sha256']==note['parent_sha256']
    assert 'frontends.py' in report['code']
