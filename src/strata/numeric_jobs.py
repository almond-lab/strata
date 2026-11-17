"""Checkpointed numeric jobs and explicit safe-candidate / bet accounting.

Jobs run a pinned worker through the existing offline sandbox, one checkpoint
stage at a time. Immutable policies, reserved wall budgets and file locks survive
restarts; the library is not a cluster scheduler. Numeric success never bypasses
the project's independent lab/build/full-verification gates.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import fcntl
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
import tempfile
import time
import uuid

from .model import StrataError, digest, canonical


def _finite(value, name, positive=False):
    if type(value) not in (int,float) or not math.isfinite(value) or (positive and value<=0):
        raise StrataError(name+' must be finite'+(' and positive' if positive else ''))
    return value


def _hash_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _runtime(invocation):
    from .declarative import CommandAdapter
    proxy=object.__new__(CommandAdapter)
    proxy.executable=Path(invocation)
    proxy.environment_root=proxy.executable.parent.parent if (proxy.executable.parent.parent/'pyvenv.cfg').is_file() else None
    return dict(invocation=str(invocation),executable_sha256=_hash_file(invocation),
                environment=proxy._environment_identity())


def _rules_match(rules,observed,stage,require_all=False):
    ops={'>':lambda a,b:a>b,'>=':lambda a,b:a>=b,'<':lambda a,b:a<b,'<=':lambda a,b:a<=b}
    matched=[rule for rule in rules if stage>=rule.get('after_stages',1)
             and observed[rule['metric']]['evidence_kind']=='measured'
             and ops[rule['op']](observed[rule['metric']]['value'],rule['value'])]
    return len(matched)==len(rules) if require_all else matched


def _put_atomic(store,records):
    rows=[(digest(dict(kind=kind,body=body)),kind,canonical(body)) for kind,body in records]
    with store.db:
        store.db.executemany('INSERT INTO records(id,kind,body) VALUES (?,?,?)',rows)


@dataclass(frozen=True)
class NumericCheckpoint:
    job: str
    policy_hash: str
    stage: int
    checkpoint_hash: str
    checkpoint_bytes: bytes
    metrics: dict


class NumericJob:
    def __init__(self,store,handle):
        self.store=store;self.id=handle;self._promotion=None
        store.get(handle,'numeric_job')

    @property
    def policy(self):
        return self.store.get(self.id,'numeric_job')

    @classmethod
    def create(cls,store,*,contract,base_artifact,command,dependencies,metric_units,
               max_stages,max_wall_seconds,stage_timeout,stop_rules=(),
               initial_checkpoint=None,settings=None,memory_bytes=2*1024**3,
               max_file_bytes=128*1024**2,max_output=1_000_000,cpu_seconds=None,
               readiness_rules=(),promotion=None):
        from .sandbox import effective_limits
        if not isinstance(contract,str) or not contract or not isinstance(metric_units,dict) or not metric_units:
            raise StrataError('numeric job needs a contract and named metric units')
        if any(not isinstance(k,str) or not k.strip() or not isinstance(v,str) or not v.strip() for k,v in metric_units.items()):
            raise StrataError('metric units must be nonempty strings')
        if settings is not None and (not isinstance(settings,dict) or any(not isinstance(k,str) for k in settings)):
            raise StrataError('numeric settings must be a mapping with string keys')
        if not isinstance(stop_rules,(list,tuple)) or not isinstance(readiness_rules,(list,tuple)):
            raise StrataError('numeric rules must be explicit lists')
        if type(max_stages)is not int or max_stages<1:raise StrataError('max_stages must be positive')
        _finite(max_wall_seconds,'wall budget',True);_finite(stage_timeout,'stage timeout',True)
        store.read_blob(base_artifact)
        if initial_checkpoint:store.read_blob(initial_checkpoint)
        if not isinstance(command,(list,tuple)) or not command or any(not isinstance(a,str) for a in command) or '{request}' not in command:
            raise StrataError('numeric worker command needs an exact {request} argv placeholder')
        invocation=Path(command[0])
        if not invocation.is_absolute() or not invocation.is_file():raise StrataError('numeric worker executable must exist')
        if isinstance(dependencies,(str,bytes)):
            raise StrataError('dependencies must be an explicit path list')
        pinned={str(Path(p).resolve()):_hash_file(p) for p in dependencies}
        command=list(command)
        for i,arg in enumerate(command[1:],1):
            if arg!='{request}' and Path(arg).is_file():
                resolved=str(Path(arg).resolve())
                if resolved not in pinned:raise StrataError('numeric worker file argument is an unpinned dependency')
                command[i]=resolved
        # Scripts and data must be explicitly declared dependencies. Executable
        # identity and a venv configuration are also frozen without relocating it.
        runtime=_runtime(invocation)
        limits=effective_limits(timeout=stage_timeout,memory_bytes=memory_bytes,max_file_bytes=max_file_bytes,
                                max_output=max_output,cpu_seconds=cpu_seconds)
        rules=[]
        for rule in list(stop_rules)+list(readiness_rules):
            if (not isinstance(rule,dict) or rule.get('metric') not in metric_units or
                set(rule)-{'metric','op','value','after_stages','unit'} or
                rule.get('op') not in {'>','>=','<','<='} or type(rule.get('after_stages',1))is not int or rule.get('after_stages',1)<1):
                raise StrataError('invalid predeclared numeric stop rule')
            if 'unit' in rule and rule['unit']!=metric_units[rule['metric']]:raise StrataError('numeric rule units disagree')
            _finite(rule.get('value'),'stop threshold')
            rules.append(dict(rule,unit=metric_units[rule['metric']]))
        from .callbacks import PinnedCallback
        if bool(readiness_rules)!=(promotion is not None) or promotion is not None and not isinstance(promotion,PinnedCallback):
            raise StrataError('automatic readiness needs rules and a PinnedCallback packager')
        promotion_hash=promotion.identity_hash() if promotion is not None else None
        body=dict(version=1,instance_id=uuid.uuid4().hex,contract=contract,base_artifact=base_artifact,command=[str(invocation),*command[1:]],
                  dependencies=pinned,runtime=runtime,metric_units=metric_units,
                  max_stages=max_stages,max_wall_seconds=max_wall_seconds,limits=limits,
                  stop_rules=rules[:len(stop_rules)],readiness_rules=rules[len(stop_rules):],promotion_hash=promotion_hash,
                  initial_checkpoint=initial_checkpoint,settings=settings or {},
                  controller_sources={name:_hash_file(Path(__file__).parent/name)
                                      for name in ('numeric_jobs.py','sandbox.py','declarative.py','callbacks.py')},
                  evidence_scope='worker-reported numeric measurements; host acceptance requires independent verification')
        job=cls(store,store.put('numeric_job',body));job._promotion=promotion
        return job

    def _records(self,kind):
        return [(h,row) for h,row in self.store.all(kind) if row.get('job')==self.id]

    def status(self):
        starts=self._records('numeric_job_start');ends=self._records('numeric_job_finish')
        finished={row['attempt']:row for _,row in ends}
        spent=sum(finished[row['attempt']]['seconds'] if row['attempt'] in finished else row['reserved_seconds'] for _,row in starts)
        checkpoints=self._records('numeric_checkpoint')
        latest=checkpoints[-1][1] if checkpoints else None
        stage=latest['stage'] if latest else 0
        terminal=next((row['status'] for _,row in reversed(ends) if row['status']=='stopped_by_rule'),None)
        state=terminal or ('complete' if stage>=self.policy['max_stages'] else
                          'budget_exhausted' if spent>=self.policy['max_wall_seconds'] else 'resumable')
        promotions=self._records('numeric_job_promotion')
        promoted=next((dict(handle=h,**r) for h,r in reversed(promotions) if r['status']=='verified' and latest and r['checkpoint']==latest['checkpoint']),None)
        if promoted and not terminal:state='promoted'
        return dict(job=self.id,status=state,completed_stages=stage,spent_wall_seconds=spent,promotion=promoted,
                    remaining_wall_seconds=max(0,self.policy['max_wall_seconds']-spent),
                    checkpoint=latest['checkpoint'] if latest else self.policy['initial_checkpoint'],
                    latest=latest,attempts=len(starts),unfinished_attempts=sum(row['attempt'] not in finished for _,row in starts),
                    accounting='unfinished attempts conservatively charged their full reserved timeout')

    @contextmanager
    def _lock(self):
        path=self.store.path/f'numeric-job-{self.id}.lock'
        with path.open('a') as f:
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as exc:raise StrataError('numeric job already running in another process') from exc
            try:yield
            finally:fcntl.flock(f,fcntl.LOCK_UN)

    def _promote(self,callback):
        state=self.status();latest=state['latest'];policy=self.policy
        if not latest or not policy.get('readiness_rules') or state['status'] in {'promoted','stopped_by_rule'}:return state
        if not _rules_match(policy['readiness_rules'],latest['metrics'],latest['stage'],require_all=True):return state
        callback.identity_hash()
        try:
            line=callback(NumericCheckpoint(self.id,self.id,latest['stage'],latest['checkpoint'],
                self.store.read_blob(latest['checkpoint']),latest['metrics']))
            from .project import Line
            if not isinstance(line,Line):raise StrataError('numeric packager must return a real frozen Line')
            result=certify_numeric_bet(line)
            if result['status']!='verified':raise StrataError('numeric certification lab rejected')
            self.store.put('numeric_job_promotion',dict(job=self.id,checkpoint=latest['checkpoint'],stage=latest['stage'],
                callback_hash=policy['promotion_hash'],**result))
        except Exception as exc:
            self.store.put('numeric_job_promotion',dict(job=self.id,checkpoint=latest['checkpoint'],stage=latest['stage'],
                status='failed',callback_hash=policy['promotion_hash'],error=str(exc)))
            raise StrataError('numeric checkpoint retained; independent certification failed: '+str(exc)) from exc
        return self.status()

    def run(self,*,stages=1,promotion=None):
        if type(stages)is not int or stages<1:raise StrataError('stage count must be positive')
        from . import sandbox
        with self._lock():
            for name,expected_hash in self.policy.get('controller_sources',{}).items():
                if _hash_file(Path(__file__).parent/name)!=expected_hash:
                    raise StrataError('numeric controller implementation changed; create a new job')
            callback=promotion or self._promotion
            expected=self.policy.get('promotion_hash')
            if expected:
                from .callbacks import PinnedCallback
                if not isinstance(callback,PinnedCallback) or callback.identity_hash()!=expected:
                    raise StrataError('resume needs the same pinned numeric packager')
                state=self._promote(callback)
                if state['status']=='promoted':return state
            for _ in range(stages):
                state=self.status()
                if state['status']!='resumable':return state
                policy=self.policy
                for path,expected in policy['dependencies'].items():
                    if _hash_file(path)!=expected:raise StrataError('numeric worker dependency changed; create a new job')
                if _runtime(Path(policy['runtime']['invocation']))!=policy['runtime']:
                    raise StrataError('numeric worker runtime environment changed')
                timeout=min(policy['limits']['timeout'],state['remaining_wall_seconds'])
                attempt=uuid.uuid4().hex;stage=state['completed_stages']+1
                self.store.put('numeric_job_start',dict(job=self.id,attempt=attempt,stage=stage,
                    parent_checkpoint=state['checkpoint'],reserved_seconds=timeout,policy_hash=self.id))
                begun=time.monotonic()
                try:
                    with tempfile.TemporaryDirectory(prefix='numeric-stage-',dir=self.store.path) as directory:
                        work=Path(directory)
                        if state['checkpoint']:(work/'input.checkpoint').write_bytes(self.store.read_blob(state['checkpoint']))
                        base=work/'base.artifact';base.write_bytes(self.store.read_blob(policy['base_artifact']))
                        request=dict(version=1,job=self.id,attempt=attempt,stage=stage,contract=policy['contract'],
                                     settings=policy['settings'],base_artifact=str(base),
                                     checkpoint=str(work/'input.checkpoint') if state['checkpoint'] else None,
                                     metric_units=policy['metric_units'],base_artifact_hash=policy['base_artifact'],
                                     parent_checkpoint_hash=state['checkpoint'])
                        request_path=work/'request.json';request_path.write_text(canonical(request))
                        staged={}
                        for source,expected_hash in policy['dependencies'].items():
                            destination=work/'dependencies'/source.lstrip('/')
                            destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
                            if _hash_file(destination)!=expected_hash:raise StrataError('staged numeric dependency changed')
                            staged[source]=str(destination)
                        argv=[str(request_path) if value=='{request}' else staged.get(value,value) for value in policy['command']]
                        runtime=Path(policy['runtime']['invocation'])
                        runtime_root=runtime.parent.parent if (runtime.parent.parent/'pyvenv.cfg').is_file() else runtime
                        readonly=[str(runtime_root),str(runtime.resolve())]
                        remaining=timeout-(time.monotonic()-begun)
                        if remaining<=0:raise StrataError('numeric stage setup exhausted its reserved timeout')
                        limits=dict(policy['limits'],timeout=remaining)
                        result=sandbox.run(argv,work,**limits,readonly_paths=list(dict.fromkeys(readonly)))
                        if request_path.read_text()!=canonical(request) or _hash_file(base)!=policy['base_artifact']:
                            raise StrataError('worker changed immutable stage inputs')
                        if state['checkpoint'] and _hash_file(work/'input.checkpoint')!=state['checkpoint']:
                            raise StrataError('worker changed its parent checkpoint')
                        for source,destination in staged.items():
                            if _hash_file(destination)!=policy['dependencies'][source] or _hash_file(source)!=policy['dependencies'][source]:
                                raise StrataError('worker dependency changed during stage')
                        if _runtime(runtime)!=policy['runtime']:raise StrataError('worker runtime changed during stage')
                        response=json.loads(result['output'])
                        if response.get('job')!=self.id or response.get('attempt')!=attempt or response.get('stage')!=stage:
                            raise StrataError('numeric checkpoint response scope mismatch')
                        metrics=response.get('metrics')
                        if not isinstance(metrics,dict) or set(metrics)!=set(policy['metric_units']):
                            raise StrataError('numeric checkpoint metric schema changed')
                        for name,value in metrics.items():_finite(value,name)
                        kinds=response.get('metric_kinds',{name:'measured' for name in metrics})
                        if not isinstance(kinds,dict) or set(kinds)!=set(metrics) or any(k not in {'measured','fitted_trend','model','judgment'} for k in kinds.values()):
                            raise StrataError('invalid numeric metric evidence kinds')
                        checkpoint=(work/response['checkpoint_file']).resolve()
                        if not checkpoint.is_relative_to(work) or not checkpoint.is_file():
                            raise StrataError('numeric checkpoint must be a file inside the isolated stage')
                        if checkpoint.stat().st_size>policy['limits']['max_file_bytes']:
                            raise StrataError('numeric checkpoint exceeds file budget')
                        blob=self.store.blob(checkpoint.read_bytes())
                        receipt=self.store.blob(canonical(dict(request=request,response=response,runtime=result)).encode())
                        observed={name:dict(value=value,unit=policy['metric_units'][name],evidence_kind=kinds[name],
                                           artifact_hash=receipt,scope='reported by pinned offline worker; not host acceptance') for name,value in metrics.items()}
                        triggers=_rules_match(policy['stop_rules'],observed,stage)
                        elapsed=time.monotonic()-begun
                        if elapsed>state['remaining_wall_seconds']:raise StrataError('numeric wall budget exhausted before checkpoint commit')
                        _put_atomic(self.store,[('numeric_checkpoint',dict(job=self.id,attempt=attempt,stage=stage,checkpoint=blob,
                            parent_checkpoint=state['checkpoint'],receipt=receipt,metrics=observed,policy_hash=self.id)),
                            ('numeric_job_finish',dict(job=self.id,attempt=attempt,stage=stage,
                            status='stopped_by_rule' if triggers else 'checkpointed',triggers=triggers,
                            seconds=elapsed,receipt=receipt))])
                        if triggers:return self.status()
                except Exception as exc:
                    self.store.put('numeric_job_finish',dict(job=self.id,attempt=attempt,stage=stage,status='failed',
                        error=str(exc),seconds=time.monotonic()-begun,checkpoint_preserved=state['checkpoint']))
                    raise StrataError('numeric stage failed; last complete checkpoint retained: '+str(exc)) from exc
                if callback and self._promote(callback)['status']=='promoted':return self.status()
        return self.status()


def score_bets(current_costs,baseline_costs,bets,*,store=None):
    """Geometric portfolio score and model/judgment-labelled safe-fallback bets."""
    if not current_costs or set(current_costs)!=set(baseline_costs):raise StrataError('portfolio task sets must match')
    def score(costs):
        for value in costs.values():_finite(value,'task cost',True)
        log_score=math.fsum(math.log(_finite(baseline_costs[k],'baseline cost',True))-math.log(v) for k,v in costs.items())/len(costs)
        try:value=math.exp(log_score)
        except OverflowError as exc:raise StrataError('portfolio score exceeds numeric range') from exc
        return _finite(value,'portfolio score',True)
    safe=score(current_costs);rows=[]
    for bet in bets:
        target=dict(current_costs)
        if not set(bet['costs'])<=set(target):raise StrataError('bet changes an unknown task')
        target.update(bet['costs']);success=score(target)
        probability=bet.get('probability');expected=None;kind='unknown'
        if probability is not None:
            if not isinstance(probability,dict) or probability.get('evidence_kind') not in {'measured','fitted_trend','model','judgment'}:
                raise StrataError('bet probability needs an explicit evidence kind')
            p=_finite(probability.get('value'),'probability')
            if not 0<=p<=1:raise StrataError('bet probability must be within zero and one')
            kind=probability['evidence_kind']
            if kind=='measured':
                if store is None:raise StrataError('measured probability requires a stored evidence artifact')
                store.read_blob(probability.get('artifact_hash'))
            expected=safe+p*(max(safe,success)-safe)
        budget=bet.get('remaining_compute_seconds')
        if budget is not None:_finite(budget,'remaining compute budget',True)
        cost_kind=bet.get('cost_evidence_kind','model')
        if cost_kind not in {'measured','fitted_trend','model','judgment'}:raise StrataError('unknown bet cost evidence kind')
        if cost_kind=='measured':
            if store is None:raise StrataError('measured bet costs require a stored evidence artifact')
            store.read_blob(bet.get('cost_artifact_hash'))
        if expected is not None and budget and not math.isfinite((expected-safe)/budget):
            raise StrataError('bet gain rate exceeds numeric range')
        rows.append(dict(id=bet['id'],success_score=success,score_delta=success-safe,
                         probability=probability,expected_score=expected,expected_gain=expected-safe if expected is not None else None,
                         expected_gain_per_second=(expected-safe)/budget if expected is not None and budget else None,
                         evidence_kind=kind,fallback_score=safe,
                         cost_evidence_kind=bet.get('cost_evidence_kind','model'),
                         scope='conditional on supplied success costs/probability; verified safe candidate retained on failure'))
    return dict(safe_score=safe,current_costs=current_costs,bets=rows,formula='geometric_mean(baseline_cost / candidate_cost)',
                safe_verification='caller must supply a retained certified portfolio; this calculation does not certify artifacts')


def certify_numeric_bet(line):
    """Package/build and independently verify through the ordinary frozen Line.

    This writes local artifacts and retained certificates only. It never submits
    a competition entry, publishes a release or bypasses the host verifier.
    """
    lab=line.lab()
    if not lab.passed:return dict(status='lab_rejected',line=line.id,evidence=lab)
    price=line.build();design=line.verify()
    return dict(status='verified',line=line.id,design=design.id,price=price,
                certificate=design.record['certificate'],artifact=design.record['artifact'],
                project_store=str(line.project.store.path),task=line.task.id,
                improves_parent=price['delta']<0,promotion_scope='local certified design; portfolio selection remains separate',
                external_submission=False,gate_sequence=['lab','build','full_verification'])
