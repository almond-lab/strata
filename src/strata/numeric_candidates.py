"""Content-addressed numeric candidates, calibrated scoring and stage pipelines.

The project supplies its mathematics. This module freezes data, arithmetic
callbacks, units, preprocessing and objective so agents compare the same thing.
It never invents a universal chemistry metric or a derivative of integer rounding.
"""
from __future__ import annotations

from dataclasses import dataclass,field
import math
import json
from itertools import islice
import numpy as np

from .callbacks import PinnedCallback
from .cost import Formula
from .model import StrataError,digest,canonical


class NumericCandidate:
    def __init__(self,store,handle):
        self.store=store;self.id=handle
        self.manifest=store.get(handle,'numeric_candidate')

    @classmethod
    def create(cls,store,*,contract,arrays,units,roles=None,trainable=None,
               parameters=None,parent=None,stage='input',actor='unspecified',settings=None):
        if not isinstance(contract,str) or not contract or not isinstance(arrays,dict) or not arrays:
            raise StrataError('numeric candidate needs a contract and named arrays')
        if set(units)!=set(arrays):raise StrataError('every numeric component needs a unit')
        if parent:
            previous=store.get(parent,'numeric_candidate')
            if previous['contract']!=contract:raise StrataError('candidate lineage cannot change the external contract')
        components={}
        for name,raw in sorted(arrays.items()):
            if not isinstance(name,str) or not name or not isinstance(units[name],str) or not units[name]:
                raise StrataError('component names and units must be nonempty strings')
            array=np.asarray(raw)
            if array.dtype.kind not in 'biufc' or not np.all(np.isfinite(array)):
                raise StrataError('candidate arrays must be finite numeric data; object/pickle arrays are forbidden')
            array=np.array(array,dtype=array.dtype.newbyteorder('<'),order='C',copy=True)
            mask=np.asarray((trainable or {}).get(name,np.ones(array.shape,dtype=bool)))
            if mask.dtype.kind!='b' or mask.shape!=array.shape:raise StrataError('trainable mask must be boolean and match component shape')
            components[name]=dict(dtype=array.dtype.str,shape=list(array.shape),unit=units[name],
                role=(roles or {}).get(name,'numeric'),data=store.blob(array.tobytes()),
                trainable=store.blob(np.ascontiguousarray(mask).tobytes()),
                trainable_count=int(np.count_nonzero(mask)),elements=int(array.size))
        body=dict(version=1,contract=contract,components=components,parameters=parameters or {},
                  parent=parent,stage=stage,actor=actor,settings=settings or {})
        return cls(store,store.put('numeric_candidate',body))

    def arrays(self):
        result={}
        manifest=self.store.get(self.id,'numeric_candidate')
        for name,row in manifest['components'].items():
            dtype=np.dtype(row['dtype'])
            if dtype.kind not in 'biufc':raise StrataError('invalid stored numeric dtype')
            data=self.store.read_blob(row['data']);expected=math.prod(row['shape'])*dtype.itemsize
            if len(data)!=expected:raise StrataError('numeric component shape/bytes disagree')
            array=np.frombuffer(data,dtype=dtype).reshape(row['shape'])
            if not np.all(np.isfinite(array)):raise StrataError('numeric component contains nonfinite values')
            result[name]=array
        return result

    def masks(self):
        result={}
        for name,row in self.store.get(self.id,'numeric_candidate')['components'].items():
            data=self.store.read_blob(row['trainable']);elements=math.prod(row['shape'])
            if len(data)!=elements or row['elements']!=elements or any(value not in (0,1) for value in data):
                raise StrataError('numeric trainable mask shape/bytes disagree')
            mask=np.frombuffer(data,dtype=bool).reshape(row['shape'])
            if row['trainable_count']!=int(np.count_nonzero(mask)):
                raise StrataError('numeric trainable count disagrees with stored mask')
            result[name]=mask
        return result

    def save(self):
        self.arrays();self.masks()
        return self.id

    @property
    def artifact_hash(self):
        """Stored manifest bytes binding every component/mask blob and lineage."""
        self.save()
        return self.store.blob(canonical(self.store.get(self.id,'numeric_candidate')).encode())

    @classmethod
    def load(cls,store,handle):
        result=cls(store,handle);result.save();return result

    def push(self,label):
        if not isinstance(label,str) or not label:raise StrataError('handoff label must be nonempty')
        self.save()
        return self.store.put('numeric_handoff',dict(label=label,candidate=self.id,
                              contract=self.store.get(self.id,'numeric_candidate')['contract']))


class NumericScorer:
    """One pinned reference, optionally a calibrated fast path and gradients.

    Callbacks receive {candidate, arrays, parameters}. The scoring response needs
    finite ``metrics``, and may supply boolean ``checks`` and additive
    ``partitions``. Numeric values are priced before they enter the results store.
    Calibration is empirical comparison, never a proof over unseen candidates.
    """
    def __init__(self,store,*,contract,reference,formula,metric_units,cost_unit='cost',fast=None,gradient=None):
        if any(callback is not None and not isinstance(callback,PinnedCallback) for callback in (reference,fast,gradient)):
            raise StrataError('numeric callbacks must be PinnedCallback instances')
        if reference is None or not contract:raise StrataError('numeric scorer needs a pinned reference and contract')
        if not metric_units or any(not isinstance(v,str) or not v for v in metric_units.values()):raise StrataError('metric units are required')
        self.store=store;self.contract=contract;self.reference=reference;self.fast=fast;self.gradient=gradient
        self.formula=Formula(formula);self.metric_units=dict(metric_units);self.cost_unit=cost_unit;self.calibration=None

    def identity(self):
        return digest(dict(contract=self.contract,reference=self.reference.identity_hash(),fast=self.fast.identity_hash() if self.fast else None,
            gradient=self.gradient.identity_hash() if self.gradient else None,formula=self.formula.expression,units=self.metric_units,cost_unit=self.cost_unit))

    def _call(self,candidate,callback):
        candidate=NumericCandidate.load(self.store,candidate.id)
        if candidate.manifest['contract']!=self.contract:raise StrataError('numeric scorer contract mismatch')
        answer=callback(dict(candidate=candidate.id,arrays=candidate.arrays(),parameters=candidate.manifest['parameters']))
        if not isinstance(answer,dict) or not isinstance(answer.get('metrics'),dict) or set(answer['metrics'])!=set(self.metric_units):
            raise StrataError('numeric scorer metric schema mismatch')
        metrics={key:float(value) for key,value in answer['metrics'].items() if type(value)in(int,float) or isinstance(value,(np.integer,np.floating))}
        if set(metrics)!=set(self.metric_units) or any(not math.isfinite(v) for v in metrics.values()):raise StrataError('nonfinite or invalid numeric metrics')
        checks=answer.get('checks',{})
        if not isinstance(checks,dict) or any(type(value)is not bool for value in checks.values()):raise StrataError('scorer checks must be explicit booleans')
        partitions=answer.get('partitions',{})
        if not isinstance(partitions,dict):raise StrataError('numeric attributions must be a mapping')
        for name,partition in partitions.items():
            if not isinstance(partition,dict):raise StrataError('numeric attribution must be a mapping')
            metric=partition.get('metric');terms=partition.get('terms',{})
            if metric not in metrics or partition.get('unit')!=self.metric_units[metric] or not isinstance(terms,dict) or not terms:
                raise StrataError('numeric attribution must name a metric, terms and matching unit')
            if any(type(v)not in(int,float) or not math.isfinite(v) for v in terms.values()):raise StrataError('invalid attribution term')
            if not math.isclose(math.fsum(terms.values()),metrics[metric],rel_tol=1e-10,abs_tol=1e-12):
                raise StrataError('numeric attribution terms do not sum to the reported metric')
        return dict(metrics=metrics,checks=checks,partitions=partitions,cost=self.formula(metrics))

    def calibrate(self,candidates,*,atol=0.,rtol=0.,cost_atol=0.,cost_rtol=0.):
        if self.fast is None:raise StrataError('no fast scorer configured')
        if any(type(v)not in(int,float) or not math.isfinite(v) or v<0 for v in (atol,rtol,cost_atol,cost_rtol)):
            raise StrataError('calibration tolerances must be finite and nonnegative')
        rows=[];identity=self.identity()
        for candidate in candidates:
            reference=self._call(candidate,self.reference);fast=self._call(candidate,self.fast)
            matched=reference['checks']==fast['checks'] and all(math.isclose(reference['metrics'][k],fast['metrics'][k],abs_tol=atol,rel_tol=rtol) for k in self.metric_units)
            matched=matched and math.isclose(reference['cost'],fast['cost'],abs_tol=cost_atol,rel_tol=cost_rtol)
            # Attribution is part of the scorer contract, not optional fast-path decoration.
            if reference['partitions'].keys()!=fast['partitions'].keys():matched=False
            else:
                for key,part in reference['partitions'].items():
                    other=fast['partitions'][key]
                    if part['metric']!=other['metric'] or part['terms'].keys()!=other['terms'].keys():matched=False
                    elif not all(math.isclose(value,other['terms'][name],abs_tol=atol,rel_tol=rtol) for name,value in part['terms'].items()):matched=False
            rows.append(dict(candidate=candidate.id,reference=reference,fast=fast,passed=matched))
        if not rows:raise StrataError('calibration requires actual candidates')
        report=dict(scorer=identity,passed=all(row['passed'] for row in rows),rows=rows,atol=atol,rtol=rtol,
                    cost_atol=cost_atol,cost_rtol=cost_rtol,
                    evidence_kind='measured',scope='empirical scorer agreement on listed artifacts only; final host verification still required')
        self.calibration=self.store.put('numeric_scorer_calibration',report)
        return dict(record=self.calibration,**report)

    def assess(self,candidate,*,fast=False):
        candidate=NumericCandidate.load(self.store,candidate.id)
        identity=self.identity()
        if fast:
            if not self.calibration:raise StrataError('fast scoring requires a passing calibration')
            calibrated=self.store.get(self.calibration,'numeric_scorer_calibration')
            if not calibrated['passed'] or calibrated['scorer']!=identity:raise StrataError('fast scorer calibration is failed or stale')
        result=self._call(candidate,self.fast if fast else self.reference)
        receipt=dict(candidate=candidate.id,artifact_hash=candidate.artifact_hash,
                     contract=self.contract,scorer=identity,formula=self.formula.expression,
                     cost_unit=self.cost_unit,metric_units=self.metric_units,**result,
                     evidence_kind='model' if fast else 'measured',arithmetic='provided pinned callback arithmetic',
                     calibration=self.calibration if fast else None,
                     eligible=bool(result['checks']) and all(result['checks'].values()),
                     frozen_parameters={name:dict(total=row['elements'],trainable=row['trainable_count'],
                         frozen=row['elements']-row['trainable_count']) for name,row in candidate.manifest['components'].items()},
                     scope='priced numeric reference result; independent full artifact verification remains required')
        receipt['record']=self.store.put('numeric_score',receipt)
        return receipt

    def gradients(self,candidate,*,metric):
        candidate=NumericCandidate.load(self.store,candidate.id)
        if candidate.manifest['contract']!=self.contract:raise StrataError('numeric scorer contract mismatch')
        if self.gradient is None or metric not in self.metric_units:raise StrataError('no gradient callback for requested metric')
        result=self.gradient(dict(candidate=candidate.id,arrays=candidate.arrays(),parameters=candidate.manifest['parameters'],metric=metric))
        if not isinstance(result,dict) or result.get('kind') not in {'continuous','surrogate'} or not isinstance(result.get('gradients'),dict):
            raise StrataError('gradient callback must label continuous or surrogate derivatives')
        arrays=candidate.arrays();checked={}
        for name,raw in result['gradients'].items():
            grad=np.asarray(raw)
            if name not in arrays or grad.shape!=arrays[name].shape or grad.dtype.kind not in 'fiu' or not np.all(np.isfinite(grad)):
                raise StrataError('gradient shape or values invalid')
            if arrays[name].dtype.kind in 'biu' and result['kind']!='surrogate':
                raise StrataError('discrete words have no continuous derivative; use labelled surrogate or finite moves')
            checked[name]=grad.tolist()
        return dict(candidate=candidate.id,artifact_hash=candidate.artifact_hash,
                    scorer=self.identity(),metric=metric,kind=result['kind'],gradients=checked,
                    evidence_kind='model',scope='plugin-supplied derivative; not an acceptance proof or automatically validated gradient')

    def _probe(self,candidate,arrays,stage,settings):
        manifest=candidate.manifest
        return NumericCandidate.create(self.store,contract=manifest['contract'],arrays=arrays,
            units={k:v['unit'] for k,v in manifest['components'].items()},
            roles={k:v['role'] for k,v in manifest['components'].items()},trainable=candidate.masks(),
            parameters=manifest['parameters'],parent=candidate.id,stage=stage,settings=settings,actor='numeric_scorer')

    def check_gradients(self,candidate,*,metric,step=1e-5,atol=1e-6,rtol=1e-4,max_evaluations=16):
        """Bounded central-difference consistency on trainable real coordinates."""
        if any(type(v)not in(int,float) or not math.isfinite(v) or v<0 for v in (step,atol,rtol)) or step==0:
            raise StrataError('gradient-check step/tolerances must be finite and valid')
        if type(max_evaluations)is not int or not 2<=max_evaluations<=16:
            raise StrataError('gradient check permits 2 through 16 reference evaluations')
        candidate=NumericCandidate.load(self.store,candidate.id);gradient=self.gradients(candidate,metric=metric)
        if gradient['kind']!='continuous':raise StrataError('surrogate gradients require a declared discrete model, not continuous validation')
        arrays=candidate.arrays();masks=candidate.masks();coordinates=[]
        for name,values in gradient['gradients'].items():
            if arrays[name].dtype.kind!='f':raise StrataError('central differences support real floating parameters only')
            coordinates.extend((name,index) for index in np.ndindex(arrays[name].shape) if masks[name][index])
        if not coordinates:raise StrataError('gradient check has no trainable continuous coordinates')
        rows=[];evaluations=0
        for name,index in coordinates:
            if evaluations+2>max_evaluations:break
            plus={k:np.array(v,copy=True) for k,v in arrays.items()};minus={k:np.array(v,copy=True) for k,v in arrays.items()}
            plus[name][index]+=step;minus[name][index]-=step
            distance=float(plus[name][index]-minus[name][index])
            if distance==0:raise StrataError('gradient step is not representable in the parameter dtype')
            a=self._probe(candidate,plus,'gradient_probe',dict(component=name,index=list(index),direction=1,step=step))
            b=self._probe(candidate,minus,'gradient_probe',dict(component=name,index=list(index),direction=-1,step=step))
            ap=self.assess(a);bp=self.assess(b);evaluations+=2
            finite=(ap['metrics'][metric]-bp['metrics'][metric])/distance
            analytic=float(np.asarray(gradient['gradients'][name])[index])
            valid=ap['eligible'] and bp['eligible']
            agrees=math.isclose(analytic,finite,abs_tol=atol,rel_tol=rtol)
            rows.append(dict(component=name,index=list(index),analytic=analytic,finite_difference=finite,
                passed=agrees,derivative_agrees=agrees,probe_eligible=valid,
                plus_eligible=ap['eligible'],minus_eligible=bp['eligible'],
                plus_candidate=a.id,minus_candidate=b.id,plus_score=ap['record'],minus_score=bp['record']))
        complete=len(rows)==len(coordinates)
        report=dict(candidate=candidate.id,scorer=self.identity(),metric=metric,rows=rows,
            status='bounded' if not complete else 'consistent_on_probes' if all(r['passed'] for r in rows) else 'inconsistent_on_probes',
            passed=all(r['passed'] for r in rows) if complete else None,evaluations=evaluations,total_coordinates=len(coordinates),
            probes_eligible=all(r['probe_eligible'] for r in rows),
            step=step,atol=atol,rtol=rtol,evidence_kind='model',scope='local finite-difference consistency only; probe eligibility is reported separately; no global derivative or host acceptance theorem')
        report['record']=self.store.put('numeric_gradient_check',report)
        return report

    def discrete_moves(self,candidate,moves,*,max_evaluations=16):
        """Price listed exact integer/boolean coordinate moves with the reference."""
        if type(max_evaluations)is not int or not 2<=max_evaluations<=16:
            raise StrataError('discrete moves permit 2 through 16 reference evaluations including baseline')
        candidate=NumericCandidate.load(self.store,candidate.id);baseline=self.assess(candidate)
        arrays=candidate.arrays();masks=candidate.masks();rows=[];moves=list(islice(iter(moves),max_evaluations))
        for move in moves[:max_evaluations-1]:
            if not isinstance(move,dict) or set(move)!={'component','index','value'}:raise StrataError('discrete move needs component, index and value')
            name=move['component'];index=tuple(move['index']);value=move['value']
            if name not in arrays or arrays[name].dtype.kind not in 'biu':raise StrataError('exact discrete moves require integer or boolean components')
            if len(index)!=arrays[name].ndim or any(type(i)is not int or not 0<=i<size for i,size in zip(index,arrays[name].shape)):
                raise StrataError('discrete coordinate outside component')
            if not masks[name][index]:raise StrataError('discrete move changes a frozen parameter')
            if type(value)not in(int,bool):raise StrataError('discrete value must be an exact integer or boolean')
            dtype=arrays[name].dtype
            if dtype.kind=='b' and value not in (0,1) or dtype.kind!='b' and not np.iinfo(dtype).min<=value<=np.iinfo(dtype).max:
                raise StrataError('discrete value outside component dtype range')
            child_arrays={k:np.array(v,copy=True) for k,v in arrays.items()};child_arrays[name][index]=value
            child=self._probe(candidate,child_arrays,'discrete_move',move);score=self.assess(child)
            rows.append(dict(move=move,candidate=child.id,score=score['record'],cost=score['cost'],
                             cost_delta=score['cost']-baseline['cost'],eligible=score['eligible']))
        return dict(candidate=candidate.id,baseline_score=baseline['record'],moves=rows,evaluations=1+len(rows),
            status='bounded' if len(rows)<len(moves) else 'complete_for_listed_moves',evidence_kind='measured',
            scope='listed exact reference evaluations; no discrete gradient, global optimum, or host acceptance claim')


@dataclass(frozen=True)
class NumericStage:
    name: str
    callback: PinnedCallback
    settings: dict = field(default_factory=dict)
    invariant_metrics: dict = field(default_factory=dict)


class NumericPipeline:
    """Compose order/picker/refit/repair/normalization plugins under one policy."""
    def __init__(self,store,scorer,stages):
        if not stages or any(not isinstance(s,NumericStage) or not isinstance(s.callback,PinnedCallback) for s in stages):
            raise StrataError('numeric pipeline needs pinned named stages')
        if len({s.name for s in stages})!=len(stages):raise StrataError('numeric stage names must be unique')
        for stage in stages:
            if not isinstance(stage.name,str) or not stage.name or not isinstance(stage.settings,dict) or not isinstance(stage.invariant_metrics,dict):
                raise StrataError('numeric stage needs a name, settings and invariant mappings')
            for metric,tolerance in stage.invariant_metrics.items():
                if metric not in scorer.metric_units or type(tolerance)not in(int,float) or not math.isfinite(tolerance) or tolerance<0:
                    raise StrataError('invalid numeric stage invariant')
        self.store=store;self.scorer=scorer;self.stages=tuple(stages)
        self.policy=dict(scorer=scorer.identity(),stages=[dict(name=s.name,source=s.callback.identity_hash(),settings=s.settings,invariants=s.invariant_metrics) for s in stages])
        self.id=store.put('numeric_pipeline',self.policy)
        self.policy=store.get(self.id,'numeric_pipeline')

    def run(self,candidate,*,resume=None):
        self.policy=self.store.get(self.id,'numeric_pipeline')
        if self.scorer.identity()!=self.policy['scorer']:raise StrataError('pipeline scorer changed')
        if len(self.stages)!=len(self.policy['stages']):raise StrataError('pipeline stage policy changed')
        candidate=NumericCandidate.load(self.store,candidate.id)
        original=candidate.id;start=0;records=[]
        if resume:
            saved=self.store.get(resume,'numeric_pipeline_checkpoint')
            if saved['pipeline']!=self.id or saved['original']!=original:raise StrataError('pipeline checkpoint scope mismatch')
            if (type(saved['next_stage'])is not int or not 1<=saved['next_stage']<=len(self.stages) or
                    saved['stage']!=self.policy['stages'][saved['next_stage']-1]['name']):
                raise StrataError('pipeline checkpoint stage mismatch')
            candidate=NumericCandidate.load(self.store,saved['candidate']);start=saved['next_stage']
        before=self.scorer.assess(candidate)
        for index,stage in enumerate(self.stages[start:],start=start):
            frozen=self.policy['stages'][index]
            if stage.name!=frozen['name'] or stage.callback.identity_hash()!=frozen['source'] or digest(stage.settings)!=digest(frozen['settings']) or digest(stage.invariant_metrics)!=digest(frozen['invariants']):
                raise StrataError('pipeline stage policy changed')
            old_arrays=candidate.arrays();old_masks=candidate.masks()
            try:
                child=stage.callback(dict(candidate=NumericCandidate.load(self.store,candidate.id),
                    settings=json.loads(canonical(frozen['settings'])),score=json.loads(canonical(before))))
                if not isinstance(child,NumericCandidate) or child.store.path!=self.store.path:
                    raise StrataError('numeric stage must return a candidate in the same store')
                child=NumericCandidate.load(self.store,child.id)
                if child.id!=candidate.id and child.manifest['parent']!=candidate.id:raise StrataError('numeric stage must preserve parent lineage')
                arrays=child.arrays()
                child_masks=child.masks()
                for name,old in old_arrays.items():
                    frozen_mask=~old_masks[name]
                    if np.any(frozen_mask) and (name not in arrays or arrays[name].shape!=old.shape or not np.array_equal(arrays[name][frozen_mask],old[frozen_mask])):
                        raise StrataError('numeric stage modified a frozen parameter')
                    if np.any(frozen_mask) and np.any(child_masks[name][frozen_mask]):
                        raise StrataError('numeric stage made a frozen parameter trainable')
                after=self.scorer.assess(child)
                for metric,tolerance in stage.invariant_metrics.items():
                    if metric not in before['metrics'] or type(tolerance)not in(int,float) or not math.isfinite(tolerance) or tolerance<0:
                        raise StrataError('invalid numeric stage invariant')
                    if abs(after['metrics'][metric]-before['metrics'][metric])>tolerance:
                        raise StrataError('numeric stage broke declared invariant: '+metric)
                receipt=dict(pipeline=self.id,original=original,parent=candidate.id,candidate=child.id,next_stage=index+1,
                             stage=stage.name,before_score=before['record'],after_score=after['record'],cost_delta=after['cost']-before['cost'],
                             stage_authority='pinned trusted numeric plugin; independent full verifier still required')
                handle=self.store.put('numeric_pipeline_checkpoint',receipt);records.append(handle)
                candidate=child;before=after
            except Exception as exc:
                failure=self.store.put('numeric_pipeline_failure',dict(pipeline=self.id,original=original,candidate=candidate.id,stage=stage.name,
                    error=str(exc),last_checkpoint=records[-1] if records else resume))
                raise StrataError('numeric pipeline failed; checkpoint retained '+failure+': '+str(exc)) from exc
        return dict(pipeline=self.id,candidate=candidate.id,score=before,checkpoints=records,
                    completed_stages=len(self.stages),headline='cost after the entire declared preparation pipeline',
                    external_submission=False)
