"""Generic numeric evidence, budgets, experiments, and priced design spaces.

Caller code supplies the mathematics, resource evaluator, and pinned verifier.
A model prediction is never upgraded to a measurement by assigning it a score.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from itertools import product, combinations as choose
import json
import math

from .cost import Formula, ContextFunction
from .model import StrataError, canonical, digest


EVIDENCE_KINDS = {'measured', 'fitted_trend', 'model', 'judgment'}


@dataclass(frozen=True)
class EvidenceValue:
    value: float
    unit: str
    evidence_kind: str
    artifact_hash: str | None = None
    source_hash: str | None = None
    scope: str | None = None
    context: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ResourceReceipt:
    """Trusted verifier output; its exact payload must exist as a stored blob.

    This type does not launch a verifier. The trusted caller must obtain the
    receipt from its pinned runner. Validation checks byte/context bindings and
    cannot attest that arbitrary caller-supplied Python code is trustworthy.
    """
    resources: dict
    artifact_hash: str
    verifier_hash: str
    context_hash: str
    settings_hash: str
    passed: bool
    receipt_hash: str


def _real(value, name, *, nonnegative=False):
    if type(value) not in (int, float) or not math.isfinite(value) or nonnegative and value < 0:
        raise StrataError(name + ' must be a finite ' + ('nonnegative ' if nonnegative else '') + 'real number')
    return value


def _hash(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise StrataError(name + ' must be a SHA-256 hash')
    return value


def _blob(store, value, name):
    _hash(value, name)
    if store is None:
        raise StrataError(name + ' requires a Store to check its artifact')
    try:
        return store.read_blob(value)
    except (OSError, StrataError) as exc:
        raise StrataError(name + ' artifact is missing or invalid') from exc


def _mapping(value, name):
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise StrataError(name + ' must be a mapping with string keys')
    try:
        return json.loads(canonical(value))
    except (TypeError, ValueError) as exc:
        raise StrataError(name + ' must contain canonical JSON values') from exc


def evidence_value(value, *, store=None):
    """Validate an evidence value without changing its claimed evidence tier."""
    row = asdict(value) if isinstance(value, EvidenceValue) else dict(value) if isinstance(value, dict) else None
    if row is None or set(row) - {'value','unit','evidence_kind','artifact_hash','source_hash','scope','context'}:
        raise StrataError('invalid numeric evidence fields')
    _real(row.get('value'), 'evidence value')
    if not isinstance(row.get('unit'), str) or not row['unit'].strip():
        raise StrataError('numeric evidence needs an explicit unit')
    if row.get('evidence_kind') not in EVIDENCE_KINDS:
        raise StrataError('numeric evidence kind must be measured, fitted_trend, model, or judgment')
    if not row.get('source_hash') and not row.get('artifact_hash'):
        raise StrataError('numeric evidence needs hashed provenance')
    for name in ('source_hash','artifact_hash'):
        if row.get(name):
            _hash(row[name], name)
    if row['evidence_kind'] == 'measured':
        _blob(store, row.get('artifact_hash'), 'measured evidence')
    row.setdefault('context', {})
    row['context'] = _mapping(row['context'], 'evidence context')
    return row


def error_budget(terms, *, unit, limit, groups=None, roots=None, store=None):
    """Add explicitly distinct error leaves once, preserving signed cross terms.

    A group is a list of leaf/group IDs, not another numeric subtotal. Overlapping
    views share the same leaf and never add it twice. Units are exact caller
    labels: conversions require an explicit separately evidenced input value.
    """
    _real(limit, 'error limit', nonnegative=True)
    if not isinstance(unit, str) or not unit.strip():
        raise StrataError('error budget needs an explicit unit')
    leaves = {}
    for term in terms:
        if not isinstance(term, dict) or not isinstance(term.get('id'),str) or not term['id'] or term['id'] in leaves:
            raise StrataError('error leaves require unique nonempty IDs')
        if set(term) - {'id','evidence','role','labels','cross_of'}:
            raise StrataError('unknown error-leaf fields')
        ev = evidence_value(term.get('evidence'), store=store)
        if ev['unit'] != unit:
            raise StrataError('error budget units do not match')
        role = term.get('role','component')
        if role not in {'component','cross','rounding'}:
            raise StrataError('unknown error-leaf role')
        if role != 'cross' and ev['value'] < 0:
            raise StrataError('only explicitly declared cross terms may be signed')
        references = term.get('cross_of', [])
        if role == 'cross' and (not isinstance(references,list) or len(references)<2 or
                any(not isinstance(ref,str) for ref in references) or len(set(references))!=len(references)):
            raise StrataError('a cross term must identify at least two distinct component leaves')
        if role != 'cross' and references:
            raise StrataError('cross_of is reserved for explicit cross terms')
        leaves[term['id']] = dict(id=term['id'], evidence=ev, role=role,
                                  labels=_mapping(term.get('labels',{}),'error labels'), cross_of=references)
    for leaf in leaves.values():
        if any(ref not in leaves or leaves[ref]['role']=='cross' for ref in leaf['cross_of']):
            raise StrataError('cross term references an unknown or cross-term component')
    groups = _mapping(groups or {}, 'error groups')
    if set(groups) & set(leaves):
        raise StrataError('group and leaf IDs must be distinct')
    for members in groups.values():
        if not isinstance(members,list) or any(not isinstance(item,str) for item in members):
            raise StrataError('groups must list leaf or group IDs')
    def expand(name, active):
        if name in leaves:
            return {name}
        if name not in groups:
            raise StrataError('unknown error-budget reference: '+str(name))
        if name in active:
            raise StrataError('cyclic error-budget hierarchy')
        return set().union(*(expand(child, active | {name}) for child in groups[name])) if groups[name] else set()
    expanded = {name: expand(name,set()) for name in groups}
    roots = list(leaves) if roots is None else list(roots)
    if any(not isinstance(name,str) for name in roots):
        raise StrataError('error-budget roots must be leaf or group IDs')
    selected = set().union(*(expand(name,set()) for name in roots)) if roots else set()
    chosen = [leaves[name] for name in sorted(selected)]
    total = math.fsum(leaf['evidence']['value'] for leaf in chosen)
    if total < 0:
        raise StrataError('signed error budget has a negative total; no nonnegative error bound is established')
    without_rounding = math.fsum(leaf['evidence']['value'] for leaf in chosen if leaf['role']!='rounding')
    by_kind = {kind: math.fsum(leaf['evidence']['value'] for leaf in chosen if leaf['evidence']['evidence_kind']==kind)
               for kind in sorted(EVIDENCE_KINDS)}
    result = dict(version=1, unit=unit, limit=limit, total=total, remaining=limit-total,
                  within_budget=total<=limit, allowed_rounding_budget=max(0,limit-without_rounding),
                  headroom_scope='conditional on recorded nonrounding and cross terms; new rounding may change cross terms; not guaranteed feasibility',
                  nonrounding_over_budget=without_rounding>limit, leaves=chosen,
                  groups={name:dict(leaves=sorted(ids),total=math.fsum(leaves[i]['evidence']['value'] for i in ids))
                          for name,ids in expanded.items()}, roots=roots, counted_leaf_ids=sorted(selected),
                  evidence_totals=by_kind, count_rule='each selected leaf ID exactly once, even across overlapping groups',
                  scope='caller supplied additive error model including explicit signed cross terms; not an independently proved error theorem')
    result['ledger_hash']=digest(result)
    return result


def _validate_receipt(receipt, *, store, verifier_hash, context, settings, artifact_hashes=None):
    if not isinstance(receipt, ResourceReceipt):
        raise StrataError('measured resources require a typed pinned verifier receipt')
    expected = _hash(verifier_hash, 'pinned verifier hash')
    if receipt.verifier_hash != expected:
        raise StrataError('numeric verifier identity does not match pinned source')
    _blob(store, expected, 'verifier source')
    _blob(store, receipt.artifact_hash, 'verified numeric artifact')
    if artifact_hashes is not None and receipt.artifact_hash not in artifact_hashes:
        raise StrataError('verifier receipt is bound to a different experiment artifact')
    if receipt.context_hash != digest(context) or receipt.settings_hash != digest(settings):
        raise StrataError('numeric receipt context/settings do not match this experiment')
    if type(receipt.passed) is not bool:
        raise StrataError('numeric verifier result must be a boolean')
    resources=_mapping(receipt.resources,'numeric resources')
    for name,value in resources.items():
        _real(value,'resource '+name,nonnegative=True)
    payload=asdict(receipt);payload.pop('receipt_hash')
    try:
        saved=json.loads(_blob(store,receipt.receipt_hash,'numeric verifier receipt'))
    except (ValueError,UnicodeError) as exc:
        raise StrataError('numeric verifier receipt is not valid JSON') from exc
    if canonical(saved)!=canonical(payload):
        raise StrataError('numeric verifier receipt payload differs from pinned bytes')
    return resources


class NumericStudy:
    """Immutable shared experiment records; every query rereads the shared Store."""
    def __init__(self,store):
        self.store=store

    def record(self, *, contract, base, artifacts, settings, factors, line,
               observations, siblings=(), baseline=None, verifier_receipt=None, verifier_hash=None):
        if not isinstance(contract,str) or not contract or not isinstance(line,str) or not line:
            raise StrataError('numeric experiments require a contract and line identity')
        _blob(self.store,base,'experiment base')
        artifacts=_mapping(artifacts,'experiment artifacts')
        if not artifacts:
            raise StrataError('numeric experiment must bind an artifact')
        for value in artifacts.values():
            _blob(self.store,value,'experiment artifact')
        settings=_mapping(settings,'experiment settings');factors=_mapping(factors,'experiment factors')
        if not isinstance(observations,dict) or not observations:
            raise StrataError('numeric experiment needs named observations')
        observed={key:evidence_value(value,store=self.store) for key,value in observations.items()}
        if any(not isinstance(key,str) for key in observed):
            raise StrataError('observation names must be strings')
        if any(ev.get('artifact_hash') not in artifacts.values() for ev in observed.values() if ev['evidence_kind']=='measured'):
            raise StrataError('measured observation is not bound to this experiment artifact')
        context=dict(contract=contract,base=base,factors=factors)
        receipt=None
        if verifier_receipt is not None:
            resources=_validate_receipt(verifier_receipt,store=self.store,verifier_hash=verifier_hash,
                                        context=context,settings=settings,artifact_hashes=set(artifacts.values()))
            for key,ev in observed.items():
                if ev['evidence_kind']=='measured' and (ev.get('artifact_hash')!=verifier_receipt.artifact_hash or
                        key not in resources or resources[key]!=ev['value']):
                    raise StrataError('measured observation differs from verifier resources')
            receipt=asdict(verifier_receipt)
        if baseline is not None:
            other=self.get(baseline)
            if other['contract']!=contract or other['factors']!=factors or other['base']!=base:
                raise StrataError('numeric baseline contract, base, or factors differ')
        if any(not isinstance(s,str) or not s for s in siblings):
            raise StrataError('sibling lines must be nonempty identities')
        return self.store.put('numeric_experiment',dict(version=1,contract=contract,base=base,artifacts=artifacts,
            settings=settings,factors=factors,line=line,siblings=sorted(set(siblings)),observations=observed,
            baseline=baseline,verifier_receipt=receipt,verifier_hash=verifier_hash,
            context_hash=digest(context),scope='one recorded experiment; transfer to another line is a hypothesis'))

    def get(self,handle):
        row=self.store.get(handle,'numeric_experiment')
        for value in [row['base'],*row['artifacts'].values()]:
            _blob(self.store,value,'experiment provenance')
        for ev in row['observations'].values():
            evidence_value(ev,store=self.store)
            if ev['evidence_kind']=='measured' and ev.get('artifact_hash') not in row['artifacts'].values():
                raise StrataError('measured observation is not bound to experiment artifacts')
        context=dict(contract=row['contract'],base=row['base'],factors=row['factors'])
        if row['context_hash']!=digest(context):
            raise StrataError('numeric experiment context hash mismatch')
        if row.get('verifier_receipt'):
            resources=_validate_receipt(ResourceReceipt(**row['verifier_receipt']),store=self.store,verifier_hash=row['verifier_hash'],
                context=context,settings=row['settings'],
                artifact_hashes=set(row['artifacts'].values()))
            for key,ev in row['observations'].items():
                if ev['evidence_kind']=='measured' and (ev.get('artifact_hash')!=row['verifier_receipt']['artifact_hash'] or
                        key not in resources or resources[key]!=ev['value']):
                    raise StrataError('stored measured observation differs from verifier resources')
        return row

    def experiments(self, *, contract=None, factors=None, line=None):
        result=[]
        for handle,_ in self.store.all('numeric_experiment'):
            row=self.get(handle)
            if contract is not None and row['contract']!=contract:continue
            if factors is not None and row['factors']!=factors:continue
            if line is not None and row['line']!=line:continue
            result.append(dict(handle=handle,**row))
        return result

    def suggest_transfers(self, *, line, contract, factors, metric='cost', limit=32,
                          transfer_keys=(), include_tested=False):
        if type(limit) is not int or limit<1:raise StrataError('transfer limit must be positive')
        factors=_mapping(factors,'target factors')
        allowed=set(transfer_keys)
        if any(not isinstance(key,str) or key not in factors for key in allowed):
            raise StrataError('transfer_keys must identify explicit target factor names')
        rows=[row for row in self.experiments(contract=contract)
              if set(row['factors'])==set(factors) and
              all(key in allowed or value==factors[key] for key,value in row['factors'].items())]
        target=[row for row in rows if row['line']==line and row['factors']==factors]
        suggestions=[];seen=set()
        for row in rows:
            if row['line']==line or row['siblings'] and line not in row['siblings']:continue
            if not row['baseline'] or not row['verifier_receipt'] or not row['verifier_receipt']['passed']:continue
            before=self.get(row['baseline']);ev=row['observations'].get(metric);old=before['observations'].get(metric)
            if any(before[key]!=row[key] for key in ('contract','base','factors')):
                raise StrataError('stored numeric baseline contract, base, or factors differ')
            if (not before['verifier_receipt'] or not before['verifier_receipt']['passed'] or not ev or not old or
                ev['evidence_kind']!='measured' or old['evidence_kind']!='measured' or ev['unit']!=old['unit'] or ev['value']>=old['value']):continue
            delta={key:value for key,value in row['settings'].items() if before['settings'].get(key)!=value or key not in before['settings']}
            tested=any(all(t['settings'].get(k)==v and k in t['settings'] for k,v in delta.items()) for t in target)
            if not delta or tested and not include_tested:continue
            identity=digest(delta)
            if identity in seen:continue
            seen.add(identity)
            suggestions.append(dict(settings=delta,source_experiment=row['handle'],source_line=row['line'],target_line=line,
                observed_metric_delta=ev['value']-old['value'],unit=ev['unit'],contract=contract,factors=factors,
                source_factors=row['factors'],transfer_keys=sorted(allowed),already_tested_on_target=tested,
                status='transfer_hypothesis',gain_established_on_target=False,
                reason='verified source win with explicitly compatible context; target combination still needs verification'))
        return dict(suggestions=suggestions[:limit],total_eligible=len(suggestions),truncated=len(suggestions)>limit,
                    scope='observed source wins; changed base/line context still requires new verification')

    def combinations(self, *, line, contract, factors, metric='cost', max_terms=2, max_candidates=32,transfer_keys=()):
        if type(max_terms)is not int or max_terms<1 or type(max_candidates)is not int or max_candidates<1:
            raise StrataError('combination budgets must be positive')
        transfer=self.suggest_transfers(line=line,contract=contract,factors=factors,metric=metric,limit=max_candidates,
                                       transfer_keys=transfer_keys,include_tested=True)
        proposals=transfer['suggestions']
        target=self.experiments(contract=contract,factors=factors,line=line)
        output=[];seen=set();conflicts=0;examined=0
        for size in range(1,min(max_terms,len(proposals))+1):
            for selected in choose(proposals,size):
                if examined>=max_candidates:
                    return dict(candidates=output,examined=examined,conflicts=conflicts,status='bounded',gain_established=False)
                examined+=1;settings={};valid=True
                for row in selected:
                    for key,value in row['settings'].items():
                        if key in settings and settings[key]!=value:valid=False
                        settings[key]=value
                if not valid:conflicts+=1;continue
                if any(all(t['settings'].get(k)==v and k in t['settings'] for k,v in settings.items()) for t in target):continue
                identity=digest(settings)
                if identity in seen:continue
                seen.add(identity);output.append(dict(settings=settings,source_experiments=[r['source_experiment'] for r in selected],status='untested_combination'))
        return dict(candidates=output,examined=examined,conflicts=conflicts,
                    status='bounded' if transfer['truncated'] else 'complete_in_family',gain_established=False)


def design_space(shape_options, *, formula, resource_hook, source_hash, context,
                 store=None, verifier_hash=None, max_points=256, budget_hook=None):
    """Enumerate supplied discrete shapes, keeping exact caller counts and cost.

    Plain resource dictionaries are model results, even if the hook calls them
    measurements. A typed pinned receipt must bind this shape and context before
    a point is reported measured. No interpolation smooths discontinuous widths.
    """
    _hash(source_hash,'resource-hook source hash')
    if store is not None:_blob(store,source_hash,'resource-hook source')
    if not callable(resource_hook):raise StrataError('resource hook must be callable trusted code')
    if type(max_points)is not int or max_points<1:raise StrataError('design-space budget must be positive')
    options=_mapping(shape_options,'shape options');context=_mapping(context,'design-space context')
    if any(not isinstance(values,list) or not values for values in options.values()):
        raise StrataError('each shape option needs a nonempty explicit list')
    count=math.prod(len(values) for values in options.values())
    objective=formula if isinstance(formula,Formula) else Formula(formula)
    rows=[]
    for values in product(*options.values()):
        if len(rows)>=max_points:break
        settings=_mapping(dict(zip(options,values)),'design settings')
        returned=resource_hook(_mapping(settings,'hook settings'),_mapping(context,'hook context'))
        measured=isinstance(returned,ResourceReceipt)
        if measured:
            resources=_validate_receipt(returned,store=store,verifier_hash=verifier_hash,context=context,settings=settings)
            if not returned.passed:
                rows.append(dict(settings=settings,status='verifier_rejected',resources=resources,evidence_kind='measured',cost=None,receipt_hash=returned.receipt_hash))
                continue
        else:
            resources=_mapping(returned,'model resources')
            for name,value in resources.items():_real(value,'resource '+name,nonnegative=True)
        cost=objective(resources)
        budget=None if budget_hook is None else budget_hook(_mapping(settings,'budget settings'),_mapping(context,'budget context'))
        if budget is not None:
            if not isinstance(budget,dict) or 'ledger_hash' not in budget:raise StrataError('budget hook must return an error-budget ledger')
            unsigned={key:value for key,value in budget.items() if key!='ledger_hash'}
            if digest(unsigned)!=budget['ledger_hash']:raise StrataError('design-space error ledger hash mismatch')
        rows.append(dict(settings=settings,status='evaluated',resources=resources,cost=cost,
                         evidence_kind='measured' if measured else 'model',source_hash=source_hash,
                         artifact_hash=returned.artifact_hash if measured else None,
                         receipt_hash=returned.receipt_hash if measured else None,error_budget=budget,
                         within_error_budget=None if budget is None else budget['within_budget']))
    eligible=[row for row in rows if row.get('cost') is not None and row.get('within_error_budget') is not False]
    result=dict(version=1,formula=objective.expression,context=context,context_hash=digest(context),source_hash=source_hash,
                points=rows,total_points=count,evaluated_points=len(rows),status='bounded' if len(rows)<count else 'complete',
                best=None if not eligible else min(eligible,key=lambda r:r['cost']),
                scope='enumerated caller shapes and formula; model points do not establish host acceptance or gains')
    result['map_hash']=digest(result)
    return result


def break_even_scale(formula, values, *, scale_parameter, target, linear_scale=False):
    """Solve an explicitly declared affine scale exactly in the caller formula.

    A structural degree check rejects nonlinear dependence; sampling a formula
    at a few values would not prove linearity. Other resource values stay fixed.
    """
    if linear_scale is not True:raise StrataError('break-even requires an explicit linear-scale declaration')
    _real(target,'target cost',nonnegative=True)
    objective=formula if isinstance(formula,Formula) else Formula(formula)
    if not isinstance(scale_parameter,str) or not scale_parameter.isidentifier():raise StrataError('invalid scale parameter')
    def degree(node):
        if isinstance(node,ast.Name):return int(node.id==scale_parameter)
        if isinstance(node,ast.Constant):return 0
        if isinstance(node,ast.UnaryOp):return degree(node.operand)
        if isinstance(node,ast.Call):
            if isinstance(node.func,ast.Name) and isinstance(objective.functions.get(node.func.id),ContextFunction):
                raise StrataError('context-aware function prevents a structural linear-scale proof')
            if any(degree(arg) for arg in node.args):raise StrataError('scale appears in a nonlinear function')
            return 0
        if isinstance(node,ast.BinOp):
            a,b=degree(node.left),degree(node.right)
            if isinstance(node.op,(ast.Add,ast.Sub)):return max(a,b)
            if isinstance(node.op,ast.Mult):
                if a+b>1:raise StrataError('scale dependence is nonlinear')
                return a+b
            if isinstance(node.op,ast.Div):
                if b:raise StrataError('scale occurs in a denominator')
                return a
            if isinstance(node.op,ast.Pow):
                if b or a and not(isinstance(node.right,ast.Constant) and node.right.value==1):raise StrataError('scale occurs in a nonlinear power')
                return a
        raise StrataError('unsupported scale expression')
    if degree(objective.tree)!=1:raise StrataError('formula must depend on the declared scale')
    fixed=_mapping(values,'fixed resources');fixed.pop(scale_parameter,None)
    offset=objective(dict(fixed,**{scale_parameter:0}));one=objective(dict(fixed,**{scale_parameter:1}))
    coefficient=one-offset
    if coefficient<=0:raise StrataError('break-even scale must have a strictly positive coefficient')
    threshold=(target-offset)/coefficient
    return dict(status='solved' if threshold>=0 else 'unattainable_with_nonnegative_scale',
                scale_parameter=scale_parameter,break_even=threshold if threshold>=0 else None,
                coefficient=coefficient,offset=offset,target=target,formula=objective.expression,
                fixed_resources=fixed,condition='cost <= target iff 0 <= scale <= break_even' if threshold>=0 else 'fixed offset already exceeds target',
                evidence_kind='model',scope='algebraic threshold with all other caller resources fixed; no measured improvement')
