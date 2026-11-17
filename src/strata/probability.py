"""Precommitted stochastic comparisons; never rank by a candidate's lucky draw."""
from dataclasses import dataclass
from math import isfinite, fsum, log2, log1p, log, isclose
import ast
from .model import StrataError,digest
from .cost import Formula, ContextFunction


def _confidence_log2_delta(bound):
    """Validate a supplied confidence without rounding 1-delta to one."""
    values=[]
    for name in ('confidence','delta','log2_delta'):
        if name not in bound:continue
        value=bound[name]
        if type(value) not in (int,float):
            raise StrataError('invalid verifier-supplied confidence bound')
        try:value=float(value)
        except OverflowError as exc:raise StrataError('invalid verifier-supplied confidence bound') from exc
        if not isfinite(value):raise StrataError('invalid verifier-supplied confidence bound')
        if name=='log2_delta':
            if value>=0:raise StrataError('log2_delta must be finite and negative')
            values.append(float(value))
        else:
            if not 0<value<1:raise StrataError(f'{name} must lie strictly between zero and one; use log2_delta for extreme confidence')
            values.append(log2(value) if name=='delta' else log1p(-value)/log(2))
    if not values or any(not isclose(value,values[0],rel_tol=1e-12,abs_tol=1e-12) for value in values[1:]):
        raise StrataError('missing or inconsistent verifier-supplied confidence metadata')
    return values[0]


@dataclass(frozen=True)
class ProbabilityPolicy:
    seeds: tuple[int,...]
    compare: str = 'expected'
    failure_resource: str = 'fail_rate'
    max_failure: float = 1.0
    penalty: str | None = None
    weights: tuple[float,...] | None = None

    def __post_init__(self):
        if len(self.seeds)<2 or len(set(self.seeds))!=len(self.seeds) or any(type(s)is not int for s in self.seeds):
            raise StrataError('probability policy requires distinct precommitted seeds')
        if self.compare not in {'expected','fixed_seeds','worst'}:raise StrataError('unknown probability comparison mode')
        if not isfinite(self.max_failure) or not 0<=self.max_failure<=1:raise StrataError('invalid failure budget')
        if self.weights is not None and (len(self.weights)!=len(self.seeds) or any(not isfinite(w) or w<=0 for w in self.weights)):
            raise StrataError('positive weight required for every precommitted draw')

    @classmethod
    def from_config(cls,config):
        return cls(seeds=tuple(config['seeds']),compare=config.get('compare','expected'),
                   failure_resource=config.get('failure_resource','fail_rate'),
                   max_failure=float(config.get('max_failure',1)),penalty=config.get('penalty'),
                   weights=tuple(config['weights']) if 'weights' in config else None)

    @property
    def fingerprint(self):return digest(self)

    def cost_functions(self):
        if self.penalty is None:return {}
        penalty=Formula(self.penalty)
        # Preserve the v0.2 R(rate) API while making counts, shots and any
        # verifier-supplied confidence bound available to the penalty.
        return {'R':ContextFunction(lambda values,p:penalty({**values,self.failure_resource:p,'fail_rate':p}))}

    def receipt_schema(self,formula):
        """An offline declaration check; it does not run or certify a verifier."""
        def names(expression):
            parsed=Formula(expression)
            calls={id(node.func) for node in ast.walk(parsed.tree) if isinstance(node,ast.Call)}
            return {node.id for node in ast.walk(parsed.tree) if isinstance(node,ast.Name) and id(node) not in calls}
        required=names(formula)|{self.failure_resource}
        if self.penalty:
            required|={self.failure_resource if name=='fail_rate' else name for name in names(self.penalty)}
        if required & {'fail_count','shots'}:required|={'fail_count','shots'}
        return dict(required_draw_resources=sorted(required),draw_order=list(self.seeds),
            count_resources=dict(fail_count='integer >= 0',shots='positive integer'),
            empirical_identity=f'{self.failure_resource} == fail_count/shots when counts are supplied',
            confidence_representations=['confidence','delta','log2_delta'],
            confidence_scope='verifier-supplied metadata, never a Strata-computed bound')

    def validate_resource_schema(self,names,formula):
        schema=self.receipt_schema(formula)
        if not isinstance(names,(list,tuple)) or any(not isinstance(name,str) or not name for name in names):
            raise StrataError('receipt_schema.draw_resources must list resource names')
        declared=set(names)
        required=set(schema['required_draw_resources'])
        if declared & {'fail_count','shots'}:required|={'fail_count','shots'}
        missing=required-declared
        if missing:raise StrataError('verifier draw resource schema missing: '+', '.join(sorted(missing)))
        schema['required_draw_resources']=sorted(required)
        return schema

    def validate_selection(self,receipt):
        if not isinstance(receipt,dict):raise StrataError('probability policy requires a draw receipt')
        if list(receipt.get('seeds',[]))!=list(self.seeds):raise StrataError('draws differ from precommitted probability policy')
        selection=receipt.get('selection',{})
        if not isinstance(selection,dict) or selection.get('mode','precommitted')!='precommitted' or receipt.get('selected_seed') is not None:
            raise StrataError('candidate-selected favourable draw cannot be promoted')
        if selection.get('seeds',list(self.seeds))!=list(self.seeds):raise StrataError('selection uses different seeds')

    def evaluate(self,receipt,formula):
        self.validate_selection(receipt)
        draws=receipt.get('draws')
        if not isinstance(draws,list) or [d.get('seed') for d in draws]!=list(self.seeds):
            raise StrataError('one ordered receipt required for every precommitted draw')
        formula=Formula(formula,self.cost_functions()) if isinstance(formula,str) else formula
        costs=[];failures=[];failure_inputs=[]
        for draw in draws:
            resources=draw.get('resources',{})
            failure=resources.get(self.failure_resource)
            if type(failure) not in (int,float) or not isfinite(failure) or not 0<=failure<=1:
                raise StrataError('draw lacks a valid measured failure rate')
            count,shots=resources.get('fail_count'),resources.get('shots')
            if count is not None or shots is not None:
                if type(count) is not int or type(shots) is not int or shots<=0 or not 0<=count<=shots:
                    raise StrataError('failure counts require integer fail_count and positive shots')
                if failure!=count/shots or count and failure==0:
                    raise StrataError('empirical failure rate disagrees with fail_count/shots')
            bounds=draw.get('confidence_bounds',{})
            if not isinstance(bounds,dict):raise StrataError('confidence bounds must be structured metadata')
            for name,bound in bounds.items():
                value=resources.get(name)
                if (type(value) not in (int,float) or not isfinite(value) or not 0<=value<=1
                    or not isinstance(bound,dict) or bound.get('kind')!='upper'
                    or not isinstance(bound.get('method'),str) or not bound['method']
                    or count is None or type(bound.get('fail_count')) is not int or type(bound.get('shots')) is not int
                    or bound['fail_count']!=count or bound['shots']!=shots or value<failure):
                    raise StrataError('invalid verifier-supplied confidence bound')
                _confidence_log2_delta(bound)
            failures.append(failure)
            costs.append(formula(resources))
            failure_inputs.append(dict(empirical_rate=failure,fail_count=count,shots=shots,
                confidence_bounds={name:dict(value=resources[name],**bound,authority='verifier supplied; not recomputed by Strata') for name,bound in bounds.items()}))
        weights=self.weights or (1.0,)*len(draws)
        # Normalize before summation: individually finite weights can have an
        # infinite sum, which would erase both cost and failure probability.
        largest=max(weights)
        scaled=[w/largest for w in weights]
        denominator=fsum(scaled)
        normalized=[w/denominator for w in scaled]
        if any(w==0 for w in normalized):raise StrataError('probability weight range exceeds numerical precision')
        mean_failure=fsum(p*w for p,w in zip(failures,normalized))
        if mean_failure==0 and any(p>0 for p in failures):raise StrataError('failure aggregate underflow')
        checked_failure=max(failures) if self.compare=='worst' else mean_failure
        if checked_failure>self.max_failure:raise StrataError('probability failure budget exceeded')
        # Average objectives, not averaged resources fed to a nonlinear formula.
        cost=max(costs) if self.compare=='worst' else fsum(c*w for c,w in zip(costs,normalized))
        if not isfinite(cost) or not isfinite(mean_failure):raise StrataError('non-finite probability aggregate')
        if cost==0 and any(c>0 for c in costs):raise StrataError('cost aggregate underflow')
        return dict(policy_hash=self.fingerprint,comparison=self.compare,cost=cost,
                    per_draw_cost=costs,mean_failure=mean_failure,checked_failure=checked_failure,
                    seeds=list(self.seeds),weights=list(weights),evidence_kind='measured_precommitted_draws',
                    uncertainty='finite draw estimate; not an exact population expectation',
                    failure_inputs=failure_inputs,penalty=self.penalty,
                    penalty_inputs='per-draw resources; confidence bounds are verifier supplied',
                    formula=formula.expression)


def compare_candidates(policy,receipts,formula):
    evaluated=[dict(candidate=name,**policy.evaluate(receipt,formula)) for name,receipt in receipts.items()]
    return sorted(evaluated,key=lambda r:r['cost'])
