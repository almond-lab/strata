"""Precommitted stochastic comparisons; never rank by a candidate's lucky draw."""
from dataclasses import dataclass
from math import isfinite, fsum
from .model import StrataError,digest
from .cost import Formula


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
        return {'R':lambda p:penalty({self.failure_resource:p,'fail_rate':p})}

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
        costs=[];failures=[]
        for draw in draws:
            resources=draw.get('resources',{})
            failure=resources.get(self.failure_resource)
            if type(failure) not in (int,float) or not isfinite(failure) or not 0<=failure<=1:
                raise StrataError('draw lacks a valid measured failure rate')
            failures.append(failure)
            costs.append(formula(resources))
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
                    formula=formula.expression)


def compare_candidates(policy,receipts,formula):
    evaluated=[dict(candidate=name,**policy.evaluate(receipt,formula)) for name,receipt in receipts.items()]
    return sorted(evaluated,key=lambda r:r['cost'])
