"""Structural research rounds: widen boundaries, diversify mechanisms, verify changes.

This is an experiment selector/executor, not an algorithm synthesizer. A task
adapter or researcher still supplies the implementation. Unknown opportunities
remain eligible. Estimates never close a family.
"""
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from .model import StrataError, digest
from .engine import MOVES

DEEP_MOVES = {name for name, depth in MOVES.items() if depth.value in {'L3','L4'}}


@dataclass(frozen=True)
class Dependency:
    producer: str
    consumer: str
    reason: str
    evidence: str


@dataclass(frozen=True)
class Hypothesis:
    move: str
    regions: tuple[str, ...]
    mechanism: str
    assumption: str
    change: str
    decisive_test: str
    implementation: dict | None = None
    replacement_regions: tuple[str, ...] = ()
    unknowns: tuple[str, ...] = ()
    estimate: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.move not in MOVES or not self.regions:
            raise StrataError('hypothesis requires a typed move and regions')
        if any(not str(v).strip() for v in (self.mechanism,self.assumption,self.change,self.decisive_test)):
            raise StrataError('hypothesis needs mechanism, challenged assumption, change and decisive test')


@dataclass(frozen=True)
class ResearchPolicy:
    max_lines: int = 6
    max_per_mechanism: int = 2
    deep_only: bool = True
    max_proposals: int = 256
    max_small_instances: int = 16
    lab_max_trajectories: int = 100_000
    lab_timeout_s: float = 30.0

    def __post_init__(self):
        if min(self.max_lines,self.max_per_mechanism,self.max_proposals,self.max_small_instances,self.lab_max_trajectories,self.lab_timeout_s)<=0:
            raise StrataError('positive research budgets required')


def close_episode(tree, regions, dependencies=()):
    """Close over declared data and debt relationships, in both directions.

    This does not infer hidden dependencies from gates. Evidence and residual
    unknowns are retained so closure is not misrepresented as a proof.
    """
    index={r.id:r for r in tree.walk()}
    wanted=set(regions)
    if not wanted or any(name not in index for name in wanted):raise StrataError('unknown or empty episode regions')
    covered_ids={r.id for name in wanted for r in index[name].walk()}
    def add(name):
        wanted.add(name);covered_ids.update(r.id for r in index[name].walk())
    links=list(dependencies)
    for region in index.values():
        for debt in region.contract.debts:
            links.append(Dependency(debt.created_by,debt.repaid_by,'phase debt '+debt.id,'declared contract debt'))
    for link in links:
        if link.producer not in index or link.consumer not in index:raise StrataError('dependency names an unknown region')
        if not link.reason or not link.evidence:raise StrataError('dependency needs reason and evidence')
    followed=[]
    def covered(name):
        return name in covered_ids
    def span_targets():
        minimal=[]
        for r in sorted((index[x] for x in wanted),key=lambda r:(r.start,-r.stop)):
            if not minimal or not (minimal[-1].start<=r.start and r.stop<=minimal[-1].stop):minimal.append(r)
        lo,hi=min(r.start for r in minimal),max(r.stop for r in minimal)
        enclosing=min((r for r in index.values() if r.start<=lo and hi<=r.stop),key=lambda r:r.stop-r.start)
        if enclosing.id in wanted:return enclosing,[enclosing]
        targets=[r for r in enclosing.children if r.start<hi and lo<r.stop]
        return enclosing,targets or [enclosing]
    changed=True
    while changed:
        changed=False
        for link in links:
            if covered(link.producer) or covered(link.consumer):
                for name in (link.producer,link.consumer):
                    if not covered(name):add(name);changed=True
                if asdict(link) not in followed:followed.append(asdict(link))
        enclosing,targets=span_targets()
        for r in targets:
            if not covered(r.id):add(r.id);changed=True
    enclosing,targets=span_targets()
    return dict(regions=[r.id for r in targets],enclosing=enclosing.id,
                requested=list(regions),dependencies=followed,
                scope='closure of supplied dependencies only; source audit still required')


def structural_change(before, after, artifact_changed, witness=None):
    """Observed tree changes, not the depth claimed in the move's name.

    Names/kind labels and physical wire renumbering are ignored. Semantic
    contracts are adapter assertions checked at the episode's external boundary;
    this audit cannot independently infer internal wire meanings.
    """
    def contract(c):
        return dict(registers=[(len(r.wires),r.role,r.meaning) for r in c.registers if r.role!='clean'],
                    reference=c.reference,promise=c.promise,phase=c.phase,failure=c.failure,
                    debts=[(d.created_by,d.repaid_by,d.measurement,d.expression) for d in c.debts])
    def flatten(root):
        result={}
        def visit(r,path):
            result[path]=r
            for i,c in enumerate(r.children):visit(c,path+(i,))
        visit(root,())
        return result
    a,b=flatten(before),flatten(after)
    topology=set(a)!=set(b)
    boundaries=[list(p) for p in set(a)&set(b) if p and contract(a[p].contract)!=contract(b[p].contract)]
    root_debt=contract(before.contract)['debts']!=contract(after.contract)['debts']
    # A different top-level stage count is architecture; deeper topology or
    # typed intermediate boundary/debt changes establish declared L3 structure.
    depth='L4' if len(before.children)!=len(after.children) else 'L3' if topology or boundaries or root_debt else 'L2'
    candidate=bool(artifact_changed and depth in {'L3','L4'})
    attested=bool(witness and witness.get('passed') is True and witness.get('mechanism') and witness.get('checks') and all(v is True for v in witness['checks'].values()))
    return dict(deep=bool(candidate and attested),structural_candidate=candidate,structural_witness=witness,observed_depth=depth,
                artifact_changed=artifact_changed,topology_changed=topology,
                boundary_paths=sorted(boundaries),root_debt_changed=root_debt,
                evidence='emitted bytes plus declared contract-tree structure; not novelty or internal equivalence proof')


def templates(tree):
    """Concrete questions, with no invented generator implementation or gain."""
    for r in tree.walk():
        if not r.children:continue
        yield Hypothesis('replace-stages',(r.id,),'algorithm',
            'The current stage sequence is necessary',
            'Replace the sequence while preserving its external function',
            'Derive a different stage sequence and test its full return boundary',
            unknowns=('alternative algorithm','complete resource bill'))
        for a,b in zip(r.children,r.children[1:]):
            if a.stop!=b.start:continue
            yield Hypothesis('fuse',(a.id,b.id),'materialization',
                'The producer result must be materialized before the consumer',
                'Consume the result directly and eliminate its storage and cleanup',
                'Test the widened producer-consumer-return contract',
                unknowns=('data dependencies','cleanup owners','consumer adaptation'))
            yield Hypothesis('re-represent',(a.id,b.id),'representation',
                'The current boundary representation is required by the task',
                'Co-design the producer and consumer around a different representation',
                'Check a scaled representation change including decoding and return',
                unknowns=('new encoding','zero branches','full peak width'))


class ResearchRound:
    def __init__(self,project,parent,policy=None,dependencies=()):
        from .project import Design
        self.project=project
        self.parent=Design(project,parent if isinstance(parent,str) else parent.id)
        self.parent.validate()
        self.policy=policy or ResearchPolicy()
        self.dependencies=tuple(dependencies)

    def plan(self,hypotheses=None):
        from .project import Design
        self.parent=Design(self.project,self.parent.id)
        self.parent.validate()
        if hypotheses is None:
            adapter=self.project.task(self.parent.record['task']).adapter()
            hypotheses=adapter.restructures(self.parent.tree) if hasattr(adapter,'restructures') else templates(self.parent.tree)
        # Copy caller-owned mutable dictionaries before persisting or scheduling.
        import json
        from itertools import islice
        bounded=list(islice(iter(hypotheses),self.policy.max_proposals+1))
        truncated=len(bounded)>self.policy.max_proposals
        hypotheses=[Hypothesis(**json.loads(json.dumps(asdict(h)))) for h in bounded[:self.policy.max_proposals]]
        groups=defaultdict(deque);skipped=[];seen=set()
        for h in hypotheses:
            if self.policy.deep_only and h.move not in DEEP_MOVES:
                skipped.append(dict(reason='local move outside this structural round',hypothesis=asdict(h)));continue
            episode=close_episode(self.parent.tree,h.regions,self.dependencies)
            fp=[self.parent.tree.find(x).contract.fingerprint for x in episode['regions']]
            # Same emitted generator request need not run twice under new labels.
            key=digest(dict(parent=self.parent.id,implementation=h.implementation)) if h.implementation is not None else digest(dict(move=h.move,contracts=fp,mechanism=h.mechanism,change=h.change))
            if key in seen:
                skipped.append(dict(reason='duplicate implementation or structural question',hypothesis=asdict(h)));continue
            seen.add(key)
            history=[dict(handle=k,**v) for f in fp for k,v in self.project.store.tried(f,h.move)]
            item=dict(hypothesis=asdict(h),episode=episode,history=history,
                      estimate_status='hypothesis only; never a closure or promotion',
                      readiness='implementation_available' if h.implementation is not None else 'derivation_required')
            groups[h.mechanism].append(item)
        selected=[];per=defaultdict(int)
        while len(selected)<self.policy.max_lines:
            advanced=False
            for mechanism,queue in groups.items():
                if queue and per[mechanism]<self.policy.max_per_mechanism:
                    selected.append(queue.popleft());per[mechanism]+=1;advanced=True
                    if len(selected)==self.policy.max_lines:break
            if not advanced:break
        deferred=[item for queue in groups.values() for item in queue]
        result=dict(parent=self.parent.id,policy=asdict(self.policy),proposal_budget_reached=truncated,selected=selected,deferred=deferred,skipped=skipped,
                    note='Budgets limit this round; deferred and estimated ideas remain open.')
        handle=self.project.store.put('research_plan',result)
        return dict(handle=handle,**result)

    def run(self,plan_handle):
        from .project import Design
        self.parent=Design(self.project,self.parent.id)
        self.parent.validate()
        plan=self.project.store.get(plan_handle,'research_plan')
        if plan['parent']!=self.parent.id:raise StrataError('research plan belongs to a different base')
        outcomes=[]
        for item in plan['selected']:
            h=Hypothesis(**item['hypothesis'])
            if h.implementation is None:
                outcomes.append(dict(mechanism=h.mechanism,status='DERIVATION_REQUIRED',unknowns=h.unknowns));continue
            line=self.project.open_line(self.parent,h.move,item['episode']['regions'],h.change,
                                         h.implementation,replacement_regions=h.replacement_regions)
            stage='lab'
            design=None
            try:
                labs=[]
                # Every configured small size is covered; failures at any size
                # reject this construction. No extrapolation from one lucky size.
                from itertools import product
                scales=self.project.task(self.parent.record['task']).config.get('scale',{})
                values=[v['small'] for v in scales.values()]
                import math
                if math.prod(map(len,values))>plan['policy']['max_small_instances']:
                    raise StrataError('small-scale suite exceeds declared round budget; no partial qualification')
                for combination in product(*values):
                    scale=dict(zip(scales,combination));lab=line.lab(limits=dict(max_trajectories=plan['policy']['lab_max_trajectories'],timeout=plan['policy']['lab_timeout_s']),**scale);labs.append(asdict(lab))
                    if not lab.passed:raise StrataError('small-instance counterexample')
                stage='build_price';built=line.build()
                from .model import region_from_dict
                adapter=self.project.task(self.parent.record['task']).adapter()
                witness=None
                if hasattr(adapter,'audit_restructure'):
                    old_circuit=self.project.frontend.parse(self.project.store.read_blob(self.parent.record['artifact']).decode())
                    new_circuit=self.project.frontend.parse(self.project.store.read_blob(built['artifact']).decode())
                    witness=adapter.audit_restructure(old_circuit,self.parent.tree,new_circuit,region_from_dict(built['tree']))
                    if witness.get('before_artifact')!=self.parent.record['artifact'] or witness.get('after_artifact')!=built['artifact']:
                        raise StrataError('structural witness is bound to different artifacts')
                change=structural_change(self.parent.tree,region_from_dict(built['tree']),built['artifact']!=self.parent.record['artifact'],witness)
                if plan['policy']['deep_only'] and not change['deep']:
                    outcomes.append(dict(line=line.id,mechanism=h.mechanism,status='STRUCTURAL_AUDIT_REQUIRED' if change['structural_candidate'] and witness is None else 'BUILT_NONSTRUCTURAL',
                                         structure=change,measured_delta=built['delta'],labs=labs));continue
                stage='verify';design=line.verify()
                stage='replay';replay=self.project.replay(design.id)
                before,after=self.parent.record['cost'],design.record['cost']
                multiplier=(before/after)**(1/len(self.project.tasks)) if after>0 else None
                outcomes.append(dict(line=line.id,design=design.id,mechanism=h.mechanism,
                    status=('VERIFIED_STRUCTURAL_' if change['deep'] else 'VERIFIED_LOCAL_')+('GAIN' if after<before else 'NO_GAIN'),
                    structure=change,cost_before=before,cost_after=after,cost_delta=after-before,
                    portfolio_multiplier_fixed_other_tasks=multiplier,
                    certificate=design.record['certificate'],replay=replay['passed'],labs=labs))
            except (StrataError,ValueError,RuntimeError) as exc:
                if stage=='replay' and design is not None:
                    self.project.store.put('quarantine',dict(design=design.id,reason=str(exc),stage='replay',plan=plan_handle))
                evidence=dict(stage=stage,error=str(exc))
                self.project.store.event(line.id,dict(stage='research_attempt',passed=False,**{'failure':evidence}))
                outcomes.append(dict(line=line.id,mechanism=h.mechanism,status='ATTEMPT_FAILED',
                                     failure=evidence,scope='this implementation/run only; family remains open'))
        result=dict(parent=self.parent.id,plan=plan_handle,outcomes=outcomes,
                    qualified_gains=sum(x['status']=='VERIFIED_STRUCTURAL_GAIN' for x in outcomes))
        handle=self.project.store.put('research_result',result)
        return dict(handle=handle,**result)
