"""Bounded, extensible construction search with independently checked receipts.

Constructors receive mathematical circuit evidence. They cannot change the
boundary checker, declared clean inputs, objective, or acceptance threshold.
Verified non-winning states stay in the exploration archive. Learned native
templates are replayed on each new binding; history is never a proof.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re
import shlex
import time

import kickmix as km

from .cost import Formula
from .frontends import load_circuit
from .kickmix import Kickmix
from .model import StrataError, digest
from .semantic import analyze_window, analyze_complete, equivalent_windows, semantic_function
from .streaming import canonical_kernel, file_hash


@dataclass(frozen=True)
class Construction:
    name: str
    propose: object
    description: str


class ConstructorRegistry:
    """Host extensions are trusted code, not instructions from circuit metadata."""
    def __init__(self):
        self.constructors = {}

    def register(self, name, propose, description):
        if name in self.constructors or not callable(propose):
            raise StrataError('constructor name must be unique and implementation callable')
        self.constructors[name] = Construction(name, propose, description)


def _phase(circuit, context):
    if context['semantics']['status']=='exact_factored':
        return dict(status='not_applicable',reason='exact factored boundary recovered; phase constructor requires explicit ANF')
    from .inference import eliminate_clean_phase_workspace
    return eliminate_clean_phase_workspace(circuit, clean_inputs=context['clean_inputs'],
        drop_clean=True, max_operations=context['max_operations'],
        max_variables=context['max_variables'], max_terms=context['max_terms'],
        max_product_terms=context.get('max_product_terms',65536))


def _measured_lookup(circuit, context):
    from .lookup import resynthesize_measured_lookup
    return resynthesize_measured_lookup(circuit,clean_inputs=context['clean_inputs'],
        max_operations=context['max_operations'],max_variables=context['max_variables'],
        max_terms=context['max_terms'],max_product_terms=context.get('max_product_terms',65536),
        boundary_table=context.get('lookup_table'))


def _boundary_metadata(circuit, wire=None, name='strata_boundary'):
    """Preserve public registers and make repeated constructor calls idempotent."""
    metadata = [line for line in str(circuit).splitlines()
                if line.split() and line.split()[0] in {'REGISTER', 'APPEND_TO_REGISTER', 'INFO_TAG'}]
    if wire is None:
        return metadata
    names = set()
    for line in metadata:
        words = shlex.split(line)
        if words[0] == 'REGISTER':
            names.add(words[2])
        elif words[0] == 'APPEND_TO_REGISTER' and words[1] == f'q{wire}':
            return metadata
    base = name
    suffix = 1
    while name in names:
        name = f'{base}_{suffix}'
        suffix += 1
    register = circuit.num_registers
    return metadata + [f'APPEND_TO_REGISTER q{wire} r{register}', f'REGISTER r{register} "{name}"']


def _direct_consumer(circuit, context):
    from .file_optimizer import _resynthesize
    sem = context['semantics']
    if sem['status'] != 'exact':
        return dict(status='not_applicable' if sem['status']=='exact_factored' else 'unknown',
                    reason='exact factored boundary recovered; direct consumer requires ANF and matching measurement provenance'
                    if sem['status']=='exact_factored' else 'exact complete boundary unavailable')
    emitted, reason = _resynthesize(sem, max_operations=context['max_operations'])
    if emitted is None:
        return dict(status='unknown', reason=reason)
    # Preserve native physical boundary width and declared register metadata.
    metadata = _boundary_metadata(circuit, circuit.num_qubits - 1 if circuit.num_qubits else None)
    return dict(status='constructed', candidate='\n'.join(metadata+emitted),
                reason='resynthesize the complete producer-consumer output and phase functions')


def _tensor(circuit, context):
    from .tensor import tensor_from_anf,search_decomposition,lower_decomposition,extract_decomposition
    sem=context['semantics']
    if sem['status']!='exact' or sem['phase']['monomials']:
        return dict(status='unknown',reason='requires exact phase-free bilinear accumulator')
    feature=next((f for f in sem['features'] if f['kind']=='bilinear_accumulator'),None)
    if feature is None:return dict(status='not_applicable',reason='no complete bilinear accumulator contract')
    left,right=feature['factor_partition'];targets=feature['targets']
    tensor=tensor_from_anf(feature['deltas'],left,right,output_wires=targets)
    extracted=extract_decomposition(circuit,tensor,clean_inputs=context['clean_inputs'],
                                    max_operations=context['max_operations'],max_variables=context['max_variables'])
    seed=extracted.get('decomposition')
    result=search_decomposition(tensor,seed=seed)
    # A product-count tie cannot justify replacing the native circuit by a
    # potentially much larger operand/fanout construction. Keep its witness.
    if seed is not None and result['rank']>=len(seed):
        return dict(status='constructed',candidate=str(circuit),tensor_search=result,seed_extraction=extracted,
                    reason='retained the verified gate decomposition; bounded search found no product-count improvement')
    source_products=sum(line.startswith('CCX ') for line in str(circuit).splitlines())
    if result['rank']>source_products:
        return dict(status='not_applicable',reason='verified construction would increase the source product count',
                    tensor_search=result,seed_extraction=extracted)
    clean=[q for q in context['clean_inputs'] if q not in left+right+targets]
    try:lowered=lower_decomposition(tensor,result['decomposition'],clean_helpers=clean,max_operations=context['max_operations'])
    except StrataError as exc:
        if seed is not None:
            return dict(status='constructed',candidate=str(circuit),reason='retained verified gate decomposition; improved tensor could not be lowered',
                        lowering_obstruction=str(exc),tensor_search=result,seed_extraction=extracted)
        return dict(status='unknown',reason=str(exc),tensor_search=result,seed_extraction=extracted)
    public=set(range(circuit.num_qubits))-set(context['clean_inputs'])
    metadata=_boundary_metadata(circuit,max(public) if public else None,'strata_public_boundary')
    lowered['candidate']='\n'.join(metadata+[lowered['candidate']]);lowered['tensor_search']=result
    lowered['seed_extraction']=extracted
    return lowered


def _guided_tensor(circuit, context):
    """Opt into analysis-directed batched proposals only for bounded catalogs."""
    from .tensor import tensor_from_anf
    from .candidate_search import search_candidates
    sem=context['semantics']
    feature=next((f for f in sem.get('features',()) if f['kind']=='bilinear_accumulator'),None)
    if feature is None or sem.get('phase',{}).get('monomials'):
        return dict(status='not_applicable',reason='no exact phase-free bilinear boundary')
    left,right=feature['factor_partition'];targets=feature['targets']
    clean=[q for q in context['clean_inputs'] if q not in left+right+targets]
    if len(clean)<2:return dict(status='not_applicable',reason='two declared clean operand helpers unavailable')
    tensor=tensor_from_anf(feature['deltas'],left,right,output_wires=targets)
    search=search_candidates(tensor,strategy='guided',candidate_budget=1024,
                             materialization_limit=16,native_limit=4,clean_helpers=clean[:2])
    if search.get('best') is None:return dict(status='unknown',reason='no verified construction under bounded guided search',search=search)
    public=set(range(circuit.num_qubits))-set(context['clean_inputs'])
    metadata=_boundary_metadata(circuit,max(public) if public else None,'strata_public_boundary')
    return dict(status='constructed',candidate='\n'.join(metadata+[search['best']['native_text']]),
                search=search,resource_scope='Complete parent emission and equality are checked again by the construction loop.')


def default_registry():
    registry = ConstructorRegistry()
    registry.register('observable_phase_elimination', _phase,
                      'Replace a restored materialized phase word with its exact observable phase; retire declared clean trailing workspace.')
    registry.register('direct_consumer_synthesis', _direct_consumer,
                      'Compose producer, consumer and restoration semantics before emitting output updates.')
    registry.register('bilinear_tensor_decomposition',_tensor,
                      'Search factored multi-output products and emit a complete reversible accumulator using declared clean helpers.')
    registry.register('guided_bilinear_candidate_search',_guided_tensor,
                      'Use the recovered tensor and residual constraints to propose factorizations, then native-batch and exactly check finalists.')
    registry.register('measured_lookup_resynthesis',_measured_lookup,
                      'Factor supported lookup accumulators while preserving HMR measurements, dirty outputs and exact phase repairs.')
    return registry


def _check(original, candidate, clean, limits):
    # Common public wire indices are fixed. Omitted private inputs are promised
    # zero and must be restored. ANF equality checks implicit identity outputs.
    removed = set(range(candidate.num_qubits, original.num_qubits))
    if candidate.num_qubits > original.num_qubits or not removed <= set(clean):
        return dict(equivalent=None, reason='undeclared_or_expanded_boundary_layout')
    check = equivalent_windows(str(original).splitlines(), str(candidate).splitlines(),
                               clean_inputs=clean, **limits)
    return check


def _pareto(states):
    def resources(s):
        r=s['resources'];return (r['Q'],r['toffoli'],r['t'],r['rotations'],r['instructions'])
    return [s['id'] for s in states if not any(
        other['id']!=s['id'] and all(a<=b for a,b in zip(resources(other),resources(s)))
        and any(a<b for a,b in zip(resources(other),resources(s))) for other in states)]


def _learn_template(before, after, clean, mechanism):
    # Only pure, arbitrary-basis templates enter this executable archive. Native
    # metadata is omitted, classical scopes/measurement need stronger alignment.
    before_lines=[line for line in str(before).splitlines() if line.split()[0] not in {'REGISTER','APPEND_TO_REGISTER','INFO_TAG'}]
    after_lines=[line for line in str(after).splitlines() if line.split()[0] not in {'REGISTER','APPEND_TO_REGISTER','INFO_TAG'}]
    if any(re.search(r'\bb\d+\b', line) or line.split()[0] in {'R','HMR','PUSH_CONDITION','POP_CONDITION'} for line in before_lines+after_lines):
        return None
    source,mapping=canonical_kernel(before_lines)
    if not source or any(f'q{q}' not in mapping for q in clean):
        return None
    def rename(match):
        token=match.group()
        if token not in mapping:raise KeyError(token)
        return mapping[token]
    try:
        target='\n'.join(re.sub(r'\bq\d+\b', rename, line) for line in after_lines)
    except KeyError:
        return None
    return dict(version=1, mechanism=mechanism, pattern=source, replacement=target,
                clean_inputs=[int(mapping[f'q{q}'][1:]) for q in clean],
                predicate='exact gate template modulo quantum wire renaming, declared-zero inputs and independently replayed boundary equality',
                authority='verified_example; new applications require proof')


def _template_constructor(template):
    def propose(circuit, context):
        source_lines=str(circuit).splitlines()
        body=[line for line in source_lines if line.split()[0] not in {'REGISTER','APPEND_TO_REGISTER','INFO_TAG'}]
        canonical,mapping=canonical_kernel(body)
        if canonical != template['pattern']:
            return dict(status='not_applicable',reason='learned template precondition does not match')
        inverse={value:key for key,value in mapping.items()}
        if any(f'q{q}' not in inverse for q in template['clean_inputs']):
            return dict(status='unknown',reason='learned clean-input precondition references an undeclared template wire')
        clean={int(inverse[f'q{q}'][1:]) for q in template['clean_inputs']}
        if not clean<=set(context['clean_inputs']):
            return dict(status='not_applicable',reason='learned clean-input precondition absent')
        if any(token not in inverse for token in re.findall(r'\bq\d+\b',template['replacement'])):
            return dict(status='unknown',reason='learned replacement references an undeclared template wire')
        target=re.sub(r'\bq\d+\b', lambda m:inverse[m.group()],template['replacement'])
        public=set(range(circuit.num_qubits))-set(context['clean_inputs'])
        metadata=_boundary_metadata(circuit,max(public) if public else None,'strata_public_boundary')
        return dict(status='constructed',candidate='\n'.join(metadata+[target]),learned_preconditions_checked=True)
    return propose


def _validate_archive(memory):
    if not isinstance(memory,dict) or memory.get('version')!=1:
        raise StrataError('unsupported mechanism archive')
    if not isinstance(memory.get('templates'),list) or not isinstance(memory.get('statistics'),dict):
        raise StrataError('mechanism archive needs template list and statistics object')
    for template in memory['templates']:
        if (not isinstance(template,dict) or template.get('version')!=1 or
            any(not isinstance(template.get(key),str) for key in ('pattern','replacement','mechanism')) or
            not isinstance(template.get('clean_inputs'),list) or
            any(type(q)is not int or q<0 for q in template['clean_inputs'])):
            raise StrataError('invalid learned template schema')
    for key,counts in memory['statistics'].items():
        if not isinstance(key,str) or not isinstance(counts,dict) or any(type(counts.get(n))is not int or counts[n]<0 for n in ('attempts','verified','gains')):
            raise StrataError('invalid mechanism statistics schema')


def restructure_file(path, out, *, clean_inputs=(), cost='Q*(toffoli+t/4)',
                     registry=None, rounds=2, beam=4, max_candidates=24,
                     max_operations=10000, max_variables=128, max_terms=65536,
                     max_product_terms=65536, archive=None):
    """Construct and verify complete-boundary candidates; preserve every outcome.

    The connected skill invents further mechanisms when bounded constructors
    lack a witness. This command invokes no remote AI or untrusted code.
    """
    if min(rounds,beam,max_candidates,max_operations,max_variables,max_terms,max_product_terms)<=0:
        raise StrataError('restructure budgets must be positive')
    source=Path(path).resolve();root=Path(out).resolve();root.mkdir(parents=True,exist_ok=True)
    if source in {root/'restructure.json',root/'best.kmx'}:
        raise StrataError('restructure output would overwrite source')
    circuit=load_circuit(source)
    if len(circuit)>max_operations:
        result=dict(status='unknown',reason='complete episode exceeds construction budget',
                    next_test='Use discover --range to recover a complete producer-consumer boundary; no fragment saving is claimed.',
                    source_sha256=file_hash(source),max_operations=max_operations)
        (root/'restructure.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
    clean=tuple(sorted(set(clean_inputs)))
    if any(type(q)is not int or not 0<=q<circuit.num_qubits for q in clean):
        raise StrataError('clean inputs must be physical wires of the supplied circuit')
    formula=Formula(cost);frontend=Kickmix();limits=dict(max_operations=max_operations,max_variables=max_variables,
                                                       max_terms=max_terms,max_product_terms=max_product_terms)
    registry=registry or default_registry()
    memory_path=Path(archive).resolve() if archive else root/'mechanisms.json'
    memory=json.loads(memory_path.read_text()) if memory_path.exists() else dict(version=1,templates=[],statistics={})
    _validate_archive(memory)
    constructors=list(registry.constructors.values())
    constructors += [Construction('learned_'+digest(t)[:12],_template_constructor(t),'Replayed learned gate template with checked predicates') for t in memory['templates']]
    def state(c, parent, mechanism):
        text=str(c);key=hashlib.sha256(text.encode()).hexdigest();resources=frontend.measured(c)
        (root/f'{key}.kmx').write_text(text)
        return dict(id=key,path=str(root/f'{key}.kmx'),resources=resources,cost=formula(resources),parent=parent,mechanism=mechanism)
    baseline=state(circuit,None,'supplied_baseline');states=[baseline];frontier=[(circuit,baseline)];seen={baseline['id']};attempts=[]
    started=time.monotonic();boundary_receipts={};lookup_tables={}
    for generation in range(rounds):
        following=[]
        for parent,parent_state in frontier:
            semantics=analyze_complete(str(parent).splitlines(),clean_inputs=clean,**limits)
            boundary_receipts[parent_state['id']]=semantics
            context=dict(semantics=semantics,clean_inputs=clean,**limits)
            if semantics['status']=='exact_factored' and semantics.get('phase_zero'):
                from .factored_semantic import lift_lookup_table
                table=lift_lookup_table(parent,clean_inputs=clean,max_operations=max_operations,
                    max_variables=max_variables,max_nodes=max_terms,max_apply_steps=32*max_product_terms)
                context['lookup_table']=table
                lookup_tables[parent_state['id']]=table
            for constructor in constructors:
                if len(attempts)>=max_candidates:break
                attempt=dict(round=generation,parent=parent_state['id'],mechanism=constructor.name)
                try:
                    proposal=constructor.propose(parent,context)
                    attempt['construction']=proposal
                    if proposal.get('status')!='constructed':
                        attempt.update(status='not_constructed',reason=proposal.get('reason','preconditions unavailable'))
                    else:
                        candidate=km.Circuit(proposal['candidate'])
                        check=_check(parent,candidate,clean,limits)
                        attempt['check']=check
                        attempt['status']='verified' if check.get('equivalent')is True else 'rejected' if check.get('equivalent')is False else 'unknown'
                        if check.get('equivalent')is True:
                            child=state(candidate,parent_state['id'],constructor.name)
                            attempt.update(candidate=child['id'],cost_delta=child['cost']-parent_state['cost'],
                                           cumulative_cost_delta=child['cost']-baseline['cost'])
                            if child['id'] not in seen:
                                states.append(child);seen.add(child['id']);following.append((candidate,child))
                            if child['cost']<parent_state['cost']:
                                learned=_learn_template(parent,candidate,clean,constructor.name)
                                if learned and learned not in memory['templates']:memory['templates'].append(learned)
                except (StrataError,ValueError,RuntimeError) as exc:
                    attempt.update(status='unknown',reason=str(exc))
                attempt['inference_scope']='one construction under a frozen boundary; no entire mechanism is closed'
                attempts.append(attempt)
                stats=memory['statistics'].setdefault(constructor.name,dict(attempts=0,verified=0,gains=0))
                stats['attempts']+=1;stats['verified']+=attempt['status']=='verified';stats['gains']+=attempt.get('cost_delta',0)<0
        # Diversity first: retain valid alternative representations even when
        # they cost more. Archive is complete; beam only bounds the next round.
        frontier=sorted(following,key=lambda row:(row[1]['mechanism'],row[1]['cost']))[:beam]
        if not frontier or len(attempts)>=max_candidates:break
    best=min(states,key=lambda s:s['cost'])
    (root/'best.kmx').write_text(Path(best['path']).read_text())
    memory_path.parent.mkdir(parents=True,exist_ok=True);memory_path.write_text(json.dumps(memory,indent=2)+'\n')
    result=dict(schema_version='3',status='verified_improvement' if best['cost']<baseline['cost'] else 'no_gain',
                source=str(source),source_sha256=file_hash(source),baseline=baseline,best=best,
                cost_delta=best['cost']-baseline['cost'],states=states,pareto_frontier=_pareto(states),attempts=attempts,
                boundary_receipts=boundary_receipts,lookup_tables=lookup_tables,
                archive=str(memory_path),learned_template_count=len(memory['templates']),seconds=time.monotonic()-started,
                contract=dict(clean_inputs=clean,public_wires='every other physical wire, fixed indices',outputs='all physical outputs and exact sign equal to supplied baseline; only proved restored clean wires may leave the layout',
                              measurement_scope='branch-preserving equivalence only; no changed instrument proof'),
                objective=dict(expression=cost,accounting='static_emitted'),official_verifier_run=False,
                learning_scope='executable checked templates and observed constructor statistics; no model weight training or universal discovery claim',
                next_agent_questions=['Which costly materialization exists only to feed a later observable?',
                    'Can an alternate representation remove a dependency across the complete producer-consumer-return boundary?',
                    'Which exact support relation can produce a reversible encoder and a priced consumer adapter?',
                    'Which bounded or failed formulation could reopen under a different construction?'],
                budgets=dict(rounds=rounds,beam=beam,max_candidates=max_candidates,**limits))
    from .store import Store
    store=Store(root/'.strata')
    try:
        result['parent_sha256']=baseline['id']
        result['artifacts']={s['id']:store.blob(Path(s['path']).read_bytes()) for s in states}
        result['code']={name:store.blob((Path(__file__).parent/name).read_bytes())
                        for name in ('restructure.py','semantic.py','factored.py','factored_semantic.py','inference.py','tensor.py','candidate_search.py','domains.py','lookup.py')}
        result['mechanism_archive_blob']=store.blob(memory_path.read_bytes())
        result['record']=store.put('restructure_run',result)
    finally:
        store.db.close()
    (root/'restructure.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
