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
        from .lookup import resynthesize_supported_segments
        return resynthesize_supported_segments(circuit,clean_inputs=context['clean_inputs'],mode='phase',
            **{k:context[k] for k in ('max_operations','max_variables','max_terms','max_product_terms')})
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
        if words[0] == 'REGISTER' and len(words)>2:
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
    if sem['status'] == 'exact_factored':
        from .lookup import resynthesize_supported_segments
        return resynthesize_supported_segments(circuit,clean_inputs=context['clean_inputs'],mode='direct',
            **{k:context[k] for k in ('max_operations','max_variables','max_terms','max_product_terms')})
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
                             materialization_limit=16,native_limit=4,clean_helpers=clean[:2],
                             cost=context.get('cost','Q*toffoli'))
    if search.get('best') is None:return dict(status='unknown',reason='no verified construction under bounded guided search',search=search)
    public=set(range(circuit.num_qubits))-set(context['clean_inputs'])
    metadata=_boundary_metadata(circuit,max(public) if public else None,'strata_public_boundary')
    return dict(status='constructed',candidate='\n'.join(metadata+[search['best']['native_text']]),
                search=search,resource_scope='Complete parent emission and equality are checked again by the construction loop.')


def _joint_options(context):
    timeout = context['joint_timeout']
    wall = context.get('constructor_time_budget')
    if wall is not None:
        # Leave some time for native lowering and transferring the solver receipt.
        timeout = min(timeout, 0.8 * wall) if timeout is not None else 0.8 * wall
    return dict(max_and=context['max_and'], timeout=1.0 if timeout is None else timeout,
                max_states=context['joint_max_states'], max_expansions=context['joint_max_expansions'])


def _joint_boundary(circuit, context):
    """Search the composed function, not replacements for individual gates."""
    from .boolean_synthesis import joint_boundary_synthesis
    if len(context['clean_inputs']) < 2:
        return dict(status='not_applicable',reason='joint materialization-bank search needs at least two declared clean wires')
    result=joint_boundary_synthesis(circuit,clean_inputs=context['clean_inputs'],internal_bits=context.get('internal_bits',()), **_joint_options(context),
        **{key:context[key] for key in ('max_operations','max_variables','max_terms','max_product_terms')})
    context['joint_result']=result
    return result


def _scheduled_boundary(circuit,context):
    from .scheduled_boolean_synthesis import scheduled_boundary_synthesis
    joint=context.get('joint_result')
    if not joint or joint.get('status')!='constructed':
        return dict(status='not_applicable',reason='no synthesized joint function DAG to schedule')
    return scheduled_boundary_synthesis(circuit,joint_result=joint,cost=context['cost'],
        resource_measure=context['resource_measure'],clean_inputs=context['clean_inputs'],
        **{key:context[key] for key in ('max_operations','max_variables','max_terms','max_product_terms')})


def _algebraic_boundary(circuit,context):
    from .algebraic_synthesis import algebraic_boundary_synthesis
    result=algebraic_boundary_synthesis(circuit,clean_inputs=context['clean_inputs'],cost=context['cost'],
        resource_measure=context['resource_measure'],
        **{key:context[key] for key in ('max_operations','max_variables','max_terms','max_product_terms')})
    context['algebraic_result']=result
    return result


def _representation_boundary(circuit,context):
    from .representation_synthesis import representation_boundary_synthesis
    result=representation_boundary_synthesis(circuit,clean_inputs=context['clean_inputs'],cost=context['cost'],
        resource_measure=context['resource_measure'],
        **{key:context[key] for key in ('max_operations','max_variables','max_terms','max_product_terms')})
    if result.get('selected_mechanism') == 'affine_encoding':
        controls = [row['cost'] for row in result['alternatives'] if not row['encoding']['forward']]
        result['structural_witness'] = dict(result.get('structural_witness', {}),
            paid_input_basis_change=bool(controls) and result['cost'] < min(controls))
    return result


def _structural_options(context):
    return {key:context[key] for key in ('clean_inputs','cost','resource_measure','max_operations',
                                         'max_variables','max_terms','max_product_terms')}


def _functional_decomposition(circuit, context):
    if context['semantics'].get('phase',{}).get('monomials'):
        return dict(status='not_applicable',reason='nonzero observable phase')
    from .functional_decomposition import decomposition_synthesis
    return decomposition_synthesis(circuit, **_structural_options(context))


def _observable_search(circuit, context):
    semantics=context['semantics']
    if semantics.get('status')=='exact' and not semantics['phase']['monomials']:
        return dict(status='not_applicable',reason='exact zero boundary sign')
    from .observable_search import observable_boundary_synthesis
    return observable_boundary_synthesis(circuit, **_structural_options(context))


def _mixed_observable_search(circuit, context):
    semantics = context['semantics']
    if semantics.get('status') == 'exact' and (
        not semantics['phase']['monomials'] or not semantics['modified_wires']
    ):
        return dict(status='not_applicable', reason='requires both an observable sign and changed output values')
    from .mixed_observable import mixed_boundary_synthesis
    return mixed_boundary_synthesis(circuit, **_structural_options(context))


def _output_basis_search(circuit, context):
    if context['semantics'].get('phase',{}).get('monomials'):
        return dict(status='not_applicable',reason='nonzero observable phase')
    from .output_basis_search import output_basis_synthesis
    return output_basis_synthesis(circuit, **_structural_options(context))


def _architecture_portfolio(circuit, context):
    from .architecture_portfolio import architecture_portfolio
    # Reserve part of the external trial for final verification and transferring
    # the receipt, even when a caller chooses a much smaller trial budget.
    allotted = context.get('constructor_time_budget')
    proposal = architecture_portfolio(circuit, **_structural_options(context),
        max_hypotheses=context['max_architectures'], max_representations=context['max_architectures'],
        max_finalists=min(24, 1 + context['extra_finalist_budget']),
        timeout=.65 * allotted if allotted is not None else 2.5)
    # The search receipt keeps corpus-wide counts and the exact lowered finalists.
    # Unlowered equations can be deterministically regenerated from the source.
    formulations = proposal.pop('formulations', [])
    if formulations:
        proposal['formulation_sample'] = [dict(formulation_hash=row['formulation_hash'],
            category=row['category'], bounds=row['resource_bounds']) for row in formulations[:8]]
    proposal.pop('verified_elites', None)  # the constructor also exposes these as alternatives
    return proposal


def _affine_transport(circuit, context):
    from .affine_transport import affine_transport_synthesis
    return affine_transport_synthesis(circuit, **_structural_options(context))

def _fractional_observable(circuit, context):
    if not context.get('fractional_phase'):
        return dict(status='not_applicable',reason='no fractional rotations on this boundary')
    from .fractional_observable import fractional_boundary_synthesis
    return fractional_boundary_synthesis(circuit, **_structural_options(context))


def _qualifying_mechanism(proposal):
    witness = proposal.get('structural_witness', {})
    for kind in ('removed_materialization_bank', 'discovered_consumer_class_bit',
                 'paid_input_basis_change', 'paid_output_basis_change',
                 'removed_phase_word', 'factored_phase_observable', 'separated_value_and_phase',
                 'joint_architecture_change','affine_transport_factored','factored_fractional_observable','promise_specialization','terminal_products_fused'):
        if witness.get(kind) is True:
            return kind
    return None


def _order_deep_search(constructors, semantics, clean, remaining):
    """Prioritize a compatible family when an episode only gets a few trials."""
    if semantics.get('phase',{}).get('monomials') or semantics.get('phase_zero') is False:
        preferred = (['mixed_observable_search','factored_observable_search']
                     if semantics.get('modified_wires') else ['factored_observable_search','mixed_observable_search'])
    elif semantics.get('status') == 'exact':
        deltas = []
        for name, value in semantics.get('outputs', {}).items():
            q = int(name)
            initial = set() if q in clean else {frozenset({q})}
            delta = {frozenset(term) for term in value['monomials']} ^ initial
            if delta:
                deltas.append(delta)
        nonlinear = [p for p in deltas if any(len(t) > 1 for t in p)]
        nonlinear_support = {q for poly in nonlinear for term in poly for q in term}
        all_support = {q for poly in deltas for term in poly for q in term}
        if nonlinear and not all_support.isdisjoint(semantics.get('modified_wires',())):
            preferred = ['affine_transport_factorization']
        elif len(nonlinear) > 1 and remaining >= 5:
            # Preserve the large-support factor -> measured-cleanup path under
            # tight budgets; truth-table construction cannot seed that path.
            preferred = (['algebraic_boundary_synthesis','measurement_debt_lowering','architecture_portfolio']
                         if len(nonlinear_support)>8 else ['architecture_portfolio'])
        elif len(nonlinear) > 1 and remaining < 5:
            preferred = ['output_basis_search', 'consumer_functional_decomposition', 'algebraic_boundary_synthesis']
        elif remaining < 5 and nonlinear and max(map(len, nonlinear[0])) > 2:
            preferred = ['consumer_functional_decomposition','algebraic_boundary_synthesis']
        elif remaining < 5 and len(clean) < 2:
            preferred = ['algebraic_boundary_synthesis']
        else:
            preferred = []
    else:
        preferred = []
    rank = {name:i for i,name in enumerate(preferred)}
    return sorted(constructors, key=lambda c:rank.get(c.name,len(preferred)))


def _measured_boundary(circuit,context):
    from .measured_boolean_synthesis import measured_boundary_synthesis
    sources=[(key,context.get(key)) for key in ('joint_result','algebraic_result')
             if (context.get(key) or {}).get('status')=='constructed']
    if not sources:
        return dict(status='not_applicable',reason='no synthesized joint function for phase-debt-safe lowering')
    proposals=[]
    for name,graph in sources:
        proposals.append((name,measured_boundary_synthesis(circuit,**{name:graph},cost=context['cost'],
            resource_measure=context['resource_measure'],clean_inputs=context['clean_inputs'],
            internal_bits=context.get('internal_bits',()),classical_bit_offset=context['classical_bit_offset'],
            **{key:context[key] for key in ('max_operations','max_variables','max_terms','max_product_terms')})))
    verified=[(name,p) for name,p in proposals if p.get('status')=='constructed']
    if not verified:
        return dict(status='unknown',reason='no joint or algebraic DAG lowering passed',formulations=dict(proposals))
    name,best=min(verified,key=lambda row:Formula(context['cost'])(context['resource_measure'](km.Circuit(row[1]['candidate']))))
    return dict(best,selected_graph_source=name,formulation_summaries=[dict(source=n,status=p.get('status'),
        reason=p.get('reason'),resource_bill=p.get('resource_bill')) for n,p in proposals])


def _terminal_lookup(circuit, context):
    from .lookup_structure import terminal_cache_fusion
    return terminal_cache_fusion(circuit, context['clean_inputs'])


_SURROGATE_CONSTRUCTORS = {
    'joint_boundary_synthesis', 'joint_storage_recomputation', 'algebraic_boundary_synthesis',
    'affine_representation_search', 'consumer_functional_decomposition', 'output_basis_search',
    'architecture_portfolio', 'affine_transport_factorization', 'measurement_debt_lowering',
    'mixed_observable_search', 'terminal_lookup_fusion',
}


def default_registry():
    registry = ConstructorRegistry()
    registry.register('joint_boundary_synthesis',_joint_boundary,
                      'Search a shared XOR/AND implementation of the complete consumer function; lower and restore only necessary intermediate products.')
    registry.register('joint_storage_recomputation',_scheduled_boundary,
                      'Search complete compute-consume-cleanup schedules of the recovered joint function, lowering and verifying qubit versus recomputation alternatives.')
    registry.register('algebraic_boundary_synthesis',_algebraic_boundary,
                      'Recover and jointly factor complete output polynomials without enumerating a truth table; lower shared products with restoration.')
    registry.register('affine_representation_search',_representation_boundary,
                      'Infer invertible input encodings, synthesize in each representation, and price conversions and restoration in the parent.')
    registry.register('measurement_debt_lowering',_measured_boundary,
                      'Replace eligible joint-product cleanup by measured erasure with proved product identities and exact phase repayment.')
    registry.register('consumer_functional_decomposition', _functional_decomposition,
                      'Identify equivalent producer assignments across all consumers and synthesize their class bit.')
    registry.register('output_basis_search', _output_basis_search,
                      'Search for a shared encoding of output deltas and restore every dirty target.')
    registry.register('architecture_portfolio', _architecture_portfolio,
                      'Jointly vary input encoding, output encoding and shared nonlinear factor graphs, then verify diverse native finalists.')
    registry.register('affine_transport_factorization', _affine_transport,
                      'Factor an invertible affine transport out of an in-place update and jointly synthesize its residual.')
    registry.register('fractional_observable_search',_fractional_observable,
                      'Synthesize a restored fractional phase observable, including the price of rotations and restoration.')
    registry.register('factored_observable_search', _observable_search,
                      'Factor the complete sign observable and eliminate priced phase-only computation.')
    registry.register('mixed_observable_search', _mixed_observable_search,
                      'Decompose a value permutation with input-dependent sign into a proved entry observable and a value update.')
    registry.register('observable_phase_elimination', _phase,
                      'Replace a restored materialized phase word with its exact observable phase; retire declared clean trailing workspace.')
    registry.register('direct_consumer_synthesis', _direct_consumer,
                      'Compose producer, consumer and restoration semantics before emitting output updates.')
    registry.register('bilinear_tensor_decomposition',_tensor,
                      'Search factored multi-output products and emit a complete reversible accumulator using declared clean helpers.')
    registry.register('guided_bilinear_candidate_search',_guided_tensor,
                      'Use the recovered tensor and residual constraints to propose factorizations, then native-batch and exactly check finalists.')
    registry.register('terminal_lookup_fusion',_terminal_lookup,
                      'Keep shared lookup cache products and write copy-only terminal products directly into their consumers.')
    registry.register('measured_lookup_resynthesis',_measured_lookup,
                      'Factor supported lookup accumulators while preserving HMR measurements, dirty outputs and exact phase repairs.')
    return registry


def _check(original, candidate, clean, limits, *, internal_bits=()):
    # Common public wire indices are fixed. Omitted private inputs are promised
    # zero and must be restored. ANF equality checks implicit identity outputs.
    removed = set(range(candidate.num_qubits, original.num_qubits))
    if candidate.num_qubits > original.num_qubits or not removed <= set(clean):
        return dict(equivalent=None, reason='undeclared_or_expanded_boundary_layout')
    def registers(c):
        names={};members={}
        for line in str(c).splitlines():
            words=shlex.split(line)
            if words[0]=='REGISTER' and len(words)>2:names[words[1]]=words[2]
            elif words[0]=='APPEND_TO_REGISTER':members.setdefault(words[2],[]).append(words[1])
        return {name:tuple(members.get(index,[])) for index,name in names.items()}
    source_registers,candidate_registers=registers(original),registers(candidate)
    if any(candidate_registers.get(name)!=members for name,members in source_registers.items()):
        return dict(equivalent=False,reason='public_named_register_binding_changed')
    from .fractional_equivalence import has_fractional_phase, equivalent_fractional_boundaries
    if has_fractional_phase(original) or has_fractional_phase(candidate):
        return equivalent_fractional_boundaries(original,candidate,clean_inputs=clean,**limits)
    def body(c):
        return [line for line in str(c).splitlines() if line.split()[0] not in {'REGISTER','APPEND_TO_REGISTER'}]
    check = equivalent_windows(body(original),body(candidate),clean_inputs=clean, **limits)
    if check.get('equivalent') is not True and internal_bits:
        from .measurement_equivalence import equivalent_independent_outputs
        check=equivalent_independent_outputs(body(original),body(candidate),clean_inputs=clean,
            internal_bits=internal_bits,max_operations=limits['max_operations'],max_variables=limits['max_variables'],
            max_nodes=max(2,4*limits['max_terms']),max_apply_steps=32*limits['max_product_terms'])
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
                     max_product_terms=65536, archive=None, width=None,
                     scan_range=None, range_unit='operation', accounting='static',
                     external_distribution=None, _resource_measure=None,deep_only=False,
                     internal_bits=(),_classical_bit_offset=None,fast=False,
                     search_seconds=None,trial_seconds=None,cancel_file=None,_deadline=None,max_finalists=24,
                     goal='discoveries',max_architectures=1024,domain=None,max_assignments=65536,
                     max_and=3,joint_timeout=None,joint_max_states=10000,joint_max_expansions=100000):
    """Construct and verify complete-boundary candidates; preserve every outcome.

    The connected skill invents further mechanisms when bounded constructors
    lack a witness. This command invokes no remote AI or untrusted code.
    """
    if min(rounds,beam,max_candidates,max_operations,max_variables,max_terms,max_product_terms)<=0:
        raise StrataError('restructure budgets must be positive')
    from .search_screen import positive_seconds, stop_reason, exact_features, inapplicable, at_zero_floor
    positive_seconds(search_seconds,'search_seconds')
    positive_seconds(trial_seconds,'trial_seconds')
    positive_seconds(joint_timeout,'joint_timeout')
    from .floors import _validate_search_limits
    _validate_search_limits(max_and, joint_timeout if joint_timeout is not None else 1.0,
                            joint_max_states, joint_max_expansions)
    if type(max_finalists) is not int or max_finalists < 0:
        raise StrataError('max_finalists must be a nonnegative integer')
    if goal not in ('discoveries','best-cost') or type(max_architectures) is not int or max_architectures<1:
        raise StrataError('use goal discoveries or best-cost and a positive architecture budget')
    if cancel_file is not None and search_seconds is None and trial_seconds is None:
        trial_seconds=4.0
    source=Path(path).resolve();root=Path(out).resolve();root.mkdir(parents=True,exist_ok=True)
    deadline=_deadline if _deadline is not None else (time.monotonic()+search_seconds if search_seconds else None)
    if source in {root/'restructure.json',root/'best.kmx'}:
        raise StrataError('restructure output would overwrite source')
    circuit=load_circuit(source)
    private=tuple(sorted(set(internal_bits)))
    if any(type(bit)is not int or not 0<=bit<circuit.num_bits for bit in private):
        raise StrataError('internal bits must be existing classical bits explicitly private at this boundary')
    classical_offset=circuit.num_bits if _classical_bit_offset is None else _classical_bit_offset
    if type(classical_offset)is not int or classical_offset<circuit.num_bits:
        raise StrataError('classical bit offset must reserve every existing parent bit')
    if width is not None:
        if type(width) is not int or width < circuit.num_qubits or width < 1:
            raise StrataError('width must be a positive parent-layout size at least the touched circuit width')
        if width > circuit.num_qubits:
            metadata=_boundary_metadata(circuit)
            extra=_boundary_metadata(circuit,width-1,'strata_parent_layout')[len(metadata):]
            circuit=km.Circuit(str(circuit)+'\n'+'\n'.join(extra))
    clean=tuple(sorted(set(clean_inputs)))
    if any(type(q)is not int or not 0<=q<circuit.num_qubits for q in clean):
        raise StrataError('clean inputs must be physical wires of the supplied circuit; use width for an explicit larger parent layout')
    promise=None
    if domain is not None:
        if scan_range is not None or len(circuit)>max_operations:
            raise StrataError('a search promise applies at artifact entry; supply a bounded complete circuit or increase --max-operations')
        from .promised_domain import load_search_domain, check_promised_candidate, promised_domain_synthesis
        promise=load_search_domain(domain,circuit,clean_inputs=clean,max_assignments=max_assignments,max_operations=max_operations)
    if scan_range is not None or len(circuit)>max_operations:
        from .episode_search import restructure_episode, restructure_regions
        options=dict(clean_inputs=clean,cost=cost,registry=registry,rounds=rounds,beam=beam,
                     max_candidates=max_candidates,max_operations=max_operations,
                     max_variables=max_variables,max_terms=max_terms,max_product_terms=max_product_terms,archive=archive,
                     accounting=accounting,external_distribution=external_distribution,internal_bits=private)
        options.update(deep_only=deep_only,fast=fast,trial_seconds=trial_seconds,search_seconds=search_seconds,
                       cancel_file=cancel_file,_deadline=deadline,max_finalists=max_finalists,goal=goal,max_architectures=max_architectures,
                       max_and=max_and,joint_timeout=joint_timeout,
                       joint_max_states=joint_max_states,joint_max_expansions=joint_max_expansions)
        if scan_range is not None:
            return restructure_episode(source,circuit,root,scan_range=scan_range,range_unit=range_unit,**options)
        return restructure_regions(source,circuit,root,**options)
    from .accounting import measured_resources
    measure=_resource_measure or (lambda c:measured_resources(c,accounting=accounting,external_distribution=external_distribution))
    formula=Formula(cost);frontend=Kickmix();limits=dict(max_operations=max_operations,max_variables=max_variables,
                                                       max_terms=max_terms,max_product_terms=max_product_terms)
    built_in=registry is None
    registry=registry or default_registry()
    memory_path=Path(archive).resolve() if archive else root/'mechanisms.json'
    memory=json.loads(memory_path.read_text()) if memory_path.exists() else dict(version=1,templates=[],statistics={})
    _validate_archive(memory)
    constructors=list(registry.constructors.values())
    if deep_only and registry is not None:
        constructors=[c for c in constructors if c.name in {'joint_boundary_synthesis','joint_storage_recomputation',
            'algebraic_boundary_synthesis','affine_representation_search','measurement_debt_lowering',
            'consumer_functional_decomposition','output_basis_search','factored_observable_search','mixed_observable_search',
            'architecture_portfolio','affine_transport_factorization','fractional_observable_search','terminal_lookup_fusion'}]
    if promise is not None:
        def construct_on_promise(parent, context):
            return promised_domain_synthesis(parent,promise,max_assignments=max_assignments,
                try_joint=True, **_joint_options(context), **_structural_options(context))
        constructors=[Construction('promised_domain_synthesis',construct_on_promise,'Synthesize on the complete declared input promise')]
    elif not deep_only:
        constructors += [Construction('learned_'+digest(t)[:12],_template_constructor(t),'Replayed learned gate template with checked predicates') for t in memory['templates']]
    def state(c, parent, mechanism,private_bits=()):
        text=str(c);key=hashlib.sha256(text.encode()).hexdigest();resources=measure(c)
        (root/f'{key}.kmx').write_text(text)
        return dict(id=key,path=str(root/f'{key}.kmx'),resources=resources,cost=formula(resources),parent=parent,
                    mechanism=mechanism,private_bits=sorted(private_bits))
    baseline=state(circuit,None,'supplied_baseline',private);states=[baseline];frontier=[(circuit,baseline)];attempts=[]
    by_id={baseline['id']:baseline}
    structural_ids=set()
    qualifications={}; extra_finalists=0
    screened=[]; checked={}; proof_reuses=0; stopped=None
    started=time.monotonic();boundary_receipts={};lookup_tables={};measurement_reductions={}
    from .fractional_equivalence import has_fractional_phase
    def verify(parent,candidate,private_bits):
        if promise is not None:
            return check_promised_candidate(circuit,candidate,promise,clean_inputs=clean,max_assignments=max_assignments,
                internal_bits=private_bits,measurement_mode='independent-map' if private_bits else 'coupled',**limits)
        return _check(parent,candidate,clean,limits,internal_bits=private_bits)
    for generation in range(rounds):
        following=[]
        for parent,parent_state in frontier:
            stopped=stop_reason(deadline,cancel_file)
            if stopped:break
            if fast and (goal=='best-cost' or at_zero_floor(baseline['cost'],cost)) and any(at_zero_floor(s['cost'], cost) for s in states
                            if not deep_only or s['id']==baseline['id'] or s['id'] in structural_ids):
                stopped='verified_nonnegative_objective_floor'
                break
            parent_lines=str(parent).splitlines()
            metadata_count=sum(line.split()[0] in {'REGISTER','APPEND_TO_REGISTER'} for line in parent_lines)
            has_classical = any(
                words[0] in {'HMR','PUSH_CONDITION','POP_CONDITION'} or words[0].startswith('BIT_') or 'if' in words
                for line in parent_lines for words in [line.split()]
                if words[0] not in {'REGISTER','APPEND_TO_REGISTER','INFO_TAG'})
            semantic_limits=dict(limits,max_operations=max_operations+metadata_count)
            semantics=analyze_complete(parent_lines,clean_inputs=clean,**semantic_limits)
            fractional_phase=has_fractional_phase(parent)
            boundary_receipts[parent_state['id']]=semantics
            context=dict(semantics=semantics,clean_inputs=tuple(q for q in clean if q<parent.num_qubits),cost=cost,accounting=accounting,
                         internal_bits=tuple(parent_state['private_bits']),
                         classical_bit_offset=max(classical_offset,parent.num_bits),resource_measure=measure,
                         max_architectures=max_architectures,extra_finalist_budget=max_finalists-extra_finalists,
                         classical_trajectory=has_classical,fractional_phase=fractional_phase,
                         max_and=max_and,joint_timeout=joint_timeout,joint_max_states=joint_max_states,
                         joint_max_expansions=joint_max_expansions,**semantic_limits)
            if not deep_only and semantics['status']=='exact_factored' and semantics.get('phase_zero'):
                from .factored_semantic import lift_lookup_table
                table=lift_lookup_table(parent,clean_inputs=clean,max_operations=semantic_limits['max_operations'],
                    max_variables=max_variables,max_nodes=max_terms,max_apply_steps=32*max_product_terms)
                context['lookup_table']=table
                lookup_tables[parent_state['id']]=table
            proposal_parent = parent
            if has_classical and built_in and promise is None and not fractional_phase:
                from .lookup_structure import measured_value_surrogate
                pure, reduction = measured_value_surrogate(parent, context['clean_inputs'],
                                                           context['internal_bits'], semantic_limits)
                measurement_reductions[parent_state['id']] = reduction
                if pure is not None:
                    proposal_parent = pure
                    semantics = analyze_complete(str(pure).splitlines(), clean_inputs=clean, **semantic_limits)
                    context.update(semantics=semantics, classical_trajectory=False)
                else:
                    context['classical_blocker'] = reduction
            # A sign-only observable deserves the first attempt even in an
            # episode assigned a single candidate. Keep the graph producers
            # before their schedule/cleanup consumers on value boundaries.
            ordered=(_order_deep_search(constructors,semantics,set(context['clean_inputs']),max_candidates-len(attempts))
                     if deep_only else constructors)
            features=exact_features(semantics,set(context['clean_inputs'])) if fast and built_in else {}
            if fractional_phase and built_in and promise is None:
                ordered=sorted(ordered,key=lambda ctor:ctor.name!='fractional_observable_search')
            for constructor in ordered:
                if len(attempts)>=max_candidates:break
                stopped=stop_reason(deadline,cancel_file)
                if stopped:break
                if fast and built_in:
                    reason=inapplicable(constructor.name,features,context)
                    if reason:
                        screened.append(dict(parent=parent_state['id'],round=generation,mechanism=constructor.name,reason=reason,
                            **({'blocking_instruction': context['classical_blocker']} if 'classical trajectory' in reason
                               and 'classical_blocker' in context else {})))
                        continue
                attempt=dict(round=generation,parent=parent_state['id'],mechanism=constructor.name)
                trial_start=time.monotonic()
                trial_parent = proposal_parent if constructor.name in _SURROGATE_CONSTRUCTORS else parent
                if trial_parent is not parent:
                    attempt['construction_source'] = 'verified_measured_value_surrogate'
                try:
                    if trial_seconds is not None or deadline is not None or cancel_file is not None:
                        from .search_worker import run_bounded_constructor
                        remaining=deadline-time.monotonic() if deadline is not None else None
                        if remaining is not None and remaining <= 0:
                            stopped='search_time_budget'
                            break
                        budgets=[x for x in (remaining,trial_seconds) if x is not None]
                        context['constructor_time_budget']=min(budgets) if budgets else None
                        proposal,updates,_=run_bounded_constructor(constructor.propose,trial_parent,context,
                            timeout=context['constructor_time_budget'],cancel_file=cancel_file)
                        context.update(updates)
                    else:
                        proposal=constructor.propose(trial_parent,context)
                    attempt['construction']=proposal
                    if proposal.get('status')!='constructed':
                        attempt.update(status='not_constructed',reason=proposal.get('reason','preconditions unavailable'))
                    else:
                        candidate=km.Circuit(proposal['candidate'])
                        if len(candidate)>max_operations:
                            raise StrataError('candidate executable operation budget exceeded')
                        introduced=proposal.get('introduced_private_bits',[])
                        if (not isinstance(introduced,(list,tuple)) or any(type(b)is not int or
                                not context['classical_bit_offset']<=b<candidate.num_bits for b in introduced)):
                            raise StrataError('constructor private bits collide with an existing parent bit or lie outside candidate')
                        child_private=set(parent_state['private_bits'])|set(introduced)
                        proof_key=(parent_state['id'],hashlib.sha256(str(candidate).encode()).digest(),
                                   tuple(sorted(child_private)))
                        if fast and proof_key in checked:
                            check=checked[proof_key]
                            proof_reuses+=1
                        else:
                            check=verify(parent,candidate,child_private)
                            if fast and check.get('equivalent') is True:
                                checked[proof_key]=check
                        attempt['check']=check
                        attempt['status']='verified' if check.get('equivalent')is True else 'rejected' if check.get('equivalent')is False else 'unknown'
                        if check.get('equivalent')is True:
                            child=state(candidate,parent_state['id'],constructor.name,child_private)
                            qualification=_qualifying_mechanism(proposal)
                            attempt['structural_qualification']=qualification
                            if child['id']!=baseline['id'] and (parent_state['id'] in structural_ids or qualification):
                                structural_ids.add(child['id'])
                                kinds=qualifications.setdefault(child['id'],set())
                                kinds.update(qualifications.get(parent_state['id'],()))
                                if qualification:kinds.add(qualification)
                            attempt.update(candidate=child['id'],cost_delta=child['cost']-parent_state['cost'],
                                           cumulative_cost_delta=child['cost']-baseline['cost'])
                            if child['id'] not in by_id:
                                states.append(child);by_id[child['id']]=child;following.append((candidate,child))
                            if promise is None and child['cost']<parent_state['cost']:
                                learned=_learn_template(parent,candidate,clean,constructor.name)
                                if learned and learned not in memory['templates']:memory['templates'].append(learned)
                            # Retain distinct, independently checked structural finalists even
                            # when the constructor has already returned a cheaper sibling.
                            if deep_only:
                                for alternative in proposal.get('alternatives',()):
                                    if extra_finalists>=max_finalists or stop_reason(deadline,cancel_file):break
                                    if not isinstance(alternative,dict) or not isinstance(alternative.get('candidate'),str):
                                        continue
                                    kind=_qualifying_mechanism(alternative)
                                    if not kind:continue
                                    alt=km.Circuit(alternative['candidate'])
                                    if len(alt)>max_operations:continue
                                    key=hashlib.sha256(str(alt).encode()).hexdigest()
                                    bits=alternative.get('introduced_private_bits',[])
                                    if not isinstance(bits,(list,tuple)):continue
                                    if any(type(b)is not int or not context['classical_bit_offset']<=b<alt.num_bits for b in bits):continue
                                    alt_private=set(parent_state['private_bits'])|set(bits)
                                    if key in by_id:
                                        known=by_id[key]
                                        if known['cost'] < baseline['cost'] and set(known['private_bits'])==alt_private:
                                            # The same bytes already passed the boundary checker under the same privacy contract.
                                            structural_ids.add(key)
                                            qualifications.setdefault(key,set()).add(kind)
                                        continue
                                    if formula(measure(alt))>=baseline['cost']:continue
                                    extra_finalists+=1
                                    proof=verify(parent,alt,alt_private)
                                    if proof.get('equivalent') is not True:continue
                                    finalist=state(alt,parent_state['id'],constructor.name,alt_private)
                                    finalist['check']=proof
                                    finalist['structural_qualification']=kind
                                    states.append(finalist);by_id[key]=finalist;structural_ids.add(key)
                                    qualifications[key]=qualifications.get(parent_state['id'],set()) | {kind}
                                    following.append((alt,finalist))
                except (StrataError,ValueError,RuntimeError) as exc:
                    attempt.update(status='unknown',reason=str(exc))
                attempt['inference_scope']='one construction under a frozen boundary; no entire mechanism is closed'
                attempt['seconds']=time.monotonic()-trial_start
                attempts.append(attempt)
                stats=memory['statistics'].setdefault(constructor.name,dict(attempts=0,verified=0,gains=0))
                stats['attempts']+=1;stats['verified']+=attempt['status']=='verified';stats['gains']+=attempt.get('cost_delta',0)<0
                if fast and goal=='best-cost' and attempt.get('candidate') and (not deep_only or attempt['candidate'] in structural_ids):
                    if at_zero_floor(child['cost'],cost):
                        stopped='verified_nonnegative_objective_floor'
                        break
            if stopped:break
        if stopped:break
        # Diversity first: retain valid alternative representations even when
        # they cost more. Archive is complete; beam only bounds the next round.
        from collections import defaultdict, deque
        groups=defaultdict(deque)
        for row in sorted(following,key=lambda row:(row[1]['cost'],row[1]['id'])):
            groups[row[1]['mechanism']].append(row)
        frontier=[]
        while len(frontier)<beam and any(groups.values()):
            for queue in groups.values():
                if queue and len(frontier)<beam:
                    frontier.append(queue.popleft())
        if not frontier or len(attempts)>=max_candidates:break
    eligible=[s for s in states if not deep_only or s['id']==baseline['id'] or s['id'] in structural_ids]
    best=min(eligible,key=lambda s:s['cost'])
    (root/'best.kmx').write_text(Path(best['path']).read_text())
    memory_path.parent.mkdir(parents=True,exist_ok=True);memory_path.write_text(json.dumps(memory,indent=2)+'\n')
    result=dict(schema_version='3',status='verified_improvement' if best['cost']<baseline['cost'] else 'no_gain',
                source=str(source),source_sha256=file_hash(source),baseline=baseline,best=best,
                search_goal='verified_structural_alternatives' if deep_only else 'all_supported_constructions',
                cost_delta=best['cost']-baseline['cost'],states=states,pareto_frontier=_pareto(states),attempts=attempts,
                boundary_receipts=boundary_receipts,lookup_tables=lookup_tables,measurement_reductions=measurement_reductions,
                archive=str(memory_path),learned_template_count=len(memory['templates']),seconds=time.monotonic()-started,
                contract=dict(clean_inputs=clean,internal_bits=private,public_wires='every other physical wire, fixed indices',outputs='all physical outputs and exact sign equal to supplied baseline; only proved restored clean wires may leave the layout',
                              measurement_scope='branch-preserving or proved outcome-independent basis/sign maps; only explicitly private or fresh compiler bits may be omitted; not generic channel equivalence'),
                objective=dict(expression=cost,accounting=baseline['resources']['accounting'],
                               external_distribution=external_distribution,
                               scope='parent_context' if _resource_measure else 'supplied_circuit'),official_verifier_run=False,
                learning_scope='executable checked templates and observed constructor statistics; no model weight training or universal discovery claim',
                next_agent_questions=['Which costly materialization exists only to feed a later observable?',
                    'Can an alternate representation remove a dependency across the complete producer-consumer-return boundary?',
                    'Which exact support relation can produce a reversible encoder and a priced consumer adapter?',
                    'Which bounded or failed formulation could reopen under a different construction?'],
                budgets=dict(rounds=rounds,beam=beam,max_candidates=max_candidates,
                             max_and=max_and,joint_timeout=joint_timeout,joint_max_states=joint_max_states,
                             joint_max_expansions=joint_max_expansions,**limits))
    if promise is not None:
        result['contract'].update(domain=promise,domain_hash=promise['domain_hash'],all_input_equivalence=False,
            outputs='all physical quantum and public classical outputs, clean restoration and phase on the declared promise')
        result['search_goal']='verified_alternatives_on_declared_input_domain'
    elif has_fractional_phase(circuit):
        result['contract']['outputs']='all physical values and phase modulo one turn on the declared clean boundary'
    result['structural_gains']=[dict(candidate=a['candidate'],mechanism=a['mechanism'],
        cost_delta=a['cost_delta'],qualification=a['structural_qualification'],witness=a['construction']['structural_witness'])
        for a in attempts if a.get('status')=='verified' and a.get('cost_delta',0)<0
        and a.get('structural_qualification')]
    result['search_control']=dict(fast=fast,screened=screened,screened_count=len(screened),
        verified_proof_reuses=proof_reuses,stop_reason=stopped or stop_reason(deadline,cancel_file),
        trial_seconds=trial_seconds,search_seconds=search_seconds,
        timed_out=sum(a.get('construction',{}).get('reason')=='constructor_timeout' for a in attempts),
        cancelled=sum(a.get('construction',{}).get('reason')=='constructor_cancelled' for a in attempts))
    from .discovery_yield import summarize, constructor_comparison
    result['discovery_yield']=summarize(states,baseline,qualifications,classical_offset,clean)
    result['discovery_yield'].update(extra_finalists_checked=extra_finalists,max_extra_finalists=max_finalists,
                                     goal=goal,max_architectures_per_portfolio=max_architectures)
    result['discovery_yield']['constructor_comparison'] = constructor_comparison(attempts, baseline['cost'])
    result['discovery_yield']['architecture_trials'] = [
        dict(parent=attempt['parent'], **{key:ledger[key] for key in (
            'proposed_attempts','unique_formulations','unique_representations','unique_product_graphs',
            'not_lowerable','selected_finalists','lowered','verified','native_improvements','categories','stopped_reason')})
        for attempt in attempts if attempt['mechanism']=='architecture_portfolio'
        for ledger in [attempt.get('construction',{}).get('search')] if ledger]
    from .store import Store
    store=Store(root/'.strata')
    try:
        result['parent_sha256']=baseline['id']
        result['artifacts']={s['id']:store.blob(Path(s['path']).read_bytes()) for s in states}
        result['code']={name:store.blob((Path(__file__).parent/name).read_bytes())
                        for name in ('restructure.py','semantic.py','factored.py','factored_semantic.py','inference.py','tensor.py','candidate_search.py','domains.py','lookup.py',
                            'boolean_synthesis.py','algebraic_synthesis.py','representation_synthesis.py','scheduled_boolean_synthesis.py',
                            'measured_boolean_synthesis.py','measurement_equivalence.py','pebbling.py','accounting.py','episode_search.py',
                            'functional_decomposition.py','output_basis_search.py','observable_search.py','symbolic_boolean_graph.py',
                            'mixed_observable.py','search_screen.py','search_worker.py','architecture_portfolio.py','discovery_yield.py',
                            'affine_transport.py','fractional_observable.py','fractional_equivalence.py','promised_domain.py',
                            'promised_synthesis.py','lookup_structure.py')}
        result['mechanism_archive_blob']=store.blob(memory_path.read_bytes())
        result['record']=store.put('restructure_run',result)
    finally:
        store.db.close()
    (root/'restructure.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
