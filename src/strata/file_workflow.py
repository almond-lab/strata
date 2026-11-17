"""Circuit-file entry points, without a task-specific Python adapter.

Exact replacement of arbitrary-input subfunctions can preserve the original
circuit without knowing its contest specification. It does not establish the
validity or optimality of the original, or relax its accuracy contract.
"""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import importlib.metadata

from .kickmix import Kickmix, instructions
from .model import Contract, Register, Region, StrataError, digest
from .store import Store


def contract_tree(circuit, lifted):
    """Use recovered functions as references; unknown spans stay opaque."""
    lines = str(circuit).splitlines()
    if not lines:
        raise StrataError('cannot infer contracts from an empty circuit')
    children = []
    from .semantic import semantic_function
    for row in lifted['regions']:
        sem = row['semantics']
        registers = tuple(Register(f'input_q{q}', (q,), meaning=('declared zero boundary bit' if q in sem.get('clean_input_assumptions',()) else 'arbitrary boundary bit'))
                          for q in sem['inputs'])
        reference = digest(semantic_function(sem))
        contract = Contract(row['kind'], registers, 'anf:' + reference)
        children.append(Region(row['id'], row['start'], row['stop'], contract,
                               kind=row['kind'], evidence=sem['evidence']))
    for row in lifted['unknown']:
        start, stop = row['start'], row['stop']
        source = '\n'.join(lines[start:stop])
        # An opaque span's reference is the source, not an invented function.
        qs = sorted({q for _, _, _, wires in instructions(source) for q in wires})
        registers = tuple(Register(f'q{q}', (q,)) for q in qs)
        contract = Contract('unknown', registers, 'opaque:' + hashlib.sha256(source.encode()).hexdigest())
        children.append(Region(f'opaque-{start}-{stop}', start, stop, contract,
                               kind='opaque', evidence=row['reason']))
    root_contract = Contract('supplied_circuit',
                             (Register('state', tuple(range(circuit.num_qubits))),) if circuit.num_qubits else (),
                             'artifact:' + hashlib.sha256(str(circuit).encode()).hexdigest())
    return Region('root', 0, len(lines), root_contract,
                  tuple(sorted(children, key=lambda r:r.start)),
                  kind='file_circuit', evidence='recovered subfunctions; task-level intent not inferred')


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _source(path, out, *, writes_candidate=False):
    source = Path(path).resolve()
    if not source.is_file():
        raise StrataError('supply an individual .kmx/.kmb circuit file; bundles need their host acceptance configuration')
    root = Path(out).resolve()
    reports=('result.json','semantic-lift.json','contract-tree.json','profile.json','region-evidence.json')
    protected=reports+('candidate.kmx',) if writes_candidate else reports
    if any(source == root / name for name in protected) or (root/'analysis') in source.parents:
        raise StrataError('output would overwrite the supplied circuit')
    root.mkdir(parents=True, exist_ok=True)
    from .frontends import load_circuit
    circuit = load_circuit(source)
    return source, root, circuit


def inspect_file(path, out, *, cost='Q*(toffoli+t/4)', max_window=256, max_variables=64, max_terms=4096,
                 clean_inputs=(),scan_range=None,progress=None,growth_window=None,range_unit='operation'):
    from .semantic import lift_regions
    from .file_profile import profile_regions
    source, root, circuit = _source(path, out)
    if scan_range is not None or source.stat().st_size>=2_000_000:
        from .streaming import scan_circuit
        start,stop=scan_range if scan_range else (0,None)
        result=scan_circuit(circuit,root/'stream',source=source,start=start,stop=stop,
                           range_unit=range_unit,
                           max_window=max_window if growth_window is None else growth_window,max_variables=max_variables,max_terms=max_terms,progress=progress,clean_inputs=clean_inputs)
        _write_json(root/'result.json',result)
        return result
    lift = lift_regions(circuit, max_window=max_window, max_variables=max_variables, max_terms=max_terms,clean_inputs=clean_inputs)
    tree = contract_tree(circuit, lift)
    profile=profile_regions(circuit,lift,root/'analysis',formula=cost)
    from .region_evidence import record_region_evidence
    evidence = record_region_evidence(source, circuit, tree, lift, Store(root / '.strata'))
    _write_json(root / 'region-evidence.json', evidence)
    _write_json(root / 'semantic-lift.json', lift)
    _write_json(root / 'contract-tree.json', asdict(tree))
    _write_json(root / 'profile.json',profile)
    result = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  source_range_unit='text-line',
                  artifact_sha256=hashlib.sha256(str(circuit).encode()).hexdigest(),
                  resources=Kickmix().measured(circuit), coverage=lift['coverage'],
                  regions=len(lift['regions']), opaque_spans=len(lift['unknown']),
                  contract_tree=str(root / 'contract-tree.json'), semantic_lift=str(root / 'semantic-lift.json'),
                  profile=str(root / 'profile.json'), views=profile['views'],
                  region_evidence=str(root / 'region-evidence.json'), region_evidence_record=evidence['record'],
                  evidence_summary=evidence['summary'],
                  task_adapter_required=False,
                  limitations=['Algebraic region contracts are recovered; arithmetic algorithms and task intent may remain unknown.',
                               'No clean ancillas are inferred. Unsupported operations and bounded regions remain explicit.'])
    _write_json(root / 'result.json', result)
    return result


def optimize_file(path, out, *, cost='Q*(toffoli+t/4)', max_window=256, max_variables=64, max_terms=4096,max_candidates=64,clean_inputs=(),accounting='static',external_distribution=None):
    from .semantic import lift_regions
    from .file_profile import profile_regions
    from .file_optimizer import optimize_circuit, verify_optimization
    source, root, circuit = _source(path, out,writes_candidate=True)
    lift = lift_regions(circuit, max_window=max_window, max_variables=max_variables, max_terms=max_terms,clean_inputs=clean_inputs)
    tree = contract_tree(circuit, lift)
    profile=profile_regions(circuit,lift,root/'analysis',formula=cost)
    candidate, result = optimize_circuit(circuit, cost, max_operations=max_window,
                                          max_wires=max_variables, max_terms=max_terms,
                                          region_order=profile['candidate_order'],max_candidates=max_candidates,clean_inputs=clean_inputs,
                                          accounting=accounting,external_distribution=external_distribution)
    replay = verify_optimization(circuit, candidate, result, max_operations=max_window,
                                 max_wires=max_variables, max_terms=max_terms,clean_inputs=clean_inputs,
                                 accounting=accounting,external_distribution=external_distribution)
    if replay.get('equivalent') is not True:
        raise StrataError('candidate failed independent replacement replay; no output candidate retained')
    store = Store(root / '.strata')
    from .region_evidence import record_region_evidence
    evidence = record_region_evidence(source, circuit, tree, lift, store)
    _write_json(root / 'region-evidence.json', evidence)
    original_hash = store.blob(str(circuit).encode())
    candidate_hash = store.blob(str(candidate).encode())
    code = {name:store.blob((Path(__file__).parent / name).read_bytes())
            for name in ('semantic.py', 'file_optimizer.py', 'file_workflow.py', 'region_evidence.py', 'frontends.py')}
    certificate = store.put('file_equivalence', dict(original=original_hash, candidate=candidate_hash,
                            evidence=replay, region_evidence=evidence['record'], code=code, kickmix_version=importlib.metadata.version('kickmix')))
    result.update(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  source_contract_tree=digest(tree), coverage=lift['coverage'],
                  equivalence_certificate=certificate, replay=replay,
                  region_evidence=str(root / 'region-evidence.json'), region_evidence_record=evidence['record'],
                  evidence_summary=evidence['summary'],
                  task_adapter_required=False, official_verifier_run=False,
                  score_status='static file objective only; contest acceptance and portfolio score not measured')
    handle = store.put('file_optimization', result)
    result['record'] = handle
    (root / 'candidate.kmx').write_text(str(candidate))
    _write_json(root / 'semantic-lift.json', lift)
    _write_json(root / 'contract-tree.json', asdict(tree))
    _write_json(root / 'profile.json',profile)
    _write_json(root / 'result.json', result)
    return dict(status=result['status'], source=str(source), candidate=str(root / 'candidate.kmx'),
                result=str(root / 'result.json'), baseline=result['baseline'], optimized=result['candidate'],
                equivalence_certificate=certificate, replay=replay.get('equivalent'),
                region_evidence=str(root / 'region-evidence.json'), region_evidence_record=evidence['record'],
                evidence_summary=evidence['summary'],
                coverage=lift['coverage'], record=handle, official_verifier_run=False,
                score_status=result['score_status'])
