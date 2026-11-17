"""Byte-bound symbolic receipts for fixed, source-less circuit regions.

These receipts establish recovered Boolean output and exact sign-phase maps
for arbitrary boundary inputs. They do not invent a smaller circuit, prove
host-domain acceptance, or certify unsupported operations.
"""
from dataclasses import asdict
from pathlib import Path
import hashlib
import tempfile
from .model import StrataError, digest
from .store import Store


CODE_FILES = ('semantic.py', 'factored.py', 'factored_semantic.py', 'streaming.py', 'frontends.py', 'kickmix.py', 'model.py', 'file_workflow.py', 'region_evidence.py')
LIMIT_KEYS = {'max_window', 'max_variables', 'max_terms', 'max_product_terms', 'max_ops', 'clean_inputs'}
SCOPE = dict(inputs='all arbitrary boundary bit assignments', clean_input_assumptions=[],
             phase='exact sign phase (-1)^Boolean_polynomial',
             instance='supplied fixed circuit region; no inferred scale model',
             task_correctness='not established', whole_circuit_certificate=False)


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def _receipts(circuit, tree, lifted, source_hash, artifact_hash):
    lines = str(circuit).splitlines()
    exact = {row['id']: row for row in lifted['regions']}
    unknown = {f"opaque-{row['start']}-{row['stop']}": row for row in lifted['unknown']}
    receipts = []
    for region in tree.children:
        content = '\n'.join(lines[region.start:region.stop]).encode()
        contract = asdict(region.contract)
        binding = dict(source_sha256=source_hash, artifact_sha256=artifact_hash,
                       region_sha256=_sha(content), start=region.start, stop=region.stop,
                       offset_encoding='canonical native instruction lines; zero-based half-open',
                       contract_hash=digest(contract), logical_contract_fingerprint=region.contract.fingerprint)
        receipt = dict(region=region.id, binding=binding, contract=contract)
        if region.id in exact:
            semantics = exact[region.id]['semantics']
            if semantics.get('status') != 'exact':
                raise StrataError('symbolic region lacks complete exact semantics')
            from .semantic import semantic_function
            payload = semantic_function(semantics)
            function = dict(inputs=semantics['inputs'], **payload)
            reference = digest(payload)
            if region.contract.reference != 'anf:' + reference:
                raise StrataError('recovered contract does not bind the symbolic function')
            scope = dict(SCOPE, clean_input_assumptions=semantics.get('clean_input_assumptions', []),
                         semantics=semantics.get('semantic_scope', 'native_basis_trajectories'))
            if semantics.get('clean_input_assumptions'):
                scope['inputs'] = 'listed boundary wires zero; all other boundary inputs arbitrary'
            if semantics.get('measurement_variables'):
                scope.update(measurement_outcomes='all independent forced outcomes with ordered event provenance',
                             quantum_channel_equivalence=False)
            if semantics.get('classical_inputs'):
                scope['classical_inputs'] = 'all arbitrary boundary classical bit assignments'
            receipt.update(status='exact', evidence_kind='symbolic', scope=scope,
                           symbolic_function=function, function_hash=digest(function),
                           effects=semantics.get('effects', {}),
                           method='bounded exact Boolean ANF symbolic execution',
                           claim='supplied region implements the recovered output and sign-phase polynomials')
        elif region.id in unknown:
            receipt.update(status='unknown', evidence_kind='unknown', reason=unknown[region.id]['reason'],
                           scope=dict(instance='supplied fixed circuit region', task_correctness='not established'),
                           claim='no exact symbolic evidence for this region')
        else:
            raise StrataError('contract region is missing from the semantic lift')
        receipts.append(receipt)
    return receipts


def record_region_evidence(source, circuit, tree, lifted, store):
    """Persist an already executed lift; this does not repeat semantic checks."""
    source = Path(source)
    if not isinstance(store, Store):
        store = Store(store)
    raw = source.read_bytes()
    canonical = str(circuit).encode()
    source_blob, artifact_blob = store.blob(raw), store.blob(canonical)
    limits = {key: value for key, value in lifted['limits'].items() if key in LIMIT_KEYS}
    code = {name: store.blob((Path(__file__).parent / name).read_bytes()) for name in CODE_FILES}
    receipts = _receipts(circuit, tree, lifted, source_blob, artifact_blob)
    report = dict(schema_version='1', kind='file_region_evidence',
                  source=dict(name=source.name, suffix=source.suffix, blob=source_blob, sha256=source_blob),
                  artifact=dict(blob=artifact_blob, sha256=artifact_blob, encoding='canonical native circuit'),
                  contract_tree=asdict(tree), contract_tree_hash=digest(tree), limits=limits, code=code,
                  regions=receipts,
                  summary=dict(symbolic=sum(r['evidence_kind'] == 'symbolic' for r in receipts),
                               unknown=sum(r['evidence_kind'] == 'unknown' for r in receipts)),
                  coverage=lifted['coverage'], metadata=lifted.get('metadata', []),
                  scope='fixed supplied region semantics only; no scalable generator or task verifier inferred')
    handle = store.put('file_region_evidence', report)
    return dict(record=handle, **report)


def replay_region_evidence(store, evidence):
    """Recompute symbolic evidence from pinned input bytes, ignoring cached maps.

    The same exact algebraic backend is rerun. This is replay of a symbolic
    derivation, not an independent algorithm or an official task verification.
    """
    if not isinstance(store, Store):
        store = Store(store)
    report = store.get(evidence, 'file_region_evidence') if isinstance(evidence, str) else dict(evidence)
    report.pop('record', None)
    if report.get('schema_version') != '1' or report.get('kind') != 'file_region_evidence':
        raise StrataError('unsupported region evidence format')
    if set(report.get('code', {})) != set(CODE_FILES):
        raise StrataError('region evidence lacks pinned backend code')
    for name, expected in report['code'].items():
        store.read_blob(expected)
        if _sha((Path(__file__).parent / name).read_bytes()) != expected:
            raise StrataError('symbolic evidence backend changed')
    source = report['source']
    raw = store.read_blob(source['blob'])
    if _sha(raw) != source['sha256']:
        raise StrataError('region evidence source binding mismatch')
    canonical = store.read_blob(report['artifact']['blob'])
    if _sha(canonical) != report['artifact']['sha256']:
        raise StrataError('region evidence artifact binding mismatch')
    from .frontends import load_circuit
    from .semantic import lift_regions
    from .file_workflow import contract_tree
    suffix = source['suffix']
    if suffix not in {'.kmx', '.kmb', '.ops', '.rev', '.jsonl', ''}:
        # Other extensions are supported as native text by load_circuit.
        suffix = '.kmx'
    with tempfile.TemporaryDirectory(prefix='strata-region-replay-') as directory:
        path = Path(directory) / ('source' + suffix)
        path.write_bytes(raw)
        circuit = load_circuit(path)
    if str(circuit).encode() != canonical:
        raise StrataError('source bytes do not reproduce the pinned canonical artifact')
    limits = report.get('limits', {})
    if set(limits) - LIMIT_KEYS or not {'max_window', 'max_variables', 'max_terms', 'max_product_terms'} <= set(limits):
        raise StrataError('symbolic evidence lacks explicit analysis budgets')
    lifted = lift_regions(circuit, **limits)
    tree = contract_tree(circuit, lifted)
    if digest(tree) != report.get('contract_tree_hash') or digest(report.get('contract_tree')) != digest(tree):
        raise StrataError('region evidence contract tree mismatch')
    expected = _receipts(circuit, tree, lifted, source['sha256'], report['artifact']['sha256'])
    if digest(expected) != digest(report.get('regions')):
        raise StrataError('symbolic region evidence differs on replay')
    summary = dict(symbolic=sum(r['evidence_kind'] == 'symbolic' for r in expected),
                   unknown=sum(r['evidence_kind'] == 'unknown' for r in expected))
    if report.get('summary') != summary or report.get('coverage') != lifted['coverage'] or report.get('metadata') != lifted.get('metadata', []):
        raise StrataError('region evidence coverage mismatch')
    return dict(replayed=True, artifact=report['artifact']['sha256'], summary=summary,
                regions=[dict(region=r['region'], status=r['status'], evidence_kind=r['evidence_kind']) for r in expected],
                scope='recomputed fixed-region symbolic semantics; unknown regions remain unknown',
                official_verifier_run=False, whole_circuit_certificate=False)
