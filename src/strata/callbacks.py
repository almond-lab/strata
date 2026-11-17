"""Pinned, typed external execution hooks used by ordinary Lines and research.

Python callbacks are trusted extensions. Their code and declared dependencies
are pinned; arbitrary dictionaries cannot stand in for scoped check receipts.
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import hashlib
import inspect
import json
from functools import wraps
from contextlib import contextmanager
from .model import Region, StrataError, digest, canonical


def artifact_bytes(frontend, artifact):
    content = frontend.serialize(artifact) if hasattr(frontend, 'serialize') else str(artifact)
    return content.encode() if isinstance(content, str) else content


@dataclass(frozen=True)
class BuildRequest:
    parameters: dict
    implementation: dict


@dataclass(frozen=True)
class BuildReceipt:
    artifact: object
    tree: Region


@dataclass(frozen=True)
class CheckRequest:
    stage: str
    artifact: object
    artifact_hash: str
    parameters: dict
    domain: dict
    scope: str = 'full_artifact'
    contract_hash: str | None = None


@dataclass(frozen=True)
class CheckReceipt:
    stage: str
    passed: bool
    artifact_hash: str
    parameters_hash: str
    domain_hash: str
    scope: str
    resources: dict = field(default_factory=dict)
    details: dict = field(default_factory=dict)

    @classmethod
    def for_request(cls, request, *, passed, resources=None, details=None):
        return cls(request.stage, passed, request.artifact_hash, digest(request.parameters), digest(request.domain), request.scope, resources or {}, details or {})


@dataclass(frozen=True)
class FullArtifactLab:
    passed: bool
    exhaustive: bool
    evidence_kind: str
    scope: dict
    circuit_hash: str
    contract_hash: str
    verifier_hash: str
    receipt: dict
    small_model: str = 'unavailable; full artifact checked at declared domain'


class ExternalCheckError(StrataError):
    """A rejected external outcome, explicitly separate from invalid evidence."""
    def __init__(self, message, request, receipt=None, *, scope_valid=False, diagnostics=None):
        super().__init__(message)
        self.request = request
        try:
            self.receipt = json.loads(canonical(receipt)) if isinstance(receipt, CheckReceipt) else None
        except (TypeError, ValueError):
            self.receipt = None
        self.scope_valid = scope_valid
        self.diagnostics = diagnostics or {}


@contextmanager
def preserve_completed_check(project, frontend, artifact, parameters, domain, receipt):
    """Keep a completed external receipt when later resource validation fails.

Ingest has no Line yet, so also write a directly discoverable failure record.
Legacy adapters without typed scoped receipts retain their existing behavior.
"""
    try:
        yield
    except StrataError as error:
        required = {'stage','passed','artifact_hash','parameters_hash','domain_hash','scope'}
        if not isinstance(receipt,dict) or not required<=receipt.keys():raise
        request = CheckRequest(receipt['stage'],artifact,
            hashlib.sha256(artifact_bytes(frontend,artifact)).hexdigest(),parameters,domain)
        typed = CheckReceipt(**{name:receipt[name] for name in CheckReceipt.__dataclass_fields__ if name in receipt})
        try:
            validate_check(typed,request)
        except ExternalCheckError as exc:
            failure=exc
        else:
            failure=ExternalCheckError('completed external receipt rejected by resource validation: '+str(error),
                                       request,typed,scope_valid=True)
        failure.diagnostics.update(kind='postcheck_validation_failed', verifier_hash=receipt.get('verifier_hash'),
            sandbox_limits=receipt.get('sandbox_limits'),runtime_environment=receipt.get('runtime_environment'))
        content=artifact_bytes(frontend,artifact)
        artifact_id=project.store.blob(content)
        receipt_id=project.store.blob(canonical(receipt).encode())
        record=project.store.put('external_check_failure',dict(stage=request.stage,passed=False,
            artifact=artifact_id,receipt_artifact=receipt_id,receipt=receipt,error=str(error),
            scope_valid=failure.scope_valid,kind='postcheck_validation_failed',
            binding=dict(parameters=parameters,domain=domain,scope=request.scope)))
        failure.diagnostics['failure_record']=record
        raise failure from error


def retain_external_failure(stage):
    """Keep failed checks even when a Line gate raises to prevent promotion."""
    def decorate(method):
        @wraps(method)
        def wrapped(line, *args, **kwargs):
            try:
                return method(line, *args, **kwargs)
            except ExternalCheckError as exc:
                request = exc.request
                artifact = line.project.store.blob(artifact_bytes(line.task.frontend, request.artifact))
                if artifact != request.artifact_hash:
                    raise StrataError('failed check artifact changed after evaluation') from exc
                outcome = dict(receipt=exc.receipt, diagnostics=exc.diagnostics,
                               scope_valid=exc.scope_valid, error=str(exc))
                receipt_blob = line.project.store.blob(json.dumps(outcome, sort_keys=True).encode())
                rejection = exc.scope_valid and exc.receipt is not None and exc.receipt['passed'] is False
                event = dict(stage=stage, passed=False, artifact=artifact,
                             receipt_artifact=receipt_blob, receipt=exc.receipt,
                             binding=dict(parameters=request.parameters, domain=request.domain,
                                          scope=request.scope, contract=request.contract_hash),
                             evidence_kind='measured_rejection' if rejection else 'postcheck_rejection' if exc.scope_valid else 'invalid_external_evidence',
                             failure=dict(stage=request.stage, error=str(exc),
                                          kind='verifier_rejection' if rejection else exc.diagnostics.get('kind','scope_or_receipt_invalid')),
                             scope_valid=exc.scope_valid, diagnostics=exc.diagnostics)
                line.project.store.event(line.id, event)
                if rejection:
                    for contract in line.patch.contracts:
                        line.project.store.negative(contract, line.patch.move, 'build', event,
                            'Reopen after changing the implementation or the failed verifier condition.',
                            evidence=dict(artifact_hash=receipt_blob, checked_artifact=artifact))
                raise
        return wrapped
    return decorate


def validate_check(receipt, request):
    if not isinstance(receipt, CheckReceipt):
        raise ExternalCheckError('external check must return a typed CheckReceipt', request)
    expected = (request.stage, request.artifact_hash, digest(request.parameters), digest(request.domain), request.scope)
    actual = (receipt.stage, receipt.artifact_hash, receipt.parameters_hash, receipt.domain_hash, receipt.scope)
    if actual != expected:
        raise ExternalCheckError('external receipt scope/hash mismatch; not a measured rejection', request, receipt)
    if type(receipt.passed) is not bool:
        raise ExternalCheckError('external receipt passed must be boolean', request, receipt)
    if not isinstance(receipt.resources, dict) or not isinstance(receipt.details, dict):
        raise ExternalCheckError('external receipt needs structured resources and details', request, receipt)
    try:
        digest(asdict(receipt))
    except (TypeError, ValueError, StrataError) as exc:
        raise ExternalCheckError('external receipt is not canonical structured data', request) from exc
    if receipt.passed is False:
        excerpt = canonical(receipt.details)[:1200]
        raise ExternalCheckError('external receipt failed (passed=false, scope valid): ' + excerpt,
                                 request, receipt, scope_valid=True)
    return receipt


class PinnedCallback:
    def __init__(self, callback, *, dependencies=()):
        self.callback = callback
        source = inspect.getsourcefile(callback)
        if not source:
            raise StrataError('callback needs a source file')
        self.paths = tuple(dict.fromkeys([str(Path(source).resolve()), *(str(Path(p).resolve()) for p in dependencies)]))
        self.pins = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in self.paths}
        self.name = callback.__module__ + ':' + callback.__qualname__

    def identity_hash(self):
        for path, expected in self.pins.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise StrataError('callback source or dependency changed')
        return digest(dict(name=self.name, files=self.pins))

    def __call__(self, request):
        self.identity_hash()
        result = self.callback(request)
        self.identity_hash()
        return result


@dataclass(frozen=True)
class StructuralRequest:
    before: object
    old_tree: Region
    after: object
    new_tree: Region
    before_artifact: str
    after_artifact: str


@dataclass(frozen=True)
class StructuralReceipt:
    passed: bool
    mechanism: str
    checks: dict[str, bool]
    before_artifact: str
    after_artifact: str
    authority: str = 'pinned external structural audit'


class CallbackAdapter:
    def __init__(self, *, frontend, build, lab, measure, oracle, verify, audit_restructure=None):
        self.frontend = frontend
        self.callbacks = dict(build=build, lab=lab, measure=measure, oracle=oracle, verify=verify)
        if audit_restructure is not None:
            self.callbacks['audit_restructure'] = audit_restructure
        if any(not isinstance(callback, PinnedCallback) for callback in self.callbacks.values()):
            raise StrataError('external callbacks must be PinnedCallback instances')

    def identity_hash(self):
        source = inspect.getsourcefile(type(self.frontend))
        return digest(dict(callbacks={name: callback.identity_hash() for name, callback in self.callbacks.items()},
                           frontend_source=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                           frontend_identity=self.frontend.identity_hash() if hasattr(self.frontend, 'identity_hash') else None))

    def __getattr__(self, name):
        if name == 'audit_restructure' and name in self.callbacks:
            return self._audit_restructure
        raise AttributeError(name)

    def _audit_restructure(self, before, old_tree, after, new_tree):
        request = StructuralRequest(before, old_tree, after, new_tree,
                                    hashlib.sha256(artifact_bytes(self.frontend, before)).hexdigest(),
                                    hashlib.sha256(artifact_bytes(self.frontend, after)).hexdigest())
        receipt = self.callbacks['audit_restructure'](request)
        if not isinstance(receipt, StructuralReceipt):
            raise StrataError('external structural audit must return a typed StructuralReceipt')
        if receipt.before_artifact != request.before_artifact or receipt.after_artifact != request.after_artifact:
            raise StrataError('external structural audit describes different artifacts')
        if not receipt.mechanism or not receipt.checks or any(type(value) is not bool for value in receipt.checks.values()):
            raise StrataError('external structural audit needs explicit boolean mechanism checks')
        if receipt.passed is not all(receipt.checks.values()):
            raise StrataError('external structural audit result disagrees with its checks')
        return {**asdict(receipt), 'verifier_hash': self.callbacks['audit_restructure'].identity_hash()}

    def build(self, parameters, implementation):
        result = self.callbacks['build'](BuildRequest(parameters, implementation))
        if not isinstance(result, BuildReceipt) or not isinstance(result.tree, Region):
            raise StrataError('external build must return a typed BuildReceipt')
        return result.artifact, result.tree

    def _check(self, name, artifact, parameters, domain, contract_hash=None):
        stage = {'oracle': 'oracle_full', 'verify': 'official_verifier', 'measure': 'build_price', 'lab': 'lab'}[name]
        request = CheckRequest(stage, artifact, hashlib.sha256(artifact_bytes(self.frontend, artifact)).hexdigest(), parameters, domain, contract_hash=contract_hash)
        receipt = self.callbacks[name](request)
        try:
            validate_check(receipt, request)
        except ExternalCheckError as exc:
            exc.diagnostics['verifier_hash'] = self.callbacks[name].identity_hash()
            raise
        result = asdict(receipt)
        result['verifier_hash'] = self.callbacks[name].identity_hash()
        result['seeds'] = receipt.details.get('seeds', [])
        result['failure_model'] = receipt.details.get('failure_model', {'status': 'unspecified'})
        for key in ('draws', 'selection', 'selected_seed'):
            if key in receipt.details:
                result[key] = receipt.details[key]
        return result

    def lab(self, artifact, tree, parameters, domain, **limits):
        receipt = self._check('lab', artifact, parameters, domain, tree.contract.fingerprint)
        return FullArtifactLab(True, False, 'full_artifact', dict(parameters=parameters, domain=domain),
                               receipt['artifact_hash'], tree.contract.fingerprint, receipt['verifier_hash'], receipt)

    def measure(self, artifact, parameters, domain):
        self.measurement_receipt = self._check('measure', artifact, parameters, domain)
        return self.measurement_receipt['resources']

    def oracle(self, artifact, parameters, domain):
        return self._check('oracle', artifact, parameters, domain)

    def verify(self, artifact, parameters, domain):
        return self._check('verify', artifact, parameters, domain)

    def ledger(self, artifact, tree):
        return {'root': self.frontend.measured(artifact)}

    def floors(self, tree):
        return []
