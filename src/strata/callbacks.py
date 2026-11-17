"""Pinned, typed external execution hooks used by ordinary Lines and research.

Python callbacks are trusted extensions. Their code and declared dependencies
are pinned; arbitrary dictionaries cannot stand in for scoped check receipts.
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import hashlib
import inspect
from .model import Region, StrataError, digest


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


def validate_check(receipt, request):
    if not isinstance(receipt, CheckReceipt):
        raise StrataError('external check must return a typed CheckReceipt')
    expected = (request.stage, request.artifact_hash, digest(request.parameters), digest(request.domain), request.scope)
    actual = (receipt.stage, receipt.artifact_hash, receipt.parameters_hash, receipt.domain_hash, receipt.scope)
    if actual != expected or receipt.passed is not True:
        raise StrataError('external receipt failed or has a different artifact/domain/parameter scope')
    if not isinstance(receipt.resources, dict) or not isinstance(receipt.details, dict):
        raise StrataError('external receipt needs structured resources and details')
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
        receipt = validate_check(self.callbacks[name](request), request)
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
