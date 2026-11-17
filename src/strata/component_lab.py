"""Exact component labs with byte-bound bundle and frozen-component receipts."""
from dataclasses import asdict, dataclass
import hashlib
import inspect

from .bundle import Bundle, BundleFrontend
from .callbacks import FullArtifactLab
from .kickmix import Kickmix
from .lab import exhaustive, validate_lab_receipt
from .model import Contract, StrataError, digest


@dataclass(frozen=True)
class ComponentLab:
    contract: Contract
    reference: object
    domain: object = None


@dataclass(frozen=True)
class BundleLab:
    passed: bool
    exhaustive: bool
    evidence_kind: str
    scope: dict
    circuit_hash: str
    contract_hash: str
    components: dict
    frozen: dict


def _hash(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def _valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def run_component_labs(frontend, artifact, tree, parameters, adapter, options=None):
    """Execute every mutable component's declared lab against its actual bytes.

    Frozen components are checked by the frontend's pinned SHA-256 obligation.
    An exact lab receives its own component contract, reference and optional
    promised input domain. Numeric/opaque files need a typed external receipt.
    """
    # The frontend owns component names and immutable-byte obligations.
    if not isinstance(frontend, BundleFrontend) or not isinstance(artifact, Bundle):
        raise StrataError('component lab requires a bundle frontend and artifact')
    frontend.validate(artifact)
    hook = adapter.lab_components
    signature = inspect.signature(hook)
    try:
        signature.bind(parameters, artifact)
    except TypeError:
        try:
            signature.bind(parameters, artifact=artifact)
        except TypeError:
            signature.bind(parameters)
            specs = hook(parameters)
        else:
            specs = hook(parameters, artifact=artifact)
    else:
        specs = hook(parameters, artifact)
    mutable = {component.id for component in frontend.components if not component.frozen}
    if not isinstance(specs, dict) or set(specs) != mutable:
        raise StrataError('component labs must cover every nonfrozen component exactly')
    entries, frozen = {}, {}
    for component in frontend.components:
        content = artifact.files[component.path]
        raw_hash = _hash(content)
        if component.frozen:
            frozen[component.id] = raw_hash
            continue
        spec = specs[component.id]
        if isinstance(spec, ComponentLab):
            if component.kind != 'kickmix' or not isinstance(spec.contract, Contract) or not callable(spec.reference):
                raise StrataError('exact ComponentLab requires a circuit contract and callable reference')
            result = exhaustive(Kickmix().parse(content.decode()), spec.contract, spec.reference,
                                parameters, domain=spec.domain, **(options or {}))
        elif isinstance(spec, FullArtifactLab):
            result = spec
        else:
            raise StrataError('component lab must be a typed ComponentLab or FullArtifactLab')
        entries[component.id] = dict(id=component.id, path=component.path, kind=component.kind,
                                     artifact_hash=raw_hash, execution_hash=result.circuit_hash,
                                     lab=asdict(result))
    passed = all(entry['lab'].get('passed') is True for entry in entries.values())
    complete = passed and bool(entries) and all(entry['lab'].get('exhaustive') is True for entry in entries.values())
    result = BundleLab(passed, complete, 'component_scoped', dict(parameters),
                       _hash(frontend.serialize(artifact)), tree.contract.fingerprint, entries, frozen)
    if passed:
        validate_component_lab(asdict(result), frontend)
    return result


def validate_component_lab(lab, frontend):
    """Validate complete per-component evidence without inflating its scope.

    The project additionally binds the aggregate circuit hash to the serialized
    bundle and pins the adapter that supplied these component contracts.
    """
    def require(condition):
        if not condition:
            raise StrataError('invalid or incomplete component lab certificate')

    require(isinstance(frontend, BundleFrontend) and isinstance(lab, dict))
    require(lab.get('passed') is True and lab.get('evidence_kind') == 'component_scoped')
    require(isinstance(lab.get('scope'), dict))
    require(_valid_hash(lab.get('circuit_hash')) and _valid_hash(lab.get('contract_hash')))
    frozen = {c.id: c.sha256 for c in frontend.components if c.frozen}
    require(lab.get('frozen') == frozen)
    entries = lab.get('components')
    require(isinstance(entries, dict))
    require(set(entries) == {c.id for c in frontend.components if not c.frozen})
    all_exhaustive = bool(entries)
    # Cross-component behavior remains an obligation of the full artifact gate.
    for component in frontend.components:
        if component.frozen:
            continue
        entry = entries[component.id]
        require(isinstance(entry, dict))
        require(entry.get('id') == component.id and entry.get('path') == component.path and
                entry.get('kind') == component.kind and _valid_hash(entry.get('artifact_hash')))
        child = entry.get('lab')
        require(isinstance(child, dict))
        require(entry.get('execution_hash') == child.get('circuit_hash'))
        if child.get('evidence_kind') == 'full_artifact':
            scope, receipt = child.get('scope'), child.get('receipt')
            require(isinstance(scope, dict) and isinstance(receipt, dict))
            require(set(scope) == {'parameters', 'domain'} and scope['parameters'] == lab['scope'])
            require(isinstance(scope['domain'], dict))
            require(child.get('passed') is True and child.get('exhaustive') is False)
            require(child.get('circuit_hash') == entry['artifact_hash'])
            require(_valid_hash(child.get('contract_hash')) and _valid_hash(child.get('verifier_hash')))
            require(receipt.get('verifier_hash') == child['verifier_hash'])
            expected = dict(stage='lab', passed=True, artifact_hash=entry['artifact_hash'],
                            parameters_hash=digest(lab['scope']), domain_hash=digest(scope['domain']),
                            scope='full_artifact')
            require(all(receipt.get(key) == value for key, value in expected.items()))
            all_exhaustive = False
        else:
            require(component.kind == 'kickmix')
            validate_lab_receipt(child)
            require(child['scope'] == lab['scope'])
    require(lab.get('exhaustive') is all_exhaustive)
