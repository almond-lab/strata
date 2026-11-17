"""Canonical multi-file artifacts with explicit circuit and numeric components.

Q is the largest emitted component width: these bundles execute components
sequentially. Host-wide width must be reported separately as Q_total.
"""
from dataclasses import dataclass, asdict
from pathlib import Path, PurePosixPath
from types import MappingProxyType
import base64
import hashlib
import json
from .model import Contract, Region, StrataError, canonical, digest
from .kickmix import Kickmix, instructions


def safe_name(name):
    if not isinstance(name, str) or not name or '\\' in name or PurePosixPath(name).is_absolute() or any(p in {'', '..', '.'} for p in name.split('/')) or '\x00' in name:
        raise StrataError('invalid bundle component path')
    return name


@dataclass(frozen=True)
class Component:
    id: str
    path: str
    kind: str = 'data'
    frozen: bool = False
    sha256: str | None = None
    max_qubits: int | None = None

    def __post_init__(self):
        safe_name(self.path)
        if not self.id or self.id == 'root' or self.kind not in {'kickmix', 'data', 'opaque'}:
            raise StrataError('invalid bundle component')
        if self.frozen and (not self.sha256 or len(self.sha256) != 64):
            raise StrataError('frozen components require a pinned sha256')
        if self.max_qubits is not None and (self.kind != 'kickmix' or self.max_qubits < 0):
            raise StrataError('component width cap requires a circuit')


@dataclass(frozen=True)
class Bundle:
    files: dict[str, bytes]

    def __post_init__(self):
        if not self.files or any(not isinstance(v, bytes) for v in self.files.values()):
            raise StrataError('bundle requires component bytes')
        object.__setattr__(self, 'files', MappingProxyType({safe_name(k): v for k, v in self.files.items()}))


class BundleFrontend:
    def __init__(self, components):
        self.components = tuple(c if isinstance(c, Component) else Component(**c) for c in components)
        if not self.components or len({c.id for c in self.components}) != len(self.components) or len({c.path for c in self.components}) != len(self.components):
            raise StrataError('bundle needs distinct component ids and paths')

    def identity_hash(self):
        return digest([asdict(c) for c in self.components])

    def validate(self, bundle):
        if not isinstance(bundle, Bundle) or set(bundle.files) != {c.path for c in self.components}:
            raise StrataError('bundle component set changed')
        for c in self.components:
            content = bundle.files[c.path]
            if c.frozen and hashlib.sha256(content).hexdigest() != c.sha256:
                raise StrataError('frozen component changed: ' + c.id)
            if c.kind == 'kickmix':
                circuit = Kickmix().parse(content.decode())
                if c.max_qubits is not None and circuit.num_qubits > c.max_qubits:
                    raise StrataError('component width cap exceeded: ' + c.id)
        return bundle

    def load(self, path):
        root = Path(path)
        if not root.is_dir():
            raise StrataError('bundle base must be a directory')
        from .artifacts import manifest
        files = manifest(root)['files']
        return self.validate(Bundle({name: (root / name).read_bytes() for name in files}))

    def serialize(self, bundle):
        self.validate(bundle)
        return canonical({'format': 'strata.bundle.v1', 'files': {name: base64.b64encode(data).decode('ascii') for name, data in bundle.files.items()}})

    def parse(self, text):
        try:
            value = json.loads(text)
            if set(value) != {'format', 'files'} or value['format'] != 'strata.bundle.v1':
                raise ValueError('unknown format')
            return self.validate(Bundle({name: base64.b64decode(data, validate=True) for name, data in value['files'].items()}))
        except (ValueError, TypeError, KeyError) as exc:
            raise StrataError('invalid bundle encoding') from exc

    def measured(self, bundle):
        self.validate(bundle)
        metrics = {c.id: Kickmix().measured(Kickmix().parse(bundle.files[c.path].decode())) for c in self.components if c.kind == 'kickmix'}
        return dict(Q=max((m['Q'] for m in metrics.values()), default=0), component_widths={name: m['Q'] for name, m in metrics.items()},
                    components=metrics, artifact_bytes=sum(map(len, bundle.files.values())), width_semantics='maximum_sequential_component_emitted_width')

    def analyze(self, bundle, out, max_ops=100_000):
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        report = dict(metrics=self.measured(bundle), components={}, coverage='component inventory',
                      numeric_metrics='require pinned full-artifact verifier')
        for index, component in enumerate(self.components):
            data = bundle.files[component.path]
            item = dict(path=component.path, kind='opaque' if component.frozen else component.kind,
                        frozen=component.frozen, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
            if component.kind == 'kickmix' and not component.frozen:
                item['analysis'] = Kickmix().analyze(Kickmix().parse(data.decode()), out / ('component-' + str(index)), max_ops=max_ops)
            report['components'][component.id] = item
        (out / 'analysis.json').write_text(json.dumps(report, indent=2))
        return report

    def lift(self, bundle):
        tree = self.tree(bundle, 'strata:unconfirmed-bundle')
        return tree, dict(confirmed=False, coverage='component boundaries only',
                          frozen=[c.id for c in self.components if c.frozen],
                          note='Opaque frozen components have byte pins; task semantics require the configured adapter or full-artifact verifier.')

    def operations(self, bundle):
        for c in self.components:
            if c.kind == 'kickmix':
                yield from (op for _, _, op, _ in instructions(Kickmix().parse(bundle.files[c.path].decode())))

    def tree(self, bundle, reference, promise='declared domain'):
        self.validate(bundle)
        contract = Contract('bundle', (), reference, promise=promise)
        children = tuple(Region(c.id, i, i + 1, Contract(c.id, (), reference + ':' + c.id, promise=promise),
                                kind='opaque' if c.frozen or c.kind == 'opaque' else c.kind,
                                evidence=('frozen sha256:' + c.sha256) if c.frozen else 'full-artifact verifier required') for i, c in enumerate(self.components))
        return Region('root', 0, len(children), contract, children, kind='bundle', evidence='full-artifact verification')

    def validate_tree(self, tree):
        if {r.id for r in tree.children} != {c.id for c in self.components}:
            raise StrataError('bundle component tree changed')
        for component in self.components:
            region = tree.find(component.id)
            if component.frozen and (region.kind != 'opaque' or region.evidence != 'frozen sha256:' + component.sha256):
                raise StrataError('frozen component must retain its pinned opaque boundary')

    def validate_edit(self, before, old_tree, after, tree, regions, replacement_regions=()):
        self.validate(before); self.validate(after)
        self.validate_tree(old_tree); self.validate_tree(tree)
        targets = {child.id for name in regions for child in old_tree.find(name).walk()}
        for component in self.components:
            if before.files[component.path] != after.files[component.path] and (component.frozen or component.id not in targets):
                raise StrataError('patch changed frozen or undeclared component: ' + component.id)
        if {r.id for r in tree.children} != {c.id for c in self.components}:
            raise StrataError('bundle component tree changed')

    def write(self, bundle, root):
        self.validate(bundle)
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        for name, content in bundle.files.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)


class DataFrontend(BundleFrontend):
    """A single opaque/numeric file still uses the complete byte-bound bundle."""
    def __init__(self, name):
        super().__init__([Component('data', name, 'data')])

    def load(self, path):
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise StrataError('data base must be a regular file')
        return self.validate(Bundle({self.components[0].path: path.read_bytes()}))
