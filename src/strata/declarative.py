"""Configuration-only tasks: existing artifacts and an offline pinned verifier.

The executable receives a JSON request path and writes a CheckReceipt JSON to
stdout. No Python adapter, fabricated reference, or small model is required.
"""
from pathlib import Path
from dataclasses import asdict
import base64
import hashlib
import json
import tempfile
import shutil
from .model import Contract, Region, StrataError, digest
from .artifacts import manifest, verify_manifest
from .bundle import Bundle, BundleFrontend, DataFrontend
from .callbacks import CheckRequest, CheckReceipt, FullArtifactLab, ExternalCheckError, artifact_bytes, validate_check
from .kickmix import Kickmix
from . import sandbox


class CommandAdapter:
    def __init__(self, root, config):
        self.root, self.config = Path(root), config
        self.base = self.root / config['base']
        self.base_manifest = manifest(self.base)
        if config.get('scale'):
            raise StrataError('configuration-only artifacts have no declared scalable model')
        kind = config.get('frontend', 'kickmix')
        if kind == 'bundle':
            self.frontend = BundleFrontend(config.get('components', []))
        elif kind == 'data':
            self.frontend = DataFrontend(self.base.name)
        elif kind == 'kickmix':
            self.frontend = Kickmix()
        else:
            from .frontends import frontend_for
            self.frontend = frontend_for(kind)
        spec = config['verifier']
        self.command = spec.get('command')
        self.command = list(self.command) if isinstance(self.command, list) else self.command
        if not isinstance(self.command, list) or not self.command or any(not isinstance(x, str) for x in self.command):
            raise StrataError('verifier command must be an argument list, never a shell string')
        executable = Path(self.command[0])
        if not executable.is_absolute() or not executable.is_file():
            raise StrataError('verifier executable must be an absolute file path')
        if executable.name in {'sh', 'bash', 'dash', 'zsh', 'fish', 'ksh', 'csh', 'pwsh', 'powershell'} or executable.resolve().name in {'sh', 'bash', 'dash', 'zsh', 'fish', 'ksh', 'csh'}:
            raise StrataError('shell verifier commands are unsupported; pass executable arguments directly')
        pins = dict(spec.get('dependencies', {}))
        pins[str(executable)] = spec.get('sha256')
        self.pins = {}
        for name, expected in pins.items():
            path = Path(name)
            path = path if path.is_absolute() else self.root / path
            if not expected or len(expected) != 64 or not path.is_file():
                raise StrataError('verifier executable and dependencies need explicit sha256 pins')
            self.pins[str(path.resolve())] = expected
        # CPython discovers pyvenv.cfg from the invoked path. Resolving this
        # symlink or staging a bare binary silently loses the virtualenv.
        self.command[0] = str(executable.absolute())
        self.executable = executable.absolute()
        self.environment_root = self.executable.parent.parent if (self.executable.parent.parent / 'pyvenv.cfg').is_file() else None
        self.environment = self._environment_identity()
        keys = {'timeout','max_output','memory_bytes','max_file_bytes','cpu_seconds'}
        self.limits = sandbox.effective_limits(**{key:spec[key] for key in keys if key in spec})
        for index, argument in enumerate(self.command[1:], 1):
            if '{' not in argument and (self.root / argument).is_file():
                resolved = str((self.root / argument).resolve())
                if resolved not in self.pins:
                    raise StrataError('command file argument is an unpinned dependency')
                self.command[index] = resolved
        if not any('{request}' in arg for arg in self.command):
            raise StrataError('verifier command needs a {request} JSON input argument')
        self.identity_hash()

    def _environment_identity(self):
        if self.environment_root is None:
            return dict(kind='executable', invocation=str(self.executable), target=str(self.executable.resolve()))
        root = self.environment_root
        files = {root / 'pyvenv.cfg'}
        for site in root.glob('lib*/python*/site-packages'):
            files.update(site.glob('*.pth'))
            files.update(site.glob('*.egg-link'))
            for metadata in site.glob('*.dist-info'):
                files.update(p for p in metadata.rglob('*') if p.is_file())
        return dict(kind='python_virtualenv', invocation=str(self.executable), target=str(self.executable.resolve()),
            files={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(files)},
            coverage='interpreter, pyvenv.cfg, package metadata and path configuration; package source bytes require explicit dependency pins')

    def identity_hash(self):
        verify_manifest(self.base, self.base_manifest)
        for path, expected in self.pins.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise StrataError('pinned verifier executable or dependency changed')
        if self._environment_identity() != self.environment:
            raise StrataError('pinned verifier runtime environment changed')
        return digest(dict(adapter_source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), config=self.config,
                           pins=self.pins, runtime_environment=self.environment, limits=self.limits, input=self.base_manifest))

    def freeze_implementation(self, implementation):
        if set(implementation) - {'files'}:
            raise StrataError('declarative edits require a files mapping')
        frozen = {}
        for name, value in implementation.get('files', {}).items():
            if isinstance(value, dict) and set(value) == {'content_base64', 'sha256'}:
                content = base64.b64decode(value['content_base64'], validate=True)
                if hashlib.sha256(content).hexdigest() != value['sha256']:
                    raise StrataError('replacement component hash mismatch')
            else:
                path = self.root / value
                if path.is_symlink() or not path.is_file():
                    raise StrataError('replacement must be a regular file')
                content = path.read_bytes()
            frozen[name] = dict(content_base64=base64.b64encode(content).decode('ascii'), sha256=hashlib.sha256(content).hexdigest())
        return {'files': frozen} if frozen else {}

    def merge_implementation(self, parent, changes):
        old = self.freeze_implementation(parent)
        new = self.freeze_implementation(changes)
        files = {**old.get('files', {}), **new.get('files', {})}
        return {'files': files} if files else {}

    def build(self, parameters, implementation):
        self.identity_hash()
        if parameters:
            raise StrataError('configuration-only task has no scale parameters')
        artifact = self.frontend.load(self.base)
        changes = self.freeze_implementation(implementation).get('files', {})
        replacements = {name: base64.b64decode(value['content_base64']) for name, value in changes.items()}
        if isinstance(artifact, Bundle):
            if set(replacements) - set(artifact.files):
                raise StrataError('replacement names an unknown component')
            artifact = Bundle({**artifact.files, **replacements})
            self.frontend.validate(artifact)
            tree = self.frontend.tree(artifact, 'command:' + self.identity_hash(), self.config.get('domain', {}).get('kind', 'declared domain'))
        else:
            if set(replacements) - {self.base.name}:
                raise StrataError('replacement names an unknown artifact')
            if replacements:
                artifact = self.frontend.parse(replacements[self.base.name].decode())
            contract = Contract('artifact', (), 'command:' + self.identity_hash(), promise=self.config.get('domain', {}).get('kind', 'declared domain'))
            tree = Region('root', 0, max(1, len(str(artifact).splitlines())), contract, kind='opaque', evidence='full-artifact command verifier')
        return artifact, tree

    def _check(self, stage, artifact, parameters, domain, contract_hash=None):
        verifier_hash = self.identity_hash()
        content = artifact_bytes(self.frontend, artifact)
        request = CheckRequest(stage, artifact, hashlib.sha256(content).hexdigest(), parameters, domain, contract_hash=contract_hash)
        with tempfile.TemporaryDirectory(prefix='strata-check-') as name:
            work = Path(name)
            if isinstance(artifact, Bundle):
                target = work / 'artifact'
                self.frontend.write(artifact, target)
            else:
                target = work / 'artifact.kmx'
                target.write_bytes(content)
            request_path = work / 'request.json'
            body = {key: value for key, value in vars(request).items() if key != 'artifact'}
            body['artifact'] = str(target)
            body['artifact_files'] = manifest(target)
            request_path.write_text(json.dumps(body, sort_keys=True))
            # /tmp is private inside Sandbox. Stage every pinned input beneath
            # its writable bind, preserving relative dependency directories.
            staged = {}
            for source, expected in self.pins.items():
                if source == str(self.executable.resolve()):
                    continue
                destination = work / 'dependencies' / source.lstrip('/')
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
                    raise StrataError('staged verifier dependency hash mismatch')
                staged[source] = str(destination)
            args = [staged.get(argument, argument).replace('{artifact}', str(target)).replace('{request}', str(request_path)) for argument in self.command]
            readonly_paths = [str(self.environment_root)] if self.environment_root is not None else [str(self.executable)]
            if not self.executable.resolve().is_relative_to(self.environment_root or self.executable):
                readonly_paths.append(str(self.executable.resolve()))
            try:
                run = sandbox.run(args, work, **self.limits, readonly_paths=readonly_paths)
            except StrataError as exc:
                raise ExternalCheckError(str(exc), request, diagnostics=dict(verifier_hash=verifier_hash,
                    limits=self.limits, runtime_environment=self.environment, kind='sandbox_execution_failed')) from exc
            verify_manifest(target, body['artifact_files'])
            for source, destination in staged.items():
                if hashlib.sha256(Path(destination).read_bytes()).hexdigest() != self.pins[source]:
                    raise StrataError('verifier modified a pinned dependency')
            try:
                receipt = CheckReceipt(**json.loads(run['output']))
            except (TypeError, ValueError) as exc:
                raise ExternalCheckError('verifier must emit a typed, scoped CheckReceipt JSON',request,
                    diagnostics=dict(output=run['output'][:1200],verifier_hash=verifier_hash,limits=self.limits)) from exc
            self.identity_hash()
            try:
                validate_check(receipt, request)
            except ExternalCheckError as exc:
                exc.diagnostics.update(verifier_hash=verifier_hash, sandbox=run['sandbox'], limits=self.limits,
                                       runtime_environment=self.environment)
                raise
        self.identity_hash()
        result = asdict(receipt)
        result.update(verifier_hash=verifier_hash, sandbox=run['sandbox'], seeds=receipt.details.get('seeds', []),
                      sandbox_limits=self.limits, runtime_environment=self.environment,
                      failure_model=receipt.details.get('failure_model', {'status': 'unspecified'}))
        for key in ('draws', 'selection', 'selected_seed'):
            if key in receipt.details:
                result[key] = receipt.details[key]
        return result

    def lab(self, artifact, tree, parameters, domain, **limits):
        receipt = self._check('lab', artifact, parameters, domain, tree.contract.fingerprint)
        return FullArtifactLab(True, False, 'full_artifact', dict(parameters=parameters, domain=domain), receipt['artifact_hash'], tree.contract.fingerprint, receipt['verifier_hash'], receipt)

    def measure(self, artifact, parameters, domain):
        self.measurement_receipt = self._check('build_price', artifact, parameters, domain)
        return self.measurement_receipt['resources']

    def oracle(self, artifact, parameters, domain):
        return self._check('oracle_full', artifact, parameters, domain)

    def verify(self, artifact, parameters, domain):
        return self._check('official_verifier', artifact, parameters, domain)

    def ledger(self, artifact, tree):
        return {'root': self.frontend.measured(artifact)}

    def floors(self, tree):
        return []
