"""One core for Python, CLI and compact agent queries.

Adapters implement build, reference and independent full verification. They are
trusted Python API extensions; external CLI processes use Sandbox separately.
A recorded verifier pass is evidence from its pinned adapter, never a Lean proof.
"""
from pathlib import Path
from dataclasses import asdict
import hashlib
import inspect
import json
import math
import tomllib
import uuid
from .model import StrataError, digest, region_from_dict
from .store import Store
from .cost import Formula, portfolio, rank_gaps
from .callbacks import retain_external_failure, preserve_completed_check
from .engine import Patch, MOVES, applicable_moves, compose
from .kickmix import Kickmix
from .lab import exhaustive
from .artifacts import manifest, verify_manifest


def adapter_hash(adapter):
    if hasattr(adapter, 'identity_hash'):
        return adapter.identity_hash()
    path = inspect.getsourcefile(type(adapter))
    if not path:
        raise StrataError('adapter must have a pinned source file')
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Project:
    def __init__(self, path, adapters=None):
        self.path = Path(path).resolve()
        with self.path.open('rb') as f:
            self.config = tomllib.load(f)
        self.root = self.path.parent
        cfg = self.config
        if cfg.get('publish', {}).get('submit', 'manual') != 'manual':
            raise StrataError('automatic submission is unsupported')
        if cfg.get('gates', {}).get('order', ['lab', 'build_price', 'oracle_full', 'official_verifier']) != ['lab', 'build_price', 'oracle_full', 'official_verifier']:
            raise StrataError('verification gates cannot be reordered')
        self.store = Store(self.root / cfg['project'].get('store', '.strata'))
        self.adapters = adapters or {}
        self.frontend = Kickmix()
        ids = [t['id'] for t in cfg['task']]
        if len(ids) != len(set(ids)):
            raise StrataError('duplicate task')
        for t in cfg['task']:
            dom = t.get('domain', {})
            if dom.get('kind') == 'sampled':
                seeds = dom.get('seeds', [])
                if len(set(seeds)) < 2 or dom.get('compare') not in {'expected', 'fixed_seeds'}:
                    raise StrataError('stochastic comparison requires precommitted multiple fixed seeds')
        self.tasks = {t['id']: Task(self, t) for t in cfg['task']}

    def task(self, name):
        return self.tasks[name]

    def init(self):
        return {name: t.ingest() for name, t in self.tasks.items()}

    def open_line(self, parent, move, regions, note='', implementation=None, reads=(), writes=(), replacement_regions=()):
        if isinstance(parent, str):
            parent = Design(self, parent)
        parent.validate()
        task = self.task(parent.record['task'])
        implementation = implementation or {}
        adapter = task.adapter()
        if hasattr(adapter, 'freeze_implementation'):
            implementation = adapter.freeze_implementation(implementation)
        tree = parent.tree
        patch = Patch(move, tuple(regions), tuple(tree.find(r).contract.fingerprint for r in regions),
                      tuple(tree.find(r).occurrence for r in regions), implementation or {}, tuple(reads), tuple(writes), tuple(replacement_regions))
        h = self.store.put('line', dict(nonce=uuid.uuid4().hex, parent=parent.id, task=parent.record['task'],
                                      patch=asdict(patch), note=note, adapter_hash=parent.record['adapter_hash']))
        return Line(self, h)

    def line(self, id):
        return Line(self, id)

    def best(self):
        tasks, pairs = {}, []
        for name, task in self.tasks.items():
            quarantined={r['design'] for _,r in self.store.all('quarantine')}
            records = [(h, d) for h, d in self.store.all('design') if d['task'] == name and h not in quarantined]
            if not records:
                raise StrataError('task has no certified design: ' + name)
            # Verify referenced certificate and artifacts before considering a record.
            valid = []
            for h, d in records:
                Design(self, h).validate()
                cert = self.store.get(d['certificate'], 'certificate')
                if cert['passed'] and cert['artifact'] == d['artifact']:
                    self.store.read_blob(d['artifact'])
                    valid.append((h, d))
            h, best = min(valid, key=lambda x: x[1]['cost'])
            base = task.base_design().record
            tasks[name] = dict(design=h, cost=best['cost'], base_cost=base['cost'])
            pairs.append((base['cost'], best['cost']))
        return dict(tasks=tasks, score=portfolio(self.config['score']['formula'], pairs))

    def replay(self, design):
        d = Design(self, design)
        d.validate()
        task = self.task(d.record['task'])
        adapter = task.adapter()
        if adapter_hash(adapter) != d.record['adapter_hash']:
            raise StrataError('adapter source changed')
        built, tree = adapter.build(d.record['parameters'], d.record['implementation'])
        artifact = hashlib.sha256(task.serialize(built)).hexdigest()
        if artifact != d.record['artifact'] or asdict(tree) != json.loads(json.dumps(asdict(d.tree))):
            # Dataclass tuple/list normalization via canonical hash instead.
            if artifact != d.record['artifact'] or digest(tree) != digest(d.tree):
                raise StrataError('replay build differs')
        self.store.read_blob(artifact)
        cert = self.store.get(d.record['certificate'], 'certificate')
        if cert['artifact'] != artifact or not cert['passed']:
            raise StrataError('invalid certificate')
        checked = task.verify_built(built, d.record['parameters'], adapter)
        if checked['official']['verifier_hash'] != cert['verifier_hash'] or checked['cost'] != d.record['cost']:
            raise StrataError('replayed verifier identity or measured cost changed')
        return dict(passed=True, design=design, artifact=artifact, receipt=checked)

    def merge(self, first, second):
        a, b = self.line(first), self.line(second)
        if a.record['parent'] != b.record['parent']:
            raise StrataError('automatic merge requires the same parent')
        for line in (a, b):
            line.result('verified')
        parent = Design(self, a.record['parent'])
        patches = compose(a.patch, b.patch, parent.tree)
        adapter = self.task(parent.record['task']).adapter()
        merge_implementation = getattr(adapter, 'merge_implementation', None)
        left, right = a.patch.implementation, b.patch.implementation
        for key in set(left) & set(right):
            if key == 'files' and merge_implementation is not None:
                for name in set(left[key]) & set(right[key]):
                    if digest(left[key][name]) != digest(right[key][name]):
                        raise StrataError('component file conflict: ' + name)
            elif digest(left[key]) != digest(right[key]):
                raise StrataError('generator parameter conflict')
        impl = dict(parent.record['implementation'])
        for patch in patches:
            impl = (merge_implementation(impl, patch['implementation']) if merge_implementation is not None
                    else {**impl, **patch['implementation']})
        # Returns an unverified line. Lab/build/full gates must run on the combination.
        return self.open_line(parent, 'replace-stages', ['root'], 'composition: ' + first + ' + ' + second, impl)


class Task:
    def __init__(self, project, config):
        self.project, self.config, self.id = project, config, config['id']
        self._command_adapter = None

    def adapter(self):
        if 'verifier' in self.config and 'adapter' not in self.config:
            if self._command_adapter is None:
                from .declarative import CommandAdapter
                self._command_adapter = CommandAdapter(self.project.root, self.config)
            return self._command_adapter
        name = self.config.get('adapter', self.id)
        if name not in self.project.adapters:
            raise StrataError(f'task {self.id} needs adapter {name!r}; cannot invent a scalable model or verifier')
        return self.project.adapters[name]

    def formula(self):
        functions=dict(getattr(self.adapter(), 'cost_functions', {}))
        policy=self.probability_policy()
        if policy:
            for key,value in policy.cost_functions().items():
                if key in functions:raise StrataError('ambiguous probability penalty function')
                functions[key]=value
        return Formula(self.config['cost'],functions)

    def probability_policy(self):
        if 'probability' not in self.config:return None
        from .probability import ProbabilityPolicy
        return ProbabilityPolicy.from_config(self.config['probability'])

    @staticmethod
    def probability_data(receipt):
        if not isinstance(receipt,dict):raise StrataError('stochastic price requires precommitted draw receipts')
        details=receipt.get('details',{})
        if not isinstance(details,dict):raise StrataError('invalid probability receipt details')
        draws={key:receipt.get(key,details.get(key)) for key in ('seeds','draws','selection','selected_seed')}
        if draws['selection'] is None:draws.pop('selection')
        return draws

    def price_resources(self, resources, receipt=None):
        policy=self.probability_policy()
        if not policy:return dict(cost=self.formula()(resources),probability=None)
        evidence=policy.evaluate(self.probability_data(receipt),self.formula())
        return dict(cost=evidence['cost'],probability=evidence)

    def params(self, small=False):
        return {k: v['small'][0] if small else v['full'] for k, v in self.config.get('scale', {}).items()}

    @property
    def frontend(self):
        adapter = self.adapter()
        if hasattr(adapter, 'frontend'):
            if self.config.get('components'):
                from .bundle import BundleFrontend
                configured = BundleFrontend(self.config['components'])
                if not isinstance(adapter.frontend, BundleFrontend) or adapter.frontend.identity_hash() != configured.identity_hash():
                    raise StrataError('adapter frontend does not match configured component contract')
            return adapter.frontend
        kind = self.config.get('frontend', 'kickmix')
        if kind == 'bundle':
            from .bundle import BundleFrontend
            return BundleFrontend(self.config.get('components', []))
        if kind == 'data':
            from .bundle import DataFrontend
            return DataFrontend(Path(self.config['base']).name)
        if kind != 'kickmix':
            from .frontends import frontend_for
            return frontend_for(kind)
        return self.project.frontend

    def serialize(self, artifact):
        from .callbacks import artifact_bytes
        data = artifact_bytes(self.frontend, artifact)
        if not isinstance(data, bytes):
            raise StrataError('frontend serialization must return bytes or text')
        if artifact_bytes(self.frontend, self.frontend.parse(data.decode())) != data:
            raise StrataError('frontend does not preserve serialized artifact bytes')
        return data

    def parse_artifact(self, data):
        artifact = self.frontend.parse(data.decode())
        if self.serialize(artifact) != data:
            raise StrataError('stored artifact has a noncanonical frontend encoding')
        return artifact

    def task_hash(self):
        source = inspect.getsourcefile(type(self.frontend))
        if not source:
            raise StrataError('frontend needs a pinned source file')
        return digest(dict(config=self.config, frontend=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                           frontend_identity=self.frontend.identity_hash() if hasattr(self.frontend, 'identity_hash') else None))

    def measured(self, artifact):
        metrics = self.frontend.measured(artifact)
        if type(metrics.get('Q')) not in (int, float) or metrics['Q'] < 0:
            raise StrataError('frontend must measure nonnegative emitted width Q')
        cap = self.config.get('max_qubits')
        if cap is not None and metrics['Q'] > cap:
            raise StrataError('build exceeds hard width')
        return metrics

    def check_resources(self, artifact, resources):
        metrics = self.measured(artifact)
        if not isinstance(resources, dict) or 'Q' not in resources:
            raise StrataError('verifier must report emitted width Q; use Q_total for host width')
        for name, value in resources.items():
            if isinstance(value, (int, float)) and (isinstance(value, bool) or not math.isfinite(value)):
                raise StrataError('resources must contain finite numeric metrics')
        if 'Q_total' in resources and (type(resources['Q_total']) not in (int, float) or resources['Q_total'] < metrics['Q']):
            raise StrataError('Q_total must cover the emitted component width')
        if resources['Q'] != metrics['Q']:
            raise StrataError('verifier width disagrees with emitted circuit; Q_total is separate host width')
        widths = metrics.get('component_widths')
        if widths is not None and resources.get('component_widths') != widths:
            raise StrataError('verifier component widths disagree with emitted bundle')
        from .acceptance import check_acceptance
        check_acceptance(self.config.get('accept', {}), resources)
        return metrics

    def run_lab(self, artifact, tree, parameters, limits=None):
        adapter = self.adapter()
        if hasattr(self.frontend, 'validate_tree'):
            self.frontend.validate_tree(tree)
        options = limits or {}
        if set(options) - {'max_trajectories', 'timeout', 'batch_size', 'shard_index', 'shard_count', 'branch_strategy'}:
            raise StrataError('unknown lab limit')
        if hasattr(adapter, 'lab_components'):
            from .component_lab import run_component_labs
            result = run_component_labs(self.frontend, artifact, tree, parameters, adapter, options)
        elif hasattr(adapter, 'lab'):
            from .callbacks import FullArtifactLab
            from .lab import LabResult
            result = adapter.lab(artifact, tree, parameters, self.config.get('domain', {}), **options)
            if not isinstance(result, (FullArtifactLab, LabResult)):
                raise StrataError('external lab must return a typed scoped lab receipt')
        else:
            domain = adapter.lab_domain(parameters) if hasattr(adapter, 'lab_domain') else None
            result = exhaustive(artifact, tree.contract, adapter.reference, parameters, domain=domain,
                                evidence_kind='exact_small' if self.config.get('scale') else 'exact_full', **options)
        return result

    def lab_record(self, result, artifact, tree, parameters):
        record = asdict(result)
        if record.get('evidence_kind') not in {'full_artifact', 'component_scoped'} and record.get('circuit_hash') != hashlib.sha256(str(artifact).encode()).hexdigest():
            raise StrataError('lab checked a different circuit')
        record['binding'] = dict(artifact=hashlib.sha256(self.serialize(artifact)).hexdigest(),
                                contract=tree.contract.fingerprint, parameters=parameters, promise=tree.contract.promise,
                                adapter_hash=adapter_hash(self.adapter()), task_hash=self.task_hash())
        if record.get('evidence_kind') == 'component_scoped' and record.get('circuit_hash') != record['binding']['artifact']:
            raise StrataError('component lab checked a different bundle')
        if record.get('passed') and record.get('evidence_kind')!='sample':
            self.validate_lab(record)
        elif record.get('evidence_kind')=='sample':
            shard=record.get('shard',{})
            if (record.get('exhaustive') is not False or type(shard.get('count')) is not int or shard['count']<2
                or type(shard.get('index')) is not int or not 0<=shard['index']<shard['count']
                or record.get('contract_hash')!=tree.contract.fingerprint or record.get('scope')!=parameters
                or record.get('promise')!=tree.contract.promise):
                raise StrataError('sample lab binding mismatch')
        return record

    def validate_lab(self, lab):
        binding = lab.get('binding', {})
        if binding.get('adapter_hash') != adapter_hash(self.adapter()) or binding.get('task_hash') != self.task_hash():
            raise StrataError('lab adapter or task binding mismatch')
        if lab.get('contract_hash') != binding.get('contract'):
            raise StrataError('lab contract binding mismatch')
        if lab.get('evidence_kind') == 'component_scoped':
            from .component_lab import validate_component_lab
            validate_component_lab(lab, self.frontend)
            if lab.get('scope') != binding.get('parameters') or lab.get('circuit_hash') != binding.get('artifact'):
                raise StrataError('component lab parameter or artifact binding mismatch')
        elif lab.get('evidence_kind') == 'full_artifact':
            receipt = lab.get('receipt', {})
            scope = dict(parameters=binding.get('parameters'), domain=self.config.get('domain', {}))
            expected = dict(stage='lab', artifact_hash=binding.get('artifact'), parameters_hash=digest(binding.get('parameters')),
                            domain_hash=digest(self.config.get('domain', {})), scope='full_artifact', passed=True)
            if any(receipt.get(k) != value for k, value in expected.items()) or lab.get('scope') != scope or lab.get('passed') is not True:
                raise StrataError('invalid full-artifact lab receipt')
            if not lab.get('verifier_hash') or lab['verifier_hash'] != receipt.get('verifier_hash'):
                raise StrataError('unbound full-artifact lab verifier')
            if lab.get('circuit_hash') != binding.get('artifact') or lab.get('exhaustive') is not False:
                raise StrataError('full-artifact check cannot claim an exhaustive small model')
        else:
            from .lab import validate_lab_receipt
            validate_lab_receipt(lab)
            if lab.get('scope') != binding.get('parameters') or lab.get('promise') != binding.get('promise'):
                raise StrataError('lab parameter or promise scope mismatch')

    def verify_built(self, circuit, parameters, adapter):
        forbidden = set(self.config.get('forbid', []))
        if hasattr(self.frontend, 'operations'):
            operations = self.frontend.operations(circuit)
        else:
            from .kickmix import instructions
            operations = (op for _, _, op, _ in instructions(circuit))
        if any(op in forbidden for op in operations):
            raise StrataError('forbidden gate')
        self.measured(circuit)
        oracle = adapter.oracle(circuit, parameters, self.config.get('domain', {}))
        if oracle.get('passed') is not True:
            raise StrataError('full oracle failed')
        receipt = adapter.verify(circuit, parameters, self.config.get('domain', {}))
        with preserve_completed_check(self.project,self.frontend,circuit,parameters,self.config.get('domain',{}),receipt):
            if receipt.get('passed') is not True or not receipt.get('verifier_hash'):
                raise StrataError('official verifier failed or unpinned')
            required = self.config.get('domain', {}).get('seeds', [])
            if required and (receipt.get('seeds') != required or oracle.get('seeds') != required):
                raise StrataError('draws differ from precommitted seeds')
            resources = receipt['resources']
            self.check_resources(circuit, resources)
            priced=self.price_resources(resources,receipt)
            policy=self.probability_policy()
            if policy:
                policy.validate_selection(self.probability_data(oracle))
                for draw in receipt.get('draws',receipt.get('details',{}).get('draws',[])):
                    self.check_resources(circuit,draw['resources'])
        return dict(oracle=oracle, official=receipt, **priced,
                    binding=dict(artifact=hashlib.sha256(self.serialize(circuit)).hexdigest(), parameters=parameters,
                                 domain=self.config.get('domain', {}), task_hash=self.task_hash(), adapter_hash=adapter_hash(adapter)))

    def ingest(self):
        adapter = self.adapter()
        base = self.project.root / self.config['base']
        input_manifest = manifest(base)
        circuit = self.frontend.load(base)
        parameters = self.params()
        built, tree = adapter.build(parameters, {})
        if self.serialize(built) != self.serialize(circuit):
            raise StrataError('generator does not reproduce the supplied base')
        small_c, small_tree = adapter.build(self.params(True), {})
        lab = self.run_lab(small_c, small_tree, self.params(True))
        if not lab.passed:
            raise StrataError('base small model fails')
        lab_record = self.lab_record(lab, small_c, small_tree, self.params(True))
        self.validate_lab(lab_record)
        measured_resources = adapter.measure(circuit, parameters, self.config.get('domain', {}))
        measurement_receipt = getattr(adapter, 'measurement_receipt', None)
        with preserve_completed_check(self.project,self.frontend,circuit,parameters,self.config.get('domain',{}),measurement_receipt):
            self.check_resources(circuit, measured_resources)
            measured_cost = self.price_resources(measured_resources,measurement_receipt)['cost']
        full = self.verify_built(circuit, parameters, adapter)
        if measured_cost != full['cost']:
            raise StrataError('build price disagrees with full verifier')
        artifact = self.project.store.blob(self.serialize(circuit))
        cert = self.project.store.put('certificate', dict(passed=True, artifact=artifact, lab=lab_record,
                full=full, build_price=dict(passed=True, artifact=artifact, measured_resources=measured_resources, measured_cost=measured_cost, measurement_receipt=measurement_receipt),
                verifier_hash=full['official']['verifier_hash'], adapter_hash=adapter_hash(adapter), task_hash=self.task_hash(),
                failure_model=full['official'].get('failure_model', {'status': 'unspecified'}),
                binding=dict(task=self.id, tree=digest(tree), parameters=parameters, implementation={}, parent=None)))
        record = dict(task=self.id, parent=None, tree=asdict(tree), architecture=tree.architecture,
                      parameters=parameters, implementation={}, artifact=artifact, certificate=cert, input_manifest=input_manifest,
                      adapter_hash=adapter_hash(adapter), task_hash=self.task_hash(), cost=full['cost'], resources=full['official']['resources'])
        verify_manifest(base, input_manifest)
        return self.project.store.put('design', record)

    def base_design(self):
        for h, d in self.project.store.all('design'):
            if d['task'] == self.id and d['parent'] is None:
                return Design(self.project, h)
        raise StrataError('task not ingested')


class Design:
    def __init__(self, project, id):
        self.project, self.id = project, id
        self.record = project.store.get(id, 'design')
        self.tree = region_from_dict(self.record['tree'])

    def validate(self, seen=None):
        seen=set() if seen is None else seen
        if self.id in seen:raise StrataError('cyclic design ancestry')
        seen.add(self.id)
        d=self.record
        cert=self.project.store.get(d['certificate'],'certificate')
        lab, full=cert.get('lab',{}),cert.get('full',{})
        expected_binding=dict(task=d['task'], tree=digest(d['tree']), parameters=d['parameters'], implementation=d['implementation'], parent=d.get('parent'))
        if digest(cert.get('binding'))!=digest(expected_binding):
            raise StrataError('certificate design metadata mismatch')
        task = self.project.task(d['task'])
        if cert.get('passed') is not True or lab.get('passed') is not True:
            raise StrataError('certificate lacks passing lab')
        task.validate_lab(lab)
        if d.get('task_hash') != task.task_hash() or cert.get('task_hash') != d['task_hash']:
            raise StrataError('task configuration or frontend changed')
        if d['adapter_hash'] != adapter_hash(task.adapter()):
            raise StrataError('adapter changed since certification')
        expected_full = dict(artifact=d['artifact'], parameters=d['parameters'], domain=task.config.get('domain', {}),
                             task_hash=d['task_hash'], adapter_hash=d['adapter_hash'])
        if digest(full.get('binding')) != digest(expected_full):
            raise StrataError('full verifier artifact or domain binding mismatch')
        if full.get('oracle',{}).get('passed') is not True or full.get('official',{}).get('passed') is not True:
            raise StrataError('certificate lacks full verification')
        if task.probability_policy():
            task.probability_policy().validate_selection(task.probability_data(full['oracle']))
        for name,stage in (('oracle','oracle_full'),('official','official_verifier')):
            inner=full[name]
            if 'artifact_hash' in inner:
                expected=dict(artifact_hash=d['artifact'],parameters_hash=digest(d['parameters']),
                              domain_hash=digest(task.config.get('domain',{})),scope='full_artifact',stage=stage)
                if any(inner.get(k)!=v for k,v in expected.items()):raise StrataError('inner verifier receipt scope mismatch')
        if cert.get('artifact')!=d['artifact'] or cert.get('adapter_hash')!=d['adapter_hash']:
            raise StrataError('certificate artifact/adapter mismatch')
        if cert.get('verifier_hash')!=full['official'].get('verifier_hash'):
            raise StrataError('verifier binding mismatch')
        if d['cost']!=full.get('cost') or d['resources']!=full['official'].get('resources'):
            raise StrataError('design price does not match certificate')
        repriced=task.price_resources(d['resources'],full['official'])
        if repriced['cost'] != d['cost'] or repriced['probability']!=full.get('probability'):
            raise StrataError('project cost formula changed since certification')
        if d['architecture']!=self.tree.architecture:
            raise StrataError('architecture fingerprint mismatch')
        artifact = task.parse_artifact(self.project.store.read_blob(d['artifact']))
        if hasattr(task.frontend, 'validate_tree'):
            task.frontend.validate_tree(self.tree)
        task.check_resources(artifact, d['resources'])
        price = cert.get('build_price', {})
        if price.get('passed') is not True or price.get('artifact') != d['artifact'] or price.get('measured_cost') != d['cost']:
            raise StrataError('certificate lacks a matching measured build price')
        task.check_resources(artifact, price.get('measured_resources'))
        if price.get('measurement_receipt') is not None:
            receipt = price['measurement_receipt']
            expected = dict(stage='build_price', passed=True, artifact_hash=d['artifact'], parameters_hash=digest(d['parameters']),
                            domain_hash=digest(task.config.get('domain', {})), scope='full_artifact', resources=price['measured_resources'])
            if any(receipt.get(key) != value for key, value in expected.items()) or not receipt.get('verifier_hash'):
                raise StrataError('measurement receipt scope mismatch')
        if task.price_resources(price['measured_resources'],price.get('measurement_receipt'))['cost'] != price['measured_cost']:
            raise StrataError('build price receipt is inconsistent')
        if not task.config.get('scale') and lab['binding']['artifact'] != d['artifact']:
            raise StrataError('unscaled lab checked a different artifact')
        if d.get('parent'):
            parent=Design(self.project,d['parent']);parent.validate(seen)
            if parent.record['task']!=d['task']:raise StrataError('cross-task ancestry')
            patch=Patch(**d['patch'])
            if not patch.applies(parent.tree):raise StrataError('ancestral patch no longer applies')
            adapter=task.adapter()
            expected=adapter.merge_implementation(parent.record['implementation'],patch.implementation) if hasattr(adapter,'merge_implementation') else {**parent.record['implementation'],**patch.implementation}
            if digest(expected)!=digest(d['implementation']):raise StrataError('patch chain implementation mismatch')
        return True

    def gaps(self, top=5):
        task = self.project.task(self.record['task'])
        adapter = task.adapter()
        circuit, _ = adapter.build(self.record['parameters'], self.record['implementation'])
        ledgers = adapter.ledger(circuit, self.tree)
        rows = rank_gaps(self.tree, ledgers, adapter.floors(self.tree), task.formula(), self.record['resources'])[:top]
        self.project.store.put('gap_snapshot', dict(design=self.id, task=task.id, gaps=rows))
        return rows

    def price(self, region, resource, delta):
        self.tree.find(region)
        task = self.project.task(self.record['task'])
        adapter = task.adapter()
        constraint = dict(region=region, resource=resource, delta=delta)
        key = digest(dict(design=self.id, constraint=constraint, adapter=adapter_hash(adapter)))
        for _, result in self.project.store.all('price'):
            if result['key'] == key:
                return result
        if not hasattr(adapter, 'constrain'):
            raise StrataError('no constrained generator; cannot estimate a measured price')
        before, _ = adapter.build(self.record['parameters'], self.record['implementation'])
        after, _ = adapter.constrain(self.record['parameters'], self.record['implementation'], constraint)
        # Price a real build and verify feasibility; do not silently promote it.
        if resource in {'wire', 'helper_wire'} and after.num_qubits-before.num_qubits != delta:
            raise StrataError('constrained build did not achieve the requested wire delta')
        checked = task.verify_built(after, self.record['parameters'], adapter)
        old = task.verify_built(before, self.record['parameters'], adapter)
        result = dict(key=key, constraint=constraint, before=old, after=checked,
                      delta=checked['cost'] - old['cost'], kind='measured_build')
        self.project.store.put('price', result)
        return result


class Line:
    def __init__(self, project, id):
        self.project, self.id = project, id
        self.record = project.store.get(id, 'line')
        p = self.record['patch']
        self.patch = Patch(**{**p, **{k: tuple(p[k]) for k in ['regions', 'contracts', 'occurrences', 'reads', 'writes', 'replacement_regions']}})
        self.task = project.task(self.record['task'])

    def result(self, stage):
        events = [x for x in self.project.store.events(self.id) if x['stage'] == stage]
        if not events or not events[-1].get('passed'):
            raise StrataError(f'gate {stage} has not passed')
        return events[-1]

    def context(self):
        if digest(asdict(self.patch)) != digest(self.project.store.get(self.id, 'line')['patch']):
            raise StrataError('line implementation mutated after it was frozen')
        adapter = self.task.adapter()
        if adapter_hash(adapter) != self.record['adapter_hash']:
            raise StrataError('adapter changed after line opened')
        parent = Design(self.project, self.record['parent'])
        parent.validate()
        if not self.patch.applies(parent.tree):
            raise StrataError('patch stale')
        impl = adapter.merge_implementation(parent.record['implementation'],self.patch.implementation) if hasattr(adapter,'merge_implementation') else {**parent.record['implementation'], **self.patch.implementation}
        return adapter, parent, impl

    def validate_edit(self, old_circuit, old_tree, circuit, tree):
        def public(c):
            return [(r.name, len(r.wires), r.role, r.meaning, r.namespace) for r in c.registers if r.role != 'clean']
        if public(old_tree.contract) != public(tree.contract):
            raise StrataError('external register contract changed')
        if (old_tree.contract.reference, old_tree.contract.promise, old_tree.contract.phase, old_tree.contract.failure) != (tree.contract.reference, tree.contract.promise, tree.contract.phase, tree.contract.failure):
            raise StrataError('external specification changed')
        if hasattr(self.task.frontend, 'validate_edit'):
            self.task.frontend.validate_edit(old_circuit, old_tree, circuit, tree, self.patch.regions, self.patch.replacement_regions)
            return
        touched=[old_tree.find(name) for name in self.patch.regions if any(r.id==name for r in old_tree.walk())]
        # For scaled instances some occurrences may not exist. The full build
        # separately checks every occurrence; the root boundary remains mandatory.
        old_lines, new_lines = str(old_circuit).splitlines(), str(circuit).splitlines()
        new_touched=[tree.find(name) for name in self.patch.replacement_regions] if self.patch.replacement_regions else [tree.find(r.id) for r in touched]
        if self.patch.depth in {'L0','L1','L2'} and self.patch.replacement_regions:
            raise StrataError('replacement region topology requires L3/L4')
        if self.patch.depth in {'L0','L1','L2'}:
            for old,new in zip(touched,new_touched):
                if old.contract.fingerprint!=new.contract.fingerprint:
                    raise StrataError('L2 requires the same region contract')
        def outside(lines, regions):
            return [line for i,line in enumerate(lines) if not any(r.start<=i<r.stop for r in regions)]
        if outside(old_lines,touched)!=outside(new_lines,new_touched):
            raise StrataError('patch added or changed operations outside declared regions')
        for r in old_tree.walk():
            if any(r.start < t.stop and t.start < r.stop for t in touched):continue
            other=tree.find(r.id)
            if r.contract.fingerprint!=other.contract.fingerprint or old_lines[r.start:r.stop]!=new_lines[other.start:other.stop]:
                raise StrataError('patch changed an undeclared region: '+r.id)

    @retain_external_failure('lab')
    def lab(self, *, limits=None, **scale):
        adapter, parent, impl = self.context()
        params = self.task.params(True)
        params.update(scale)
        for k, value in params.items():
            if k not in self.task.config.get('scale', {}) or value not in self.task.config['scale'][k]['small']:
                raise StrataError('scale not declared in project')
        circuit, tree = adapter.build(params, impl)
        old, old_tree = adapter.build(params, parent.record['implementation'])
        # Reference/promise/failure cannot be quietly weakened by the builder.
        if (tree.contract.reference, tree.contract.promise, tree.contract.phase, tree.contract.failure) != (old_tree.contract.reference, old_tree.contract.promise, old_tree.contract.phase, old_tree.contract.failure):
            raise StrataError('external specification changed')
        self.validate_edit(old, old_tree, circuit, tree)
        result = self.task.run_lab(circuit, tree, params, limits)
        event = dict(stage='lab', **self.task.lab_record(result, circuit, tree, params), implementation=digest(impl))
        self.project.store.event(self.id, event)
        return result

    @retain_external_failure('build_price')
    def build(self):
        lab = self.result('lab')
        self.task.validate_lab(lab)
        adapter, parent, impl = self.context()
        if lab['implementation'] != digest(impl):
            raise StrataError('lab describes a different implementation')
        circuit, tree = adapter.build(parent.record['parameters'], impl)
        old_circuit, old_tree = adapter.build(parent.record['parameters'], parent.record['implementation'])
        self.validate_edit(old_circuit, old_tree, circuit, tree)
        metrics = self.task.measured(circuit)
        if not hasattr(adapter, 'measure'):
            raise StrataError('build-price gate needs an executed-cost measurement adapter')
        charged = adapter.measure(circuit, parent.record['parameters'], self.task.config.get('domain', {}))
        measurement_receipt=getattr(adapter,'measurement_receipt',None)
        with preserve_completed_check(self.project,self.task.frontend,circuit,parent.record['parameters'],self.task.config.get('domain',{}),measurement_receipt):
            self.task.check_resources(circuit, charged)
            measured_cost = self.task.price_resources(charged,measurement_receipt)['cost']
        cap = self.task.config.get('max_qubits')
        if cap is not None and metrics['Q'] > cap:
            raise StrataError('build exceeds hard width')
        artifact = self.project.store.blob(self.task.serialize(circuit))
        if not self.task.config.get('scale') and lab['binding']['artifact'] != artifact:
            raise StrataError('lab checked a different artifact')
        result = dict(stage='build_price', passed=True, artifact=artifact, tree=asdict(tree),
                      metrics=metrics, implementation=impl, parameters=parent.record['parameters'],
                      price_status='measured build', measurement_receipt=getattr(adapter, 'measurement_receipt', None), measured_resources=charged, measured_cost=measured_cost, delta=measured_cost-parent.record['cost'])
        self.project.store.event(self.id, result)
        return result

    @retain_external_failure('verified')
    def verify(self):
        lab, build = self.result('lab'), self.result('build_price')
        adapter, parent, impl = self.context()
        if lab['implementation'] != digest(impl) or digest(build['implementation']) != digest(impl):
            raise StrataError('verification evidence describes another implementation')
        circuit = self.task.parse_artifact(self.project.store.read_blob(build['artifact']))
        self.task.validate_lab(lab)
        rebuilt, rebuilt_tree = adapter.build(build['parameters'], impl)
        if self.task.serialize(rebuilt) != self.task.serialize(circuit) or digest(rebuilt_tree) != digest(build['tree']):
            raise StrataError('build output or tree changed before verification')
        checked = self.task.verify_built(circuit, build['parameters'], adapter)
        if checked['cost'] != build['measured_cost']:
            raise StrataError('build price disagrees with full verifier')
        cert = self.project.store.put('certificate', dict(passed=True, artifact=build['artifact'], lab=lab,
                    full=checked, build_price=build, adapter_hash=adapter_hash(adapter), task_hash=self.task.task_hash(), verifier_hash=checked['official']['verifier_hash'],
                    failure_model=checked['official'].get('failure_model', {'status': 'unspecified'}),
                    binding=dict(task=self.task.id, tree=digest(build['tree']), parameters=build['parameters'], implementation=impl, parent=parent.id)))
        tree = region_from_dict(build['tree'])
        record = dict(task=self.task.id, parent=parent.id, patch=asdict(self.patch), depth=self.patch.depth,
                      tree=build['tree'], architecture=tree.architecture, parameters=build['parameters'],
                      implementation=impl, artifact=build['artifact'], certificate=cert,
                      adapter_hash=adapter_hash(adapter), task_hash=self.task.task_hash(), cost=checked['cost'], resources=checked['official']['resources'])
        design = self.project.store.put('design', record)
        self.project.store.event(self.id, dict(stage='verified', passed=True, design=design, certificate=cert))
        return Design(self.project, design)

    def park(self, reason, reopen, evidence='estimate', scale=None, evidence_artifact=None):
        for fp in self.patch.contracts:
            self.project.store.negative(fp, self.patch.move, evidence, reason, reopen, scale, evidence=evidence_artifact)
        self.project.store.event(self.id, dict(stage='parked', passed=False, reason=reason, reopen=reopen))

    def status(self):
        return dict(id=self.id, parent=self.record['parent'], events=self.project.store.events(self.id))
