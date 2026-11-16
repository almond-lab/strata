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
from .engine import Patch, MOVES, applicable_moves, compose
from .kickmix import Kickmix
from .lab import exhaustive
from .artifacts import manifest, verify_manifest


def adapter_hash(adapter):
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
        artifact = hashlib.sha256(str(built).encode()).hexdigest()
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
        impl = dict(parent.record['implementation'])
        for patch in patches:
            for key, value in patch['implementation'].items():
                if key in impl and key in a.patch.implementation and key in b.patch.implementation and a.patch.implementation[key] != b.patch.implementation[key]:
                    raise StrataError('generator parameter conflict')
                impl[key] = value
        # Returns an unverified line. Lab/build/full gates must run on the combination.
        return self.open_line(parent, 'replace-stages', ['root'], 'composition: ' + first + ' + ' + second, impl)


class Task:
    def __init__(self, project, config):
        self.project, self.config, self.id = project, config, config['id']

    def adapter(self):
        name = self.config.get('adapter', self.id)
        if name not in self.project.adapters:
            raise StrataError(f'task {self.id} needs adapter {name!r}; cannot invent a scalable model or verifier')
        return self.project.adapters[name]

    def formula(self):
        return Formula(self.config['cost'], getattr(self.adapter(), 'cost_functions', {}))

    def params(self, small=False):
        return {k: v['small'][0] if small else v['full'] for k, v in self.config.get('scale', {}).items()}

    def verify_built(self, circuit, parameters, adapter):
        forbidden = set(self.config.get('forbid', []))
        from .kickmix import instructions
        if any(op in forbidden for _, _, op, _ in instructions(circuit)):
            raise StrataError('forbidden gate')
        limit = self.config.get('max_qubits')
        if limit is not None and circuit.num_qubits > limit:
            raise StrataError('qubit budget exceeded')
        oracle = adapter.oracle(circuit, parameters, self.config.get('domain', {}))
        if oracle.get('passed') is not True:
            raise StrataError('full oracle failed')
        receipt = adapter.verify(circuit, parameters, self.config.get('domain', {}))
        if receipt.get('passed') is not True or not receipt.get('verifier_hash'):
            raise StrataError('official verifier failed or unpinned')
        required = self.config.get('domain', {}).get('seeds', [])
        if required and (receipt.get('seeds') != required or oracle.get('seeds') != required):
            raise StrataError('draws differ from precommitted seeds')
        resources = receipt['resources']
        if 'Q' in resources and resources['Q'] != circuit.num_qubits:
            raise StrataError('verifier width disagrees with emitted circuit')
        from .acceptance import check_acceptance
        check_acceptance(self.config.get('accept', {}), resources)
        cost = self.formula()(resources)
        return dict(oracle=oracle, official=receipt, cost=cost)

    def ingest(self):
        adapter = self.adapter()
        base = self.project.root / self.config['base']
        input_manifest = manifest(base)
        circuit = self.project.frontend.load(base)
        parameters = self.params()
        built, tree = adapter.build(parameters, {})
        if str(built) != str(circuit):
            raise StrataError('generator does not reproduce the supplied base')
        small_c, small_tree = adapter.build(self.params(True), {})
        lab = exhaustive(small_c, small_tree.contract, adapter.reference, self.params(True))
        if not lab.passed:
            raise StrataError('base small model fails')
        full = self.verify_built(circuit, parameters, adapter)
        artifact = self.project.store.blob(str(circuit).encode())
        cert = self.project.store.put('certificate', dict(passed=True, artifact=artifact, lab=asdict(lab),
                full=full, verifier_hash=full['official']['verifier_hash'], adapter_hash=adapter_hash(adapter),
                failure_model=full['official'].get('failure_model', {'status': 'unspecified'}),
                binding=dict(task=self.id, tree=digest(tree), parameters=parameters, implementation={}, parent=None)))
        record = dict(task=self.id, parent=None, tree=asdict(tree), architecture=tree.architecture,
                      parameters=parameters, implementation={}, artifact=artifact, certificate=cert, input_manifest=input_manifest,
                      adapter_hash=adapter_hash(adapter), cost=full['cost'], resources=full['official']['resources'])
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
        if cert.get('passed') is not True or lab.get('passed') is not True or lab.get('exhaustive') is not True:
            raise StrataError('certificate lacks passing exhaustive lab')
        if lab.get('tested')!=lab.get('cases',0)*lab.get('branches',0) or lab.get('tested',0)<=0:
            raise StrataError('incomplete lab certificate')
        if full.get('oracle',{}).get('passed') is not True or full.get('official',{}).get('passed') is not True:
            raise StrataError('certificate lacks full verification')
        if cert.get('artifact')!=d['artifact'] or cert.get('adapter_hash')!=d['adapter_hash']:
            raise StrataError('certificate artifact/adapter mismatch')
        if cert.get('verifier_hash')!=full['official'].get('verifier_hash'):
            raise StrataError('verifier binding mismatch')
        if d['cost']!=full.get('cost') or d['resources']!=full['official'].get('resources'):
            raise StrataError('design price does not match certificate')
        if self.project.task(d['task']).formula()(d['resources']) != d['cost']:
            raise StrataError('project cost formula changed since certification')
        if d['architecture']!=self.tree.architecture:
            raise StrataError('architecture fingerprint mismatch')
        self.project.store.read_blob(d['artifact'])
        if d.get('parent'):
            parent=Design(self.project,d['parent']);parent.validate(seen)
            if parent.record['task']!=d['task']:raise StrataError('cross-task ancestry')
            patch=Patch(**d['patch'])
            if not patch.applies(parent.tree):raise StrataError('ancestral patch no longer applies')
            expected={**parent.record['implementation'],**patch.implementation}
            if digest(expected)!=digest(d['implementation']):raise StrataError('patch chain implementation mismatch')
        return True

    def gaps(self, top=5):
        task = self.project.task(self.record['task'])
        adapter = task.adapter()
        circuit, _ = adapter.build(self.record['parameters'], self.record['implementation'])
        ledgers = adapter.ledger(circuit, self.tree)
        return rank_gaps(self.tree, ledgers, adapter.floors(self.tree), task.formula(), self.record['resources'])[:top]

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
        if not self.patch.applies(parent.tree):
            raise StrataError('patch stale')
        impl = {**parent.record['implementation'], **self.patch.implementation}
        return adapter, parent, impl

    def validate_edit(self, old_circuit, old_tree, circuit, tree):
        def public(c):
            return [(r.name, len(r.wires), r.role, r.meaning) for r in c.registers if r.role != 'clean']
        if public(old_tree.contract) != public(tree.contract):
            raise StrataError('external register contract changed')
        if (old_tree.contract.reference, old_tree.contract.promise, old_tree.contract.phase, old_tree.contract.failure) != (tree.contract.reference, tree.contract.promise, tree.contract.phase, tree.contract.failure):
            raise StrataError('external specification changed')
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

    def lab(self, *, limits=None, **scale):
        adapter, parent, impl = self.context()
        params = self.task.params(True)
        params.update(scale)
        for k, value in params.items():
            if k not in self.task.config['scale'] or value not in self.task.config['scale'][k]['small']:
                raise StrataError('scale not declared in project')
        circuit, tree = adapter.build(params, impl)
        old, old_tree = adapter.build(params, parent.record['implementation'])
        # Reference/promise/failure cannot be quietly weakened by the builder.
        if (tree.contract.reference, tree.contract.promise, tree.contract.phase, tree.contract.failure) != (old_tree.contract.reference, old_tree.contract.promise, old_tree.contract.phase, old_tree.contract.failure):
            raise StrataError('external specification changed')
        self.validate_edit(old, old_tree, circuit, tree)
        options=limits or {}
        if set(options)-{'max_trajectories','timeout'}:raise StrataError('unknown lab limit')
        result = exhaustive(circuit, tree.contract, adapter.reference, params, **options)
        event = dict(stage='lab', **asdict(result), implementation=digest(impl))
        self.project.store.event(self.id, event)
        return result

    def build(self):
        lab = self.result('lab')
        adapter, parent, impl = self.context()
        if lab['implementation'] != digest(impl):
            raise StrataError('lab describes a different implementation')
        circuit, tree = adapter.build(parent.record['parameters'], impl)
        old_circuit, old_tree = adapter.build(parent.record['parameters'], parent.record['implementation'])
        self.validate_edit(old_circuit, old_tree, circuit, tree)
        metrics = self.project.frontend.measured(circuit)
        if not hasattr(adapter, 'measure'):
            raise StrataError('build-price gate needs an executed-cost measurement adapter')
        charged = adapter.measure(circuit, parent.record['parameters'], self.task.config.get('domain', {}))
        measured_cost = self.task.formula()(charged)
        cap = self.task.config.get('max_qubits')
        if cap is not None and metrics['Q'] > cap:
            raise StrataError('build exceeds hard width')
        artifact = self.project.store.blob(str(circuit).encode())
        result = dict(stage='build_price', passed=True, artifact=artifact, tree=asdict(tree),
                      metrics=metrics, implementation=impl, parameters=parent.record['parameters'],
                      price_status='measured build', measured_resources=charged, measured_cost=measured_cost, delta=measured_cost-parent.record['cost'])
        self.project.store.event(self.id, result)
        return result

    def verify(self):
        lab, build = self.result('lab'), self.result('build_price')
        adapter, parent, impl = self.context()
        if lab['implementation'] != digest(impl) or digest(build['implementation']) != digest(impl):
            raise StrataError('verification evidence describes another implementation')
        circuit = self.project.frontend.parse(self.project.store.read_blob(build['artifact']).decode())
        checked = self.task.verify_built(circuit, build['parameters'], adapter)
        cert = self.project.store.put('certificate', dict(passed=True, artifact=build['artifact'], lab=lab,
                    full=checked, adapter_hash=adapter_hash(adapter), verifier_hash=checked['official']['verifier_hash'],
                    failure_model=checked['official'].get('failure_model', {'status': 'unspecified'}),
                    binding=dict(task=self.task.id, tree=digest(build['tree']), parameters=build['parameters'], implementation=impl, parent=parent.id)))
        tree = region_from_dict(build['tree'])
        record = dict(task=self.task.id, parent=parent.id, patch=asdict(self.patch), depth=self.patch.depth,
                      tree=build['tree'], architecture=tree.architecture, parameters=build['parameters'],
                      implementation=impl, artifact=build['artifact'], certificate=cert,
                      adapter_hash=adapter_hash(adapter), cost=checked['cost'], resources=checked['official']['resources'])
        design = self.project.store.put('design', record)
        self.project.store.event(self.id, dict(stage='verified', passed=True, design=design, certificate=cert))
        return Design(self.project, design)

    def park(self, reason, reopen, evidence='estimate', scale=None):
        for fp in self.patch.contracts:
            self.project.store.negative(fp, self.patch.move, evidence, reason, reopen, scale)
        self.project.store.event(self.id, dict(stage='parked', passed=False, reason=reason, reopen=reopen))

    def status(self):
        return dict(id=self.id, parent=self.record['parent'], events=self.project.store.events(self.id))
