"""Format-neutral, immutable contract and evidence vocabulary."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import json


def canonical(value):
    if hasattr(value, '__dataclass_fields__'):
        value = asdict(value)
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class StrataError(ValueError):
    pass


class Depth(str, Enum):
    REWRITE = 'L0'
    TUNE = 'L1'
    REGION = 'L2'
    BOUNDARY = 'L3'
    ARCHITECTURE = 'L4'


@dataclass(frozen=True)
class Register:
    name: str
    wires: tuple[int, ...]
    role: str = 'through'  # through, input, output, clean, dirty
    meaning: str = 'bits'

    def __post_init__(self):
        if self.role not in {'through', 'input', 'output', 'clean', 'dirty'}:
            raise StrataError('unknown register role')
        if not self.wires or min(self.wires) < 0 or len(set(self.wires)) != len(self.wires):
            raise StrataError('invalid register wires')


@dataclass(frozen=True)
class Debt:
    id: str
    created_by: str
    repaid_by: str
    measurement: str
    expression: str


@dataclass(frozen=True)
class Contract:
    name: str
    registers: tuple[Register, ...]
    reference: str
    promise: str = 'all'
    phase: str = 'exact'
    failure: str = 'none'
    scales: dict = field(default_factory=dict)
    debts: tuple[Debt, ...] = ()
    semantic_version: str = '1'

    def __post_init__(self):
        wires = [q for r in self.registers for q in r.wires]
        if len(wires) != len(set(wires)):
            raise StrataError('overlapping register ownership')
        if len({r.name for r in self.registers}) != len(self.registers):
            raise StrataError('duplicate register names')
        if not self.reference:
            raise StrataError('contract requires a reference identifier')

    @property
    def fingerprint(self):
        # Physical lanes/names are excluded; logical argument order and meaning remain.
        return digest(dict(registers=[dict(width=len(r.wires), role=r.role,
                           meaning=r.meaning) for r in self.registers],
                           reference=self.reference, promise=self.promise,
                           phase=self.phase, failure=self.failure, scales=self.scales,
                           debts=[dict(measurement=d.measurement, expression=d.expression)
                                  for d in self.debts], version=self.semantic_version))


@dataclass(frozen=True)
class Region:
    id: str
    start: int
    stop: int
    contract: Contract
    children: tuple['Region', ...] = ()
    kind: str = 'opaque'
    evidence: str = 'untyped'
    occurrence: int = 0

    def __post_init__(self):
        if self.start < 0 or self.stop <= self.start:
            raise StrataError('empty or invalid region')
        ordered = sorted(self.children, key=lambda r: r.start)
        if any(r.start < self.start or r.stop > self.stop for r in ordered):
            raise StrataError('child outside parent')
        if any(a.stop > b.start for a, b in zip(ordered, ordered[1:])):
            raise StrataError('overlapping sibling regions')
        ids = [r.id for r in self.walk()]
        if len(ids) != len(set(ids)):
            raise StrataError('duplicate region ids')

    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()

    @property
    def architecture(self):
        return digest(dict(contract=self.contract.fingerprint, kind=self.kind,
                           children=[c.architecture for c in self.children]))

    def find(self, name):
        for r in self.walk():
            if r.id == name:
                return r
        raise StrataError(f'unknown region {name}')


@dataclass(frozen=True)
class Floor:
    contract: str
    resource: str
    value: float
    kind: str  # proof, counting, library, exact_small
    evidence: str
    scale: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in {'proof', 'counting', 'library', 'exact_small'} or self.value < 0:
            raise StrataError('invalid floor')
        if not self.evidence:
            raise StrataError('floor requires evidence')
        if self.kind == 'exact_small' and not self.scale:
            raise StrataError('exact_small requires an explicit size')

    @property
    def is_full_lower_bound(self):
        return self.kind in {'proof', 'counting'} and not self.scale


def region_from_dict(d):
    c = d['contract']
    contract = Contract(**{**c, 'registers': tuple(Register(**{**r, 'wires': tuple(r['wires'])}) for r in c['registers']),
                           'debts': tuple(Debt(**x) for x in c.get('debts', []))})
    return Region(**{**d, 'contract': contract, 'children': tuple(region_from_dict(x) for x in d.get('children', []))})
