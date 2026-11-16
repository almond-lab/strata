"""Typed moves, immutable patch applicability, dependency-aware composition."""
from dataclasses import dataclass, asdict
from .model import Depth, StrataError

MOVES = {
 'replace-stages': Depth.ARCHITECTURE, 're-represent': Depth.BOUNDARY,
 'move-debt': Depth.BOUNDARY, 'fuse': Depth.BOUNDARY, 'split': Depth.BOUNDARY,
 'trade-wires': Depth.REGION, 'host-guest': Depth.BOUNDARY,
 'relax-exactness': Depth.BOUNDARY, 'co-design-numerics': Depth.BOUNDARY,
 're-synthesise': Depth.REGION,
}


@dataclass(frozen=True)
class Patch:
    move: str
    regions: tuple[str, ...]
    contracts: tuple[str, ...]
    occurrences: tuple[int, ...]
    implementation: dict
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    replacement_regions: tuple[str, ...] = ()

    def __post_init__(self):
        if self.move not in MOVES or not self.regions:
            raise StrataError('invalid typed move')
        if len(self.regions) != len(set(self.regions)) or not (len(self.regions) == len(self.contracts) == len(self.occurrences)):
            raise StrataError('invalid patch targets')

    @property
    def depth(self):
        return MOVES[self.move].value

    def applies(self, tree):
        for name, fp, occurrence in zip(self.regions, self.contracts, self.occurrences):
            try:
                r = tree.find(name)
            except StrataError:
                return False
            if r.contract.fingerprint != fp or r.occurrence != occurrence:
                return False
        return True


def compose(a, b, tree):
    if not a.applies(tree) or not b.applies(tree):
        raise StrataError('patch does not apply to this contract/occurrence')
    for x in a.regions:
        for y in b.regions:
            r, s = tree.find(x), tree.find(y)
            if r.start < s.stop and s.start < r.stop:
                raise StrataError('overlapping patches stay siblings')
    if set(a.writes) & (set(b.reads) | set(b.writes)) or set(b.writes) & set(a.reads):
        raise StrataError('shared boundary/debt dependencies require joint verification')
    return [asdict(a), asdict(b)]  # combined build and certificate still mandatory


def applicable_moves(tree):
    """Proposals are obligations; absence of a queried proof is not permission."""
    out = []
    for r in tree.walk():
        out.append(dict(move='re-synthesise', regions=[r.id], ready=r.kind != 'opaque',
                        requires=['same external contract', 'exhaustive lab', 'measured build']))
        for a, b in zip(r.children, r.children[1:]):
            if a.stop == b.start:
                out.append(dict(move='fuse', regions=[a.id, b.id], ready=True,
                                requires=['joint boundary', 'all debts', 'combined build']))
        if r.children:
            out.append(dict(move='replace-stages', regions=[r.id], ready=True,
                            requires=['same task function and promise', 'new stage list', 'lab', 'full verification']))
        if r.contract.debts:
            out.append(dict(move='move-debt', regions=[r.id], ready=False,
                            requires=['live operand ownership proof', 'all-branch phase repayment']))
    return out
