"""kickmix front end. Large reports are bounded; unsupported evidence is explicit."""
from pathlib import Path
from collections import Counter
import hashlib
import re
import shlex
import html
import kickmix as km
from .model import Contract, Register, Region, StrataError

PAID = {'CCX', 'CCZ', 'CSWAP'}
META = {'REGISTER', 'APPEND_TO_REGISTER', 'INFO_TAG', 'DEBUG_PRINT'}


def instructions(circuit):
    for index, line in enumerate(str(circuit).splitlines()):
        words = shlex.split(line)
        if words:
            yield index, line, words[0], tuple(int(x[1:]) for x in words[1:] if re.fullmatch(r'q\d+', x))


class Kickmix:
    def load(self, path):
        return km.Circuit.from_file(str(path))

    def parse(self, text):
        return km.Circuit(text)

    def measured(self, circuit):
        s = circuit.operation_statistics()
        return dict(Q=circuit.num_qubits, toffoli=s.num_toffoli_like,
                    t=s.num_t_like, rotations=sum(1 for _, _, op, _ in instructions(circuit) if op == 'Z_POW'),
                    instructions=len(circuit), accounting='static_emitted',
                    reaction_depth=circuit.reaction_depth())

    def analyze(self, circuit, out, max_ops=100_000):
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        metrics = self.measured(circuit)
        # A conservative symbolic zero analysis, NOT exact state-dependent liveness.
        unknown = set(q.id for wires in circuit.register_data.values() for q in wires if isinstance(q, km.q))
        if not unknown:
            unknown = set(range(circuit.num_qubits))
        live, tags, conditions, debt_sites, census = [], [], [], [], Counter()
        first, last = {}, {}
        lines = []
        for i, line, op, qs in instructions(circuit):
            if i >= max_ops:
                break
            census[op] += 1
            lines.append(line)
            for q in qs:
                first.setdefault(q, i)
                last[q] = i
            if op == 'INFO_TAG':
                tags.append(dict(at=i, text=shlex.split(line)[1], trusted=False))
            if ' if ' in line or op in {'PUSH_CONDITION', 'POP_CONDITION'}:
                conditions.append(dict(at=i, instruction=line))
            if op == 'HMR':
                debt_sites.append(dict(at=i, wire=qs[0], status='repayment_unresolved'))
                if ' if ' not in line:
                    unknown.discard(qs[0])
            elif op in {'X', 'CX', 'CCX', 'SWAP', 'CSWAP', 'R'}:
                # R is not evidence of a legal clean release.
                unknown.update(qs)
            live.append(dict(at=i, may_be_nonzero=len(unknown)))
        truncated = len(circuit) > max_ops
        pack = dict(metrics=metrics, census=dict(census), coverage='prefix' if truncated else 'full',
                    scanned=len(lines), liveness=dict(kind='conservative_may_nonzero', trace=live,
                    peak=max((x['may_be_nonzero'] for x in live), default=0)),
                    allocation=dict(kind='wire_access_intervals_not_object_ownership',
                    intervals={str(q): [first[q], last[q]] for q in first}),
                    conditionals=conditions, debts=debt_sites, tags=tags,
                    executed_cost=dict(status='requires_domain_runner', reason='static is not mean executed'),
                    fingerprints=dict(status='requires_domain_runner', scope='sampled matches never prove equivalence'),
                    slack=dict(status='unknown', reason='no information-rank proof supplied'))
        if not truncated:
            (out / 'flame.svg').write_text(circuit.flame_chart_svg())
        else:
            pack['flame'] = dict(status='skipped_size_budget')
        # Bounded actual SDK diagram, not a fabricated whole-circuit image.
        excerpt = km.Circuit('\n'.join(lines[:min(100, len(lines))]))
        try:
            (out / 'diagram.html').write_text(excerpt.html_diagram())
        except (ValueError, RuntimeError) as exc:
            pack['diagram'] = dict(status='unavailable', reason=str(exc))
        import json
        (out / 'analysis.json').write_text(json.dumps(pack, indent=2))
        (out / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Strata analysis</title>'
             '<h1>Strata analysis</h1><p>Static emitted counts; conservative liveness. Missing evidence is marked.</p>'
             '<a href="analysis.json">Structured data</a> · <a href="flame.svg">SDK flame chart</a> · '
             '<a href="diagram.html">Bounded SDK diagram</a><pre>' + html.escape(json.dumps(metrics, indent=2)) + '</pre>')
        return pack

    def lift(self, circuit):
        """Honest initial recognizer: tags are hints, primitive semantics only are typed.

        Higher-level matches require a supplied semantic equivalence certificate.
        """
        ops = list(instructions(circuit))
        if not ops:
            raise StrataError('cannot lift empty circuit')
        regions = []
        typed_t, total_t = 0, 0
        for i, line, op, qs in ops:
            if op in META:
                continue
            total_t += int(op in PAID)
            # Single native operations have known semantics; never call them adders/GF products.
            kind = {'CX': 'xor', 'X': 'not', 'SWAP': 'swap', 'CCX': 'controlled_xor',
                    'CZ': 'quadratic_phase', 'CCZ': 'cubic_phase'}.get(op, 'opaque')
            contract = Contract('primitive', tuple(Register(f'q{k}', (q,), meaning='bit') for k, q in enumerate(qs)),
                                reference=f'kickmix:{line}', phase='exact')
            regions.append(Region(f'op-{i}', i, i + 1, contract, kind=kind,
                                  evidence='native_primitive_semantics' if kind != 'opaque' else 'untyped'))
            typed_t += int(op in PAID and kind != 'opaque')
        root = Contract('circuit', (Register('state', tuple(range(circuit.num_qubits))),),
                        reference='artifact:' + hashlib.sha256(str(circuit).encode()).hexdigest())
        tree = Region('root', 0, len(ops), root, tuple(regions))
        return tree, dict(primitive_paid_coverage=typed_t / max(1, total_t),
                          structural_typed_coverage=0.0, confirmed=False,
                          note='Primitive coverage does not satisfy the 90% structural lifting target.')
