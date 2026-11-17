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
    from .streaming import native_lines
    with native_lines(circuit) as lines:
        for index, raw in enumerate(lines):
            line = raw.rstrip('\r\n')
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
                    t=s.num_t_like, rotations=s['Z_POW'],
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
        """Infer exact algebraic regions, leaving task-level recognition explicit."""
        from .semantic import lift_regions
        from .file_workflow import contract_tree
        report=lift_regions(circuit,max_window=256,max_variables=64)
        tree=contract_tree(circuit,report)
        coverage=report['coverage']
        return tree,dict(primitive_paid_coverage=coverage['exact_paid_fraction'],
                         compound_algebra_paid_coverage=coverage['compound_algebra_paid_fraction'],
                         structural_typed_coverage=0.0, high_level_stage_coverage=None,
                         algebraic_coverage=coverage, confirmed=False,
                         note='Exact algebraic functions are inferred; arithmetic-stage coverage remains unestablished.')
