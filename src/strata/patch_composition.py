"""Compose complete, explicitly located donor diffs against a common Kickmix parent."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from .circuit_tape import CircuitTape, same_span
from .frontends import load_circuit
from .model import StrataError
from .streaming import file_hash

SCHEMA = "strata.patch-plan.v1"
COORDINATES = "kickmix-operation-0-based-half-open"
METADATA = {b"REGISTER", b"APPEND_TO_REGISTER"}


@dataclass
class Source:
    sha: str
    tape: CircuitTape
    metadata: Path


@dataclass
class Edit:
    start: int
    stop: int
    donor_start: int
    donor_stop: int
    before_sha256: str
    after_sha256: str
    donor: Source


def _source(path, directory, *, expected=None):
    """Parse the same private bytes that are hashed, even when inputs are replaced."""
    path = Path(path)
    if path.suffix.lower() not in {".kmx", ".kmb"}:
        raise StrataError("source must be a native .kmx or .kmb file")
    directory.mkdir()
    raw = directory / ("input" + path.suffix)
    shutil.copyfile(path, raw)
    fingerprint = file_hash(raw)
    if expected is not None and fingerprint != expected:
        raise StrataError(f"source SHA-256 mismatch for {path}")
    canonical = directory / "canonical.kmx"
    circuit = load_circuit(raw)
    native_count = len(circuit)
    layout = (circuit.num_qubits, circuit.num_bits, circuit.num_registers)
    circuit.to_file(canonical, format="kmx")
    del circuit
    if path.suffix.lower() == ".kmb":
        roundtrip = load_circuit(canonical)
        if (roundtrip.num_qubits, roundtrip.num_bits, roundtrip.num_registers) != layout:
            raise StrataError("binary circuit has layout outside the canonical text representation")
        del roundtrip
    raw.unlink()
    ops = directory / "operations"
    metadata = directory / "registers"
    with canonical.open("rb") as stream, ops.open("wb") as gates, metadata.open("wb") as regs:
        for line in stream:
            fields = line.split(None, 1)
            if not fields:
                continue
            if fields[0] in METADATA:
                regs.write(line)
            else:
                gates.write(line)
    canonical.unlink()
    tape = CircuitTape(ops)
    if len(tape) != native_count:
        raise StrataError("canonical instruction count differs from Kickmix operation coordinates")
    return Source(fingerprint, tape, metadata)


def _fragment(tape, start, stop):
    """SHA-256 of canonical UTF-8 instruction lines, each terminated by LF."""
    digest = sha256()
    for line in tape.iter_span(start, stop):
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _range(values, length, label):
    if (not isinstance(values, (list, tuple)) or len(values) != 2
            or any(type(i) is not int for i in values)):
        raise StrataError(f"{label} must contain two integer operation cuts")
    start, stop = values
    if not 0 <= start <= stop <= length:
        raise StrataError(f"{label} is outside the source or reversed")
    return start, stop


def _hash_string(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def _check_geometry(edits):
    """Adjacent replacements commute. An insertion touching another edit is ambiguous."""
    for left, right in zip(edits, edits[1:]):
        if (right.start < left.stop or
                (right.start == left.stop and
                 (left.start == left.stop or right.start == right.stop))):
            raise StrataError(f"overlapping or ambiguous parent cuts near operation {right.start}")


def _verify(parent, donor, records, *, create=False):
    if file_hash(parent.metadata) != file_hash(donor.metadata):
        raise StrataError("donor changes register declarations")
    if not isinstance(records, list) or not records:
        raise StrataError("each donor plan requires at least one edit")
    edits = []
    p = d = 0
    for record in records:
        required = {"parent", "donor"} if create else {"parent", "donor", "before_sha256"}
        if not isinstance(record, dict) or set(record) != required:
            raise StrataError(f"edit requires exactly {sorted(required)}")
        a, b = _range(record["parent"], len(parent.tape), "parent interval")
        x, y = _range(record["donor"], len(donor.tape), "donor interval")
        if a == b and x == y:
            raise StrataError("empty-to-empty edit")
        if a < p or x < d:
            raise StrataError("donor edits overlap or are out of order")
        if not same_span(parent.tape, p, a, donor.tape, d, x):
            raise StrataError("undeclared donor change before the requested interval")
        before = _fragment(parent.tape, a, b)
        if not create and (not _hash_string(record["before_sha256"]) or record["before_sha256"] != before):
            raise StrataError(f"original fragment SHA-256 mismatch at [{a}, {b})")
        after = _fragment(donor.tape, x, y)
        if before == after:
            raise StrataError("declared edit has identical original and replacement fragments")
        edits.append(Edit(a, b, x, y, before, after, donor))
        p, d = b, y
    if not same_span(parent.tape, p, len(parent.tape), donor.tape, d, len(donor.tape)):
        raise StrataError("undeclared donor change after the last interval")
    _check_geometry(edits)
    return edits


def _record(edit):
    return dict(parent=[edit.start, edit.stop], donor=[edit.donor_start, edit.donor_stop],
                before_sha256=edit.before_sha256)


def make_patch_plan(parent_path, donor_path, ranges):
    """Bind supplied (parent start, stop, donor start, stop) cuts to both artifacts."""
    records = []
    for cut in ranges:
        if not isinstance(cut, (list, tuple)) or len(cut) != 4:
            raise StrataError("each edit must supply four operation cuts")
        records.append(dict(parent=list(cut[:2]), donor=list(cut[2:])))
    with tempfile.TemporaryDirectory(prefix="strata-patch-plan-") as directory:
        root = Path(directory)
        parent = _source(parent_path, root / "parent")
        donor = _source(donor_path, root / "donor")
        edits = _verify(parent, donor, records, create=True)
        return dict(schema=SCHEMA, coordinates=COORDINATES, parent_sha256=parent.sha,
                    donor_sha256=donor.sha, donor_path=str(Path(donor_path).resolve()),
                    edits=[_record(edit) for edit in edits])


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise StrataError(f"duplicate JSON field {key}")
        result[key] = value
    return result


def _plan(path):
    data = Path(path).read_bytes()
    try:
        plan = json.loads(data, object_pairs_hook=_unique_fields)
    except (ValueError, UnicodeError) as error:
        raise StrataError(f"invalid JSON plan: {path}") from error
    fields = {"schema", "coordinates", "parent_sha256", "donor_sha256", "donor_path", "edits"}
    if not isinstance(plan, dict) or set(plan) != fields:
        raise StrataError("invalid patch plan fields")
    if plan["schema"] != SCHEMA or plan["coordinates"] != COORDINATES:
        raise StrataError("unsupported patch schema or coordinate system")
    if not _hash_string(plan["parent_sha256"]) or not _hash_string(plan["donor_sha256"]):
        raise StrataError("plans require full lower-case SHA-256 identities")
    if not isinstance(plan["donor_path"], str) or not plan["donor_path"]:
        raise StrataError("plan requires a donor path")
    return plan, sha256(data).hexdigest()


def _emit(stream, tape, start, stop):
    for line in tape.iter_span(start, stop):
        stream.write(line.encode("utf-8") + b"\n")


def compose_patches(parent_path, plan_paths, output_path, *, verify_cone=False, seed=1, clean_inputs=()):
    """Publish a new native artifact only after verifying every donor and its geometry."""
    output = Path(output_path)
    if output.suffix.lower() not in {".kmx", ".kmb"}:
        raise StrataError("output must be a .kmx or .kmb file")
    plans = [(Path(path), *_plan(path)) for path in plan_paths]
    if not plans:
        raise StrataError("at least one donor plan is required")
    pins = {plan["parent_sha256"] for _, plan, _ in plans}
    if len(pins) != 1:
        raise StrataError("donor plans pin different parents")
    with tempfile.TemporaryDirectory(prefix="strata-compose-") as directory:
        root = Path(directory)
        parent = _source(parent_path, root / "parent", expected=pins.pop())
        edits, sources = [], []
        for index, (path, plan, plan_hash) in enumerate(plans):
            donor = _source(path.parent / plan["donor_path"], root / str(index), expected=plan["donor_sha256"])
            edits.extend(_verify(parent, donor, plan["edits"]))
            sources.append(dict(donor_sha256=donor.sha, plan_sha256=plan_hash))
        edits.sort(key=lambda e: (e.start, e.stop))
        _check_geometry(edits)
        cone_receipts = None
        if verify_cone:
            from .session import Session
            cone_receipts = []
            with Session.open(parent_path) as sess:
                for edit in edits:
                    rep_kmx = "\n".join(edit.donor.tape.iter_span(edit.donor_start, edit.donor_stop)) + "\n"
                    chk = sess.verify_cone(
                        edit.start,
                        edit.stop,
                        rep_kmx,
                        seed=seed,
                        clean_wires=clean_inputs,
                    )
                    if not chk["passed"]:
                        raise StrataError(
                            f"causal-cone equivalence failed at [{edit.start}, {edit.stop}): {chk['reason']}"
                        )
                    cone_receipts.append(
                        dict(
                            parent=[edit.start, edit.stop],
                            donor=[edit.donor_start, edit.donor_stop],
                            passed=True,
                            cone_qubits=chk["cone_qubits"],
                            elapsed_ms=chk["elapsed_ms"],
                        )
                    )
        text = root / "result.kmx"
        p = 0
        with text.open("wb") as stream, parent.metadata.open("rb") as regs:
            shutil.copyfileobj(regs, stream)
            for edit in edits:
                _emit(stream, parent.tape, p, edit.start)
                _emit(stream, edit.donor.tape, edit.donor_start, edit.donor_stop)
                p = edit.stop
            _emit(stream, parent.tape, p, len(parent.tape))
        # Parse the combination as a complete circuit. This checks native syntax only.
        circuit = load_circuit(text)
        with tempfile.TemporaryDirectory(prefix=".strata-patch-", dir=output.parent) as staging:
            result = Path(staging) / ("result" + output.suffix.lower())
            circuit.to_file(result, format=output.suffix[1:].lower())
            result_sha = file_hash(result)
            count = len(parent.tape) + sum((e.donor_stop - e.donor_start) - (e.stop - e.start) for e in edits)
            if count != len(circuit):
                raise StrataError("merged native instruction count differs from composed spans")
            sem_status = "cone_sampled_256_verified" if verify_cone else "unverified"
            receipt = dict(schema="strata.patch-composition.v1", coordinates=COORDINATES,
                           parent_sha256=parent.sha, donors=sorted(sources, key=lambda s: s["plan_sha256"]),
                           result_sha256=result_sha, result_operations=count,
                           edits=[dict(**_record(e), donor_sha256=e.donor.sha, after_sha256=e.after_sha256)
                                  for e in edits], geometry="verified", semantic_equivalence=sem_status,
                           accepted=False)
            if cone_receipts is not None:
                receipt["cone_checks"] = cone_receipts
            # link publishes atomically and never overwrites another job or a pinned input.
            try:
                os.link(result, output)
            except FileExistsError as error:
                raise StrataError(f"output already exists: {output}") from error
        return receipt


def compose_kmb_fast(parent_kmb, edits, output_kmb, *, verify_cone=False, seed=1, clean_inputs=()):
    """Zero-copy binary ``.kmb`` v2 columnar splicer with optional 256-lane causal-cone verification."""
    output = Path(output_kmb)
    if output.exists():
        raise StrataError(f"output already exists: {output}")
    from .native import KmbTape
    from .session import Session

    tape = KmbTape.open(parent_kmb)
    cone_checks = []
    if verify_cone:
        with Session(tape) as sess:
            for start, stop, donor in edits:
                chk = sess.verify_cone(start, stop, donor, seed=seed, clean_wires=clean_inputs)
                if not chk["passed"]:
                    raise StrataError(f"causal-cone equivalence failed at [{start}, {stop}): {chk['reason']}")
                cone_checks.append(chk)
    res = tape.compose_kmb(edits, output)
    return dict(
        schema="strata.kmb-fast-compose.v1",
        output=str(output),
        verify_cone=bool(verify_cone),
        cone_checks=cone_checks,
        **res,
    )


def register_patch_composition(subparsers):
    plan = subparsers.add_parser("patch-plan", help="Pin the complete native diff at supplied operation cuts")
    plan.add_argument("parent")
    plan.add_argument("donor")
    plan.add_argument("--edit", nargs=4, type=int, action="append", required=True,
                      metavar=("P_START", "P_STOP", "D_START", "D_STOP"))
    plan.set_defaults(handler=_run_plan)
    compose = subparsers.add_parser("patch-compose", help="Compose pinned, disjoint common-parent native edits")
    compose.add_argument("parent")
    compose.add_argument("plans", nargs="+")
    compose.add_argument("--output", required=True)
    compose.add_argument("--verify-cone", action="store_true",
                         help="Verify each edit on 256 random basis states via checkpointed causal-cone replay")
    compose.add_argument("--seed", type=int, default=1)
    compose.set_defaults(handler=_run_compose)


def _run_plan(args):
    print(json.dumps(make_patch_plan(args.parent, args.donor, args.edit), indent=2))
    return 0


def _run_compose(args):
    print(json.dumps(compose_patches(args.parent, args.plans, args.output,
                                     verify_cone=args.verify_cone, seed=args.seed), indent=2))
    return 0
