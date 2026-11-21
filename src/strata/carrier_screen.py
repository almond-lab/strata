"""Width lower bounds for transporting live Boolean functions through named cuts."""

import json
from hashlib import file_digest
from pathlib import Path

import numpy as np

from .floors import _basis
from .model import StrataError
from .table_screen import _matrix, _packed


def screen_carrier_cut(live, helpers, *, row_count, width):
    """Price simultaneous affine recovery of the supplied live and helper functions.

    Input matrices have one function per row, packed with NumPy big bit order.
    Each witness index identifies a supplied assignment (column of the unpacked
    matrix). Restricting to those assignments reproduces both reported ranks.
    """
    if type(row_count) is not int or row_count < 1:
        raise StrataError("row_count must be a positive integer")
    if type(width) is not int or width < 0:
        raise StrataError("width must be a nonnegative integer")

    live = _matrix(live, row_count)
    helpers = _matrix(helpers, row_count)
    constant = (1 << row_count) - 1
    live_basis = _basis([constant, *(_packed(f, row_count) for f in live)])
    joint_basis = _basis([*live_basis, *(_packed(f, row_count) for f in helpers)])

    # The pivot columns of this row-reduced function matrix form a nonsingular
    # minor. _packed places assignment zero at bit row_count - 1.
    witness_rows = [row_count - pivot.bit_length() for pivot in joint_basis]
    live_width = len(live_basis) - 1
    joint_width = len(joint_basis) - 1
    return {
        "row_count": row_count,
        "width": width,
        "live_affine_width_lower_bound": live_width,
        "joint_affine_width_lower_bound": joint_width,
        "additional_independent_helpers": joint_width - live_width,
        "width_shortfall": max(0, joint_width - width),
        "status": "affine_width_exceeded" if joint_width > width else "within_rank_bound",
        "witness_rows": witness_rows,
    }


def _integer_vector(data, name, length):
    values = data[name]
    if values.shape != (length,) or values.dtype.kind not in "ui":
        raise StrataError(f"{name} must contain one integer per named cut")
    return values


def run_carrier_screen(path):
    """Read a packed NPZ, screen its cuts in order and pin the input bytes."""
    with Path(path).open("rb") as source:
        fingerprint = file_digest(source, "sha256").hexdigest()
        source.seek(0)
        with np.load(source, allow_pickle=False) as data:
            names = data["cut_names"]
            if names.ndim != 1 or names.dtype.kind != "U" or not len(names):
                raise StrataError("cut_names must be a nonempty vector of Unicode strings")
            names = names.tolist()
            if any(not name.strip() for name in names) or len(set(names)) != len(names):
                raise StrataError("cut names must be nonempty and unique")
            counts = _integer_vector(data, "row_counts", len(names))
            widths = _integer_vector(data, "widths", len(names))
            cuts = []
            for index, name in enumerate(names):
                result = screen_carrier_cut(
                    data[f"live_{index}"], data[f"helpers_{index}"],
                    row_count=int(counts[index]), width=int(widths[index]),
                )
                ids_key = f"row_ids_{index}"
                if ids_key in data:
                    ids = data[ids_key]
                    if ids.shape != (result["row_count"],) or ids.dtype.kind not in "uiU":
                        raise StrataError(f"{ids_key} must identify each row by integer or string")
                    result["witness_row_ids"] = ids[result["witness_rows"]].tolist()
                cuts.append({"cut": name, **result})

    return {
        "kind": "affine_carrier_screen",
        "input_sha256": fingerprint,
        "cuts": cuts,
        "exceeded_cuts": [cut["cut"] for cut in cuts if cut["width_shortfall"]],
        "scope": (
            "Each supplied function must be simultaneously recoverable as an affine "
            "function of the carried bits on the supplied rows. The constant is free. "
            "An exceeded width rules out this affine recovery at that cut. A rank "
            "within the budget does not prove a reversible, phase-correct, or "
            "in-budget circuit across cuts. Reachability and any additional side "
            "information are properties of the supplied model."
        ),
    }


def register_carrier_screen(subparsers):
    """Register the CLI and attach the JSON-producing handler to its namespace."""
    parser = subparsers.add_parser(
        "carrier-screen", help="Check affine carrier width at named cuts in a packed NPZ"
    )
    parser.add_argument("input", help="packed carrier-screen NPZ fixture")
    parser.set_defaults(handler=_run_command)


def _run_command(args):
    print(json.dumps(run_carrier_screen(args.input), indent=2))
    return 0

