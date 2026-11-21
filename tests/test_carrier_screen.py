import argparse
import json
from hashlib import sha256

import numpy as np
import pytest

from strata.carrier_screen import register_carrier_screen, run_carrier_screen, screen_carrier_cut
from strata.model import StrataError


def _rank_from_assignments(columns):
    """Independent row-oriented GF(2) oracle with an explicit constant coordinate."""
    pivots = {}
    for values in columns.T:
        row = 1 | sum(int(value) << (i + 1) for i, value in enumerate(values))
        while row:
            key = row.bit_length()
            if key not in pivots:
                pivots[key] = row
                break
            row ^= pivots[key]
    return len(pivots) - 1


def _pack(columns):
    return np.packbits(columns, axis=1, bitorder="big")


def _two_cut_fixture():
    rows = np.arange(8)
    x, y, z = ((rows >> i) & 1 for i in range(3))
    live = _pack([x, y, z])
    return dict(cut_names=["before", "after"], row_counts=[8, 8], widths=[3, 3],
                live_0=live, helpers_0=_pack([x ^ y ^ 1]),
                live_1=live, helpers_1=_pack([x & y, x & z]),
                row_ids_1=[f"x={r}" for r in rows])


def test_named_cuts_identify_independent_nonlinear_helpers(tmp_path, capsys):
    fixture = tmp_path / "cuts.npz"
    data = _two_cut_fixture()
    np.savez(fixture, **data)
    parser = argparse.ArgumentParser()
    register_carrier_screen(parser.add_subparsers())
    args = parser.parse_args(["carrier-screen", str(fixture)])
    assert args.handler(args) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["input_sha256"] == sha256(fixture.read_bytes()).hexdigest()
    assert receipt["exceeded_cuts"] == ["after"]
    first, second = receipt["cuts"]
    assert (first["live_affine_width_lower_bound"], first["joint_affine_width_lower_bound"]) == (3, 3)
    assert first["status"] == "within_rank_bound"
    assert second["status"] == "affine_width_exceeded"
    assert second["additional_independent_helpers"] == second["width_shortfall"] == 2
    assert second["witness_row_ids"] == [f"x={r}" for r in second["witness_rows"]]
    assert len(second["witness_rows"]) == 6


@pytest.mark.parametrize("n", [1, 2, 9, 29, 80])
def test_row_witness_replays_both_ranks_and_ignores_padding(n):
    rng = np.random.default_rng(n)
    values = rng.integers(0, 2, (11, n), dtype=np.uint8)
    values[9] = values[2] ^ values[5] ^ 1
    values[10] = values[8]
    packed = _pack(values)
    if n % 8:
        packed[:, -1] |= (1 << (8 - n % 8)) - 1
    result = screen_carrier_cut(packed[:7], packed[7:], row_count=n, width=8)
    for signals, key in [(values[:7], "live_affine_width_lower_bound"),
                         (values, "joint_affine_width_lower_bound")]:
        assert result[key] == _rank_from_assignments(signals)
        assert result[key] == _rank_from_assignments(signals[:, result["witness_rows"]])
    assert len(result["witness_rows"]) == result["joint_affine_width_lower_bound"] + 1


def test_no_helpers_and_existing_width_shortfall():
    live = _pack([[0, 1, 0, 1], [0, 0, 1, 1]])
    no_signals = np.empty((0, 1), dtype=np.uint8)
    r = screen_carrier_cut(live, no_signals, row_count=4, width=1)
    assert (r["additional_independent_helpers"], r["width_shortfall"]) == (0, 1)
    r = screen_carrier_cut(no_signals, no_signals, row_count=4, width=0)
    assert r["joint_affine_width_lower_bound"] == r["width_shortfall"] == 0
    assert len(r["witness_rows"]) == 1


@pytest.mark.parametrize("key, value", [
    ("cut_names", ["before", "before"]),
    ("cut_names", [" ", "after"]),
    ("row_counts", [0, 8]),
    ("widths", [3, -1]),
    ("widths", [3.5, 4.5]),
    ("row_ids_1", [1]),
    ("helpers_1", np.empty((0, 2), dtype=np.uint8)),
])
def test_malformed_cut_fixture(tmp_path, key, value):
    data = _two_cut_fixture()
    data[key] = value
    path = tmp_path / "invalid.npz"
    np.savez(path, **data)
    with pytest.raises(StrataError):
        run_carrier_screen(path)

