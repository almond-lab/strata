import json
from hashlib import sha256
from itertools import product

import kickmix as km
import numpy as np
import pytest

from strata.cli import main
from strata.hmr_debt import hmr_debt
from strata.model import StrataError
from strata.table_screen import affine_repair_floor, fit_care_table, screen_care_table


def test_hmr_debt_diagnostic_belongs_to_final_degree():
    circ = km.Circuit('CCX q0 q1 q2\nCZ q2 q3 if b0\nHMR q2 b1')
    def query(degree):
        return hmr_debt(circ, 2, [3], support=[0, 1], clean_inputs=[2],
                        degree=degree)['results'][0]
    match = query(2)
    assert match['status'] == 'yes' and match['minimum_degree_in_support'] == 2
    assert 'reason' not in match
    assert query(1)['status'] == 'no' and 'outside' in query(1)['reason']


def test_care_screen_matches_scalar_oracle_and_reports_first_conflict():
    rng = np.random.default_rng(123)
    values = rng.integers(0, 3, size=57)
    c = rng.integers(0, 2, size=(80, 57), dtype=np.uint8)
    c[0] = (values == 1)
    z = np.packbits(values == 0)
    one = np.packbits(values == 1)
    data = np.packbits(c, axis=1)
    data[:, -1] |= 127  # Padding is outside the domain.
    result = screen_care_table(one, z, data, row_count=57)
    for i, r in enumerate(result['results']):
        conflicts = np.flatnonzero((values != 2) & (values != c[i]))
        assert r['matches'] == (len(conflicts) == 0)
        if len(conflicts):
            assert r['counterexample_row'] == conflicts[0]
            assert r['expected'] == values[conflicts[0]]
    with pytest.raises(StrataError):
        screen_care_table(one, one, data, row_count=57)
    with pytest.raises(StrataError):
        screen_care_table(one, z, data[:, :-1], row_count=57)


def test_repair_floor_accounts_for_free_constant_and_dependent_targets():
    rows = np.array(list(product(range(2), repeat=3)), dtype=np.uint8).T
    a, b, c = rows
    available = np.packbits([a, b, c, a & b], axis=1)
    goal = np.packbits([a ^ b ^ 1, a & b, b & c, (b & c) ^ a], axis=1)
    out = affine_repair_floor(available, goal, row_count=8)
    assert out['available_rank_including_constant'] == 5
    assert out['additional_and_lower_bound'] == 1
    assert affine_repair_floor(available, available, row_count=8)['additional_and_lower_bound'] == 0


def test_cli_pins_fixture_and_allows_blind_rows(tmp_path, capsys):
    path = tmp_path / 'care.npz'
    np.savez(path, row_count=3, required_one=np.uint8([0x80]), required_zero=np.uint8([0x20]),
             candidates=np.uint8([[0xC0], [0x40]]))
    assert main(['table-screen', 'care', str(path)]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt['input_sha256'] == sha256(path.read_bytes()).hexdigest()
    assert [r['matches'] for r in receipt['results']] == [True, False]
    assert receipt['results'][1]['counterexample_row'] == 0



def test_affine_fit_and_dual_witness_match_exhaustive_truth_tables():
    # Every truth function and every care domain on a 2-bit table.
    rows = np.array(list(product(range(2), repeat=2)), dtype=np.uint8).T
    cols = np.packbits(rows, axis=1)
    family = [((np.uint8(c) ^ (rows[0] * a) ^ (rows[1] * b)))
              for c, a, b in product(range(2), repeat=3)]
    for pattern in product(range(3), repeat=4):
        y = np.array(pattern)
        fit = fit_care_table(np.packbits(y == 1), np.packbits(y == 0), cols, row_count=4)
        assert fit['fits'] == any(np.all(f[y != 2] == y[y != 2]) for f in family)
        if fit['fits']:
            v = np.full(4, fit['constant'], dtype=np.uint8)
            for i in fit['feature_indices']:
                v ^= rows[i]
            assert np.all(v[y != 2] == y[y != 2])
        else:
            idx = fit['contradictory_rows']
            assert len(idx) % 2 == 0 and np.all(y[idx] != 2)
            assert not np.any(np.bitwise_xor.reduce(rows[:, idx], axis=1))
            assert np.bitwise_xor.reduce(y[idx]) == 1


def test_repeated_native_line_scans_release_descriptors():
    from pathlib import Path
    from strata.streaming import native_lines
    descriptors = Path('/proc/self/fd')
    if not descriptors.is_dir():
        pytest.skip('Linux descriptor accounting')
    c = km.Circuit('CX q0 q1')
    count = len(list(descriptors.iterdir()))
    for _ in range(32):
        with native_lines(c) as lines:
            assert list(lines) == ['CX q0 q1\n']
    assert len(list(descriptors.iterdir())) <= count + 1


def test_repeated_binary_and_text_bundle_reads_release_descriptors(tmp_path):
    from pathlib import Path
    from strata.bundle import BundleFrontend
    fds = Path('/proc/self/fd')
    if not fds.is_dir():
        pytest.skip('Linux descriptor accounting')
    circuit = km.Circuit('CX q0 q1')
    path = tmp_path / 'pair.kmb'
    circuit.to_file(path)
    payloads = [path.read_bytes(), b'CX q0 q1\n']
    count = len(list(fds.iterdir()))
    for _ in range(32):
        for payload in payloads:
            assert str(BundleFrontend.circuit(payload)).strip() == 'CX q0 q1'
    assert len(list(fds.iterdir())) <= count + 1
