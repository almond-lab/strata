import argparse
from hashlib import sha256
import json

import kickmix as km
import pytest

from strata.lab import exhaustive
from strata.model import Contract, Register, StrataError
from strata.patch_composition import compose_patches, make_patch_plan, register_patch_composition


def native(path, gates):
    km.Circuit('\n'.join(gates)).to_file(path, format=path.suffix[1:])
    return path


def plan(path, parent, donor, *cuts):
    path.write_text(json.dumps(make_patch_plan(parent, donor, cuts)))
    return path


def test_ordered_binary_and_text_composition(tmp_path):
    metadata = ['APPEND_TO_REGISTER q3 r0', 'REGISTER r0 "io"']
    gates = ['X q0', 'X q1', 'X q2', 'X q3']
    parent = native(tmp_path / 'parent.kmb', metadata + gates)
    left = native(tmp_path / 'left.kmx', metadata + ['Z q0', 'Z q0'] + gates[1:])
    right = native(tmp_path / 'right.kmb', metadata + gates[:2] + ['NEG'] + gates[3:])
    a = plan(tmp_path / 'a.json', parent, left, (0, 1, 0, 2))
    b = plan(tmp_path / 'b.json', parent, right, (2, 3, 2, 3))
    out = tmp_path / 'result.kmb'
    report = compose_patches(parent, [b, a], out)
    assert report == compose_patches(parent, [a, b], tmp_path / 'result2.kmb')
    expected = km.Circuit('\n'.join(metadata + ['Z q0', 'Z q0', 'X q1', 'NEG', 'X q3']))
    assert str(km.Circuit.from_file(out)) == str(expected)
    assert report['result_operations'] == 5
    assert report['result_sha256'] == sha256(out.read_bytes()).hexdigest()
    assert report['parent_sha256'] == sha256(parent.read_bytes()).hexdigest()
    assert report['edits'][0]['before_sha256'] == sha256(b'X q0\n').hexdigest()
    assert report['edits'][0]['after_sha256'] == sha256(b'Z q0\nZ q0\n').hexdigest()
    assert [e['parent'] for e in report['edits']] == [[0, 1], [2, 3]]
    assert report['geometry'] == 'verified' and report['semantic_equivalence'] == 'unverified'
    assert report['accepted'] is False
    with pytest.raises(StrataError, match='already exists'):
        compose_patches(parent, [a], out)
    assert report['result_sha256'] == sha256(out.read_bytes()).hexdigest()


def test_phase_interaction_despite_disjoint_donors(tmp_path):
    cx = 'CX q0 q1'
    gates = [cx, cx, 'Z q2', 'Z q2', cx, cx]
    parent = native(tmp_path / 'p.kmx', gates)
    left = native(tmp_path / 'a.kmx', [cx, 'Z q2', 'Z q2', cx])
    right = native(tmp_path / 'b.kmx', gates[:2] + ['Z q1'] + gates[2:])
    a = plan(tmp_path / 'a.json', parent, left, (1, 2, 1, 1), (5, 6, 4, 4))
    b = plan(tmp_path / 'b.json', parent, right, (2, 3, 2, 4))
    out = tmp_path / 'merged.kmx'
    report = compose_patches(parent, [a, b], out)
    assert report['geometry'] == 'verified' and report['accepted'] is False
    contract = Contract('identity', (Register('x', (0,)), Register('scratch', (1, 2), 'clean')), 'identity')
    def check(path):
        return exhaustive(km.Circuit.from_file(path), contract, lambda inputs, _: ({'x': inputs['x']}, 0))
    for item in (parent, left, right):
        result = check(item)
        assert result.passed and result.exhaustive
    failure = check(out)
    assert not failure.passed and 'phase' in str(failure.counterexample).lower()


def test_exact_parent_and_donor_identity(tmp_path):
    p = native(tmp_path / 'p.kmx', ['X q0', 'X q1'])
    q = native(tmp_path / 'other.kmb', ['X q0', 'X q1'])
    donor = native(tmp_path / 'd.kmx', ['Z q0', 'X q1'])
    a = plan(tmp_path / 'a.json', p, donor, (0, 1, 0, 1))
    b = plan(tmp_path / 'b.json', q, donor, (0, 1, 0, 1))
    out = tmp_path / 'out.kmb'
    with pytest.raises(StrataError, match='different parents'):
        compose_patches(p, [a, b], out)
    with pytest.raises(StrataError, match='source SHA-256 mismatch'):
        compose_patches(q, [a], out)
    donor.write_text('Z q1\nX q1\n')
    with pytest.raises(StrataError, match='source SHA-256 mismatch'):
        compose_patches(p, [a], out)
    assert not out.exists()


def test_omitted_diff_and_metadata(tmp_path):
    p = native(tmp_path / 'p.kmx', ['X q0', 'X q1', 'X q2'])
    d = native(tmp_path / 'd.kmx', ['Z q0', 'X q1', 'Z q2'])
    for cut in [(0, 1, 0, 1), (2, 3, 2, 3)]:
        with pytest.raises(StrataError, match='undeclared donor change'):
            make_patch_plan(p, d, [cut])
    d = native(tmp_path / 'd.kmx', ['APPEND_TO_REGISTER q2 r0', 'X q0', 'NEG', 'X q2'])
    with pytest.raises(StrataError, match='register declarations'):
        make_patch_plan(p, d, [(1, 2, 1, 2)])


@pytest.mark.parametrize(('child', 'othercut'), [
    (['X q0', 'Z q1', 'X q2'], (1, 2, 1, 2)),
    (['X q0', 'NEG', 'X q1', 'X q2'], (1, 1, 1, 2)),
    (['X q0', 'X q1', 'NEG', 'X q2'], (2, 2, 2, 3)),
])
def test_colliding_edits(tmp_path, child, othercut):
    p = native(tmp_path / 'p.kmx', ['X q0', 'X q1', 'X q2'])
    d = native(tmp_path / 'd.kmx', ['X q0', 'Z q0', 'X q2'])
    e = native(tmp_path / 'e.kmx', child)
    a = plan(tmp_path / 'a.json', p, d, (1, 2, 1, 2))
    b = plan(tmp_path / 'b.json', p, e, othercut)
    out = tmp_path / 'o.kmb'
    with pytest.raises(StrataError, match='ambiguous'):
        compose_patches(p, [b, a], out)
    assert not out.exists()


def test_insertion_and_prior_hash(tmp_path):
    p = native(tmp_path / 'p.kmx', [])
    d = native(tmp_path / 'd.kmx', ['X q0'])
    a = plan(tmp_path / 'a.json', p, d, (0, 0, 0, 1))
    out = tmp_path / 'o.kmb'
    with pytest.raises(StrataError, match='ambiguous'):
        compose_patches(p, [a, a], out)
    data = json.loads(a.read_text())
    assert data['edits'][0]['before_sha256'] == sha256(b'').hexdigest()
    assert compose_patches(p, [a], out)['result_operations'] == 1
    data['edits'][0]['before_sha256'] = '0' * 64
    a.write_text(json.dumps(data))
    with pytest.raises(StrataError, match='original fragment'):
        compose_patches(p, [a], tmp_path / 'invalid.kmb')


@pytest.mark.parametrize('cut', [(1, 0, 0, 1), (0, 2, 0, 1), (False, 1, 0, 1), (0, 1, 0, 3)])
def test_invalid_cuts(tmp_path, cut):
    p = native(tmp_path / 'p.kmx', ['X q0'])
    d = native(tmp_path / 'd.kmx', ['Z q0'])
    with pytest.raises(StrataError):
        make_patch_plan(p, d, [cut])


def test_commands_and_plan_relative_donor(tmp_path, capsys):
    parser = argparse.ArgumentParser()
    register_patch_composition(parser.add_subparsers(required=True))
    p = native(tmp_path / 'p.kmx', ['X q0'])
    d = native(tmp_path / 'd.kmx', ['Z q0'])
    a = parser.parse_args(['patch-plan', str(p), str(d), '--edit', '0', '1', '0', '1'])
    assert a.handler(a) == 0
    data = json.loads(capsys.readouterr().out)
    data['donor_path'] = d.name
    file = tmp_path / 'plan.json'
    file.write_text(json.dumps(data))
    out = tmp_path / 'out.kmx'
    a = parser.parse_args(['patch-compose', str(p), str(file), '--output', str(out)])
    assert a.handler(a) == 0
    assert json.loads(capsys.readouterr().out)['result_operations'] == 1
    assert str(km.Circuit.from_file(out)) == 'Z q0'
