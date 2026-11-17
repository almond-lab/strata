import json

import kickmix as km
import pytest

from strata.cli import _wire_list, main
from strata.floors import circuit_floors
from strata.model import StrataError


@pytest.mark.parametrize('text', ['16..27', '16-27', ' 16 .. 27 '])
def test_inclusive_ranges(text):
    assert _wire_list(text) == tuple(range(16, 28))


def test_mixed_range_list_and_empty():
    assert _wire_list('0,4..7,12,20-21') == (0, 4, 5, 6, 7, 12, 20, 21)
    assert _wire_list('') == ()


@pytest.mark.parametrize('text', ['q1', '1:4', '-1', '1,,2', '1...', '1..x'])
def test_invalid_lists_name_accepted_syntax(text):
    with pytest.raises(StrataError, match='inclusive ranges'):
        _wire_list(text)


def test_ranges_are_bounded_and_nondescending():
    with pytest.raises(StrataError, match='range end'):
        _wire_list('9..2')
    with pytest.raises(StrataError, match='expansion'):
        _wire_list('0..1000000000')


def test_floors_cli_range_and_usage_error(tmp_path, capsys):
    path = tmp_path / 'test.kmx'
    path.write_text('CX q0 q2\nCX q0 q2')
    assert main(['floors', str(path), '--at', '2', '--clean-inputs', '1..2']) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'exact'
    assert main(['floors', str(path), '--at', '2', '--clean-inputs', 'bad']) == 2
    assert 'inclusive ranges' in capsys.readouterr().out


def test_measurement_semantics_do_not_establish_boolean_gate_floor():
    measured = km.Circuit('CX q0 q1\nHMR q1 b0\nZ q0 if b0')
    result = circuit_floors(measured, clean_inputs=[1])
    assert result['status'] == 'unknown'
    assert 'measurement or classical channels' in result['reason']
