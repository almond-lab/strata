"""Independent acceptance-boundary regressions for public names and fresh bits."""

import kickmix as km
import pytest

from strata.measurement_equivalence import equivalent_independent_outputs
from strata.model import StrataError
from strata.restructure import ConstructorRegistry, restructure_file


@pytest.mark.parametrize('replacement_metadata', [
    'APPEND_TO_REGISTER b1 r0\nAPPEND_TO_REGISTER q0 r0\nREGISTER r0 "result"',
    'APPEND_TO_REGISTER q0 r0\nAPPEND_TO_REGISTER b0 r0\nREGISTER r0 "result"',
    'APPEND_TO_REGISTER q0 r1\nREGISTER r1 "compiler_layout"',
])
def test_constructor_cannot_rebind_reorder_or_remove_named_public_outputs(tmp_path, replacement_metadata):
    source = tmp_path / 'source.kmx'
    source.write_text('APPEND_TO_REGISTER b0 r0\nAPPEND_TO_REGISTER q0 r0\n'
                      'REGISTER r0 "result"\nBIT_STORE1 b0\nX q0\nX q0')
    registry = ConstructorRegistry()
    registry.register('change_public_binding', lambda circuit, context:
                      dict(status='constructed', candidate=replacement_metadata + '\nBIT_STORE1 b0'),
                      'Preserve physical values but corrupt the named public output')
    result = restructure_file(source, tmp_path / 'out', registry=registry,
                              rounds=1, cost='instructions')
    assert result['status'] == 'no_gain'
    assert result['attempts'][0]['check'] == dict(
        equivalent=False, reason='public_named_register_binding_changed')
    assert (tmp_path / 'out' / 'best.kmx').read_text() == str(km.Circuit.from_file(source))


def test_preserving_public_binding_allows_extra_compiler_layout_register(tmp_path):
    metadata = ('APPEND_TO_REGISTER b0 r0\nAPPEND_TO_REGISTER q0 r0\n'
                'REGISTER r0 "result"')
    source = tmp_path / 'source.kmx'
    source.write_text(metadata + '\nBIT_STORE1 b0\nX q0\nX q0')
    registry = ConstructorRegistry()
    registry.register('keep_public_binding', lambda circuit, context:
                      dict(status='constructed', candidate=metadata + '\n'
                           'APPEND_TO_REGISTER q0 r1\nREGISTER r1 "compiler_layout"\nBIT_STORE1 b0'),
                      'Preserve the original output and add compiler layout metadata')
    result = restructure_file(source, tmp_path / 'out', registry=registry,
                              rounds=1, cost='instructions')
    assert result['status'] == 'verified_improvement'
    retained = km.Circuit.from_file(result['best']['path'])
    assert list(retained.register_data['result']) == [km.b(0), km.q(0)]
    assert 'compiler_layout' in retained.register_data


def test_default_constructors_accept_native_unnamed_register_padding(tmp_path):
    source = tmp_path / 'source.kmx'
    source.write_text('APPEND_TO_REGISTER q0 r1\nREGISTER r1 "public"\n'
                      'CCX q0 q1 q4\nCX q4 q2\nCCX q0 q1 q4')
    # The native serializer supplies an unnamed `REGISTER r0` for the gap.
    assert str(km.Circuit.from_file(source)).startswith('REGISTER r0\n')
    result = restructure_file(source, tmp_path / 'out', clean_inputs=[4], rounds=1)
    assert result['status'] == 'verified_improvement'
    retained = km.Circuit.from_file(result['best']['path'])
    assert list(retained.register_data['public']) == [km.q(0)]


@pytest.mark.parametrize('new_bit', [99, 100])
def test_episode_reserves_bits_mentioned_only_in_parent_public_metadata(tmp_path, new_bit):
    source = tmp_path / 'source.kmx'
    source.write_text('APPEND_TO_REGISTER b99 r0\nREGISTER r0 "classical_result"\n'
                      'X q0\nX q1\nX q1\nZ q0')
    registry = ConstructorRegistry()
    observed_offsets = []

    def measured_identity(circuit, context):
        observed_offsets.append(context['classical_bit_offset'])
        return dict(status='constructed', candidate=f'HMR q1 b{new_bit}',
                    introduced_private_bits=[new_bit])

    registry.register('measured_identity', measured_identity, 'Erase a declared-zero temporary')
    result = restructure_file(source, tmp_path / 'out', registry=registry,
                              scan_range=(1, 3), clean_inputs=[1], rounds=1,
                              cost='instructions', max_operations=16)
    assert observed_offsets == [100]
    if new_bit == 99:
        assert result['status'] == 'no_gain'
        attempt = result['construction']['attempts'][0]
        assert attempt['status'] == 'unknown'
        assert 'collide with an existing parent bit' in attempt['reason']
    else:
        assert result['status'] == 'verified_improvement'
        retained = km.Circuit.from_file(result['best']['path'])
        assert list(retained.register_data['classical_result']) == [km.b(99)]
        assert str(retained).endswith('X q0\nHMR q1 b100\nZ q0')
        assert equivalent_independent_outputs(str(km.Circuit.from_file(source)), str(retained),
                                               clean_inputs=[1], internal_bits=[100])['equivalent']


def test_episode_cannot_hide_a_classical_bit_consumed_by_untouched_suffix(tmp_path):
    source = tmp_path / 'source.kmx'
    source.write_text('HMR q1 b0\nX q0 if b0')
    with pytest.raises(StrataError, match='referenced outside the selected boundary'):
        restructure_file(source, tmp_path / 'out', scan_range=(0, 1),
                         clean_inputs=[1], internal_bits=[0], rounds=1)
