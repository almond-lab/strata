import json

import kickmix as km
import pytest

from strata.cli import main
from strata.discovery import evaluate_candidate
from strata.experiments import load_manifest
from strata.model import StrataError
from strata.restructure import restructure_file
from strata.semantic import analyze_complete


def compact(address, arbitrary, clean):
    return dict(version=1, type='address_promise', address_wires=address,
                arbitrary_wires=arbitrary, clean_wires=clean, interval=[0, (1 << len(address))-1])


def test_explicit_cut_promise_is_bound_and_priced_in_unchanged_parent(tmp_path):
    source = tmp_path / 'source.kmx'
    source.write_text('X q3\nCCX q0 q1 q2\nX q3')
    promise = compact([0,1], [2,3], [])
    promise.update(addresses=[0,3])
    del promise['interval']
    with pytest.raises(StrataError, match='domain-at-cut'):
        restructure_file(source, tmp_path / 'ambiguous', scan_range=(1,2), domain=promise)
    report = restructure_file(source, tmp_path / 'out', scan_range=(1,2), domain=promise,
                              domain_at_cut=True, cost='toffoli')
    assert report['status'] == 'verified_domain_improvement'
    assert report['cost_delta'] == -1
    assert report['episode']['host_reachability_verified'] is False
    assert report['check']['domain_hash'] == report['episode']['promise']['domain_hash']
    assert str(km.Circuit.from_file(report['best']['path'])).splitlines() == [
        'X q3', 'CX q0 q2', 'X q3',
    ]


def test_promise_elides_proved_resets_and_rejects_dirty_reset(tmp_path):
    source = tmp_path / 'source.kmx'
    prep = 'CCX q0 q1 q3\nCX q3 q2\nCCX q0 q1 q3'
    source.write_text('R q3\n' + prep + '\nR q3')
    promise = compact([0,1], [2], [3])
    options = dict(clean_inputs=(3,), domain=promise, cost='toffoli')
    report = restructure_file(source, tmp_path / 'out', **options)
    assert report['cost_delta'] == -1
    assert report['attempts'][0]['construction']['reset_elision_proof']['equivalent']
    source.write_text('X q3\nR q3\n' + prep)
    rejected = restructure_file(source, tmp_path / 'dirty', **options)
    assert rejected['status'] == 'no_gain'
    assert rejected['attempts'][0]['construction']['reason'] == 'reset_not_proven_zero'


def test_promised_search_recovers_measured_product_with_delayed_affine_repair(tmp_path):
    source = tmp_path / 'source.kmx'
    source.write_text('\n'.join([
        'CX q0 q5', 'CX q1 q5', 'CCX q5 q2 q4', 'CX q1 q5', 'CX q0 q5',
        'CX q4 q3', 'HMR q4 b0', 'CX q0 q5', 'CX q1 q5', 'CZ q5 q2 if b0', 'CX q1 q5', 'CX q0 q5',
        'CCX q0 q2 q4', 'CX q4 q3', 'CCX q0 q2 q4',
    ]))
    promise = compact([0,1,2], [3], [4,5])
    report = restructure_file(source, tmp_path / 'out', clean_inputs=(4,5), internal_bits=(0,),
                              domain=promise, cost='toffoli', fast=True, deep_only=True)
    assert report['cost_delta'] == -2
    first = report['attempts'][0]
    assert first['construction']['measurement_preparation']['status'] == 'verified'
    assert first['check']['equivalent'] and first['check']['measurement_independent']


def test_factor_limits_and_stdout_summary_reach_both_constructors(tmp_path, capsys):
    source = tmp_path / 'source.kmx'
    source.write_text('CCX q0 q1 q4\nCCX q4 q2 q3\nCCX q0 q1 q4')
    out = tmp_path / 'out'
    assert main(['restructure', str(source), '--out', str(out), '--clean-inputs', '4',
                 '--factor-timeout', '0.5', '--max-factor-nodes', '512', '--max-factor-steps', '20000',
                 '--max-orders', '6', '--max-partitions', '17', '--rounds', '1', '--summary']) == 0
    assert len(capsys.readouterr().out) < 2048
    receipt = json.loads((out / 'restructure.json').read_text())
    trials = {row['mechanism']: row['construction'] for row in receipt['attempts']}
    assert trials['algebraic_boundary_synthesis']['configured_factor_limits'] == {
        'timeout': .5, 'max_factor_nodes': 512, 'max_factor_steps': 20000, 'max_orders': 6,
    }
    assert trials['consumer_functional_decomposition']['configured_factor_limits'] == {
        'timeout': .5, 'max_factor_nodes': 512, 'max_steps': 20000, 'max_partitions': 17,
    }


def test_wrap_configures_output_cap(tmp_path, capsys):
    import sys

    path = tmp_path / 'wrap.json'
    assert main(['wrap', '--out', str(path), '--max-output', '33554432', '--', sys.executable, '--version']) == 0
    capsys.readouterr()
    assert load_manifest(path)['limits']['max_output'] == 33554432


def test_capped_anf_stage_reports_requested_and_effective_limits():
    report = analyze_complete('CCX q0 q1 q2', max_terms=2**20, max_product_terms=2**21)
    assert report['anf_stage_caps'] == {
        'max_terms': {'requested': 2**20, 'effective': 2**16},
        'max_product_terms': {'requested': 2**21, 'effective': 2**16},
    }


def test_evaluate_measured_uncompute_with_narrower_clean_layout(tmp_path):
    src = tmp_path / 'src.kmx'
    dst = tmp_path / 'dst.kmx'
    src.write_text('CCX q0 q1 q3\nCX q3 q2\nHMR q3 b0\nCZ q0 q1 if b0\nR q4')
    dst.write_text('CCX q0 q1 q2')
    result = evaluate_candidate(src, dst, tmp_path / 'out', clean_inputs=(3,4), cost='toffoli',
                                measurement_mode='independent-map', internal_bits=(0,))
    assert result['status'] == 'verified_no_gain'
    assert result['checks']['omitted_trailing_clean_wires'] == [3,4]
    assert result['checks']['outcome_independence_proved']
    public = evaluate_candidate(src, dst, tmp_path / 'public', clean_inputs=(3,4))
    assert public['status'] == 'unknown'
    assert 'measurement_obligation' in public['checks']


def test_regenerated_nonterminal_cache_can_be_erased_by_measurement():
    from strata.lookup_structure import measured_cache_cleanup
    from strata.promised_domain import check_promised_candidate

    pure = km.Circuit('CCX q0 q1 q4\nCCX q4 q2 q3\nCCX q0 q1 q4')
    measured = measured_cache_cleanup(pure, (4,), 0)
    assert 'HMR q4 b0\nCZ q0 q1 if b0' in str(measured)
    checked = check_promised_candidate(pure, measured, compact([0,1,2], [3], [4]),
                                       clean_inputs=(4,), internal_bits=(0,), measurement_mode='independent-map')
    assert checked['equivalent'] and checked['clean_outputs_restored']
