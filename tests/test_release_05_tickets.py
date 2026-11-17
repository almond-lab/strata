import json
import sys

import kickmix as km
import pytest

from strata.cli import main
from strata.experiments import create_experiment, load_manifest, run_experiment
from strata.inference import is_function_of
from strata.store import Store
from strata.tensor import tensor_bounds, tensor_from_anf
from test_v04_experiments import local_transport


def test_wrap_source_tree_imports_and_thread_derived_stage_limits(tmp_path, local_transport, capsys):
    tree = tmp_path / 'src'
    tree.mkdir()
    (tree / 'helper.py').write_text('value=17')
    worker = tree / 'worker.py'
    worker.write_text("from helper import value\nfrom pathlib import Path\nPath('result.json').write_text(str(value))")
    output = tmp_path / 'run.json'
    assert main([
        'wrap', '--out', str(output), '--threads', '2', '--timeout', '11',
        '--dependency', str(tree), '--output', 'result=result.json:json', '--', sys.executable, str(worker),
    ]) == 0
    assert json.loads(capsys.readouterr().out)['dependency_notice'] is None
    assert load_manifest(output)['stages'][0]['limits']['cpu_seconds'] == 22

    store = Store(tmp_path / 'store')
    try:
        result = run_experiment(store, create_experiment(store, output))
        assert result['status'] == 'artifact_validated', result.get('error')
        assert json.loads(store.read_blob(result['artifacts']['result']['sha256'])) == 17
    finally:
        store.db.close()
    manifest = json.loads(output.read_text())
    manifest['limits']['cpu_seconds'] = 7
    output.write_text(json.dumps(manifest))
    assert load_manifest(output)['stages'][0]['limits']['cpu_seconds'] == 7
    assert main(['wrap', '--out', str(tmp_path / 'without.json'), '--', sys.executable, str(worker)]) == 0
    assert '--dependency' in json.loads(capsys.readouterr().out)['dependency_notice']


def test_form_count_and_hmr_debt_reachable_scope(tmp_path, capsys):
    circuit = km.Circuit('X q38')
    query = is_function_of(circuit, 38, 1, range(38), degree=3, max_forms=4096)
    assert query['status'] == 'unknown' and query['forms_needed'] == 9178

    source = tmp_path / 'debt.kmx'
    source.write_text('CCX q0 q1 q2\nHMR q2 b0\nCZ q0 q1 if b0')
    args = ['hmr-debt', str(source), '--measurement', '1', '--clean-inputs', '2']
    assert main([*args, '--cuts', '2,3']) == 0
    report = json.loads(capsys.readouterr().out)
    assert all(row['minimum_degree_in_support'] == 2 for row in report['results'])

    promise = tmp_path / 'rows.json'
    promise.write_text(json.dumps({'assignments': [[0, 0, 0], [1, 1, 0]]}))
    assert main([*args, '--cuts', '2', '--domain', str(promise)]) == 0
    restricted = json.loads(capsys.readouterr().out)
    assert restricted['domain_hash'] and restricted['results'][0]['minimum_degree_in_support'] == 1


def test_retimed_hmr_independent_cli_and_compact_register_range(tmp_path, capsys):
    source = tmp_path / 'src.kmx'
    candidate = tmp_path / 'cand.kmx'
    source.write_text('CX q0 q2\nCX q0 q1\nHMR q2 b0\nZ q0 if b0')
    candidate.write_text('CX q0 q2\nHMR q2 b1\nZ q0 if b1\nCX q0 q1')
    args = [
        'evaluate-file', str(source), str(candidate), '--out', str(tmp_path / 'out'),
        '--clean-inputs', '2', '--measurement-mode', 'independent-map', '--internal-bits', 'b0,b1',
        '--max-window', '1', '--max-proof-operations', '64',
    ]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)['checks']['equivalent'] is True

    promise = tmp_path / 'promise.json'
    promise.write_text(json.dumps(dict(
        version=1, type='address_promise', address_wires=[0], arbitrary_wires=[1],
        clean_wires=[2], interval=[0, 1],
    )))
    assert main([*args, '--domain', str(promise)]) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'verified_domain_no_gain'
    candidate.write_text(candidate.read_text().replace('Z q0 if b1', 'Z q1 if b1'))
    assert main([*args, '--domain', str(promise)]) == 0
    assert json.loads(capsys.readouterr().out)['checks']['equivalent'] is not True


@pytest.mark.parametrize(
    'bits, modulus, old_bound, new_bound',
    [(4, 0x13, 11, 15), (12, 0x1053, 35, 54)],
)
def test_griesmer_bound_paired_field_products(bits, modulus, old_bound, new_bound):
    # The same field element x multiplies two independent words y and z.
    outputs = [[] for _ in range(2 * bits)]
    for group in range(2):
        for i in range(bits):
            for j in range(bits):
                word = 1 << (i + j)
                while word.bit_length() > bits:
                    word ^= modulus << (word.bit_length() - bits - 1)
                for k in range(bits):
                    if (word >> k) & 1:
                        outputs[group * bits + k].append([i, (group + 1) * bits + j])
    tensor = tensor_from_anf(outputs, range(bits), range(bits, 3 * bits))
    result = tensor_bounds(tensor, max_substitution_combinations=(1 << bits) - 1)
    assert result['substitution_bound']['value'] == old_bound
    assert sum(result['griesmer_bound']['per_mode']['left']['terms']) == new_bound
    assert result['value'] == new_bound
    assert tensor_bounds(tensor, max_substitution_combinations=1)['griesmer_bound']['value'] is None
