import hashlib
import json
from pathlib import Path

import kickmix as km

from strata import episode_search
from strata.accounting import measured_resources
from strata.restructure import ConstructorRegistry, _boundary_metadata, restructure_file
from strata.semantic import equivalent_windows


def _options(**overrides):
    registry = ConstructorRegistry()
    def propose(source, context):
        def child(lines):
            return dict(status='constructed', candidate='\n'.join(_boundary_metadata(source) + lines),
                        structural_witness=dict(removed_materialization_bank=True))
        first = child(['CCX q0 q1 q2'])
        first['alternatives'] = [child(['CCX q1 q0 q2']), child(['X q2','CCX q0 q1 q2','X q2'])]
        return first
    registry.register('algebraic_boundary_synthesis', propose, 'Controlled distinct equivalent lowerings')
    return dict(dict(clean_inputs=[4],max_operations=6,rounds=1,max_candidates=6,
                     registry=registry,deep_only=True),**overrides)


BANK = ['CCX q0 q1 q4','CX q4 q2','CCX q0 q1 q4']


def test_qualifying_region_siblings_are_individually_checked_and_priced_on_the_parent(tmp_path):
    source = tmp_path/'input.kmx'
    source.write_text('\n'.join(['X q8',*BANK,'Z q8']))
    result = restructure_file(source,tmp_path/'out',scan_range=(1,4),**_options(max_finalists=2))
    ledger = result['discovery_yield']
    assert ledger['unique_verified_cheaper_native_designs'] == 3
    assert ledger['extra_parent_finalists_checked'] == 2
    assert len(list((tmp_path/'out'/'parents').glob('*.kmx'))) == 3
    for record in ledger['designs'].values():
        full = km.Circuit.from_file(record['path'])
        assert record['full_parent_sha256'] == hashlib.sha256(Path(record['path']).read_bytes()).hexdigest()
        assert str(full).startswith('X q8\n') and str(full).endswith('Z q8')
        assert measured_resources(full) == record['resources']
        assert equivalent_windows(source.read_text(),str(full),clean_inputs=[4])['equivalent']
    assert all(row['check']['equivalent'] for row in ledger['contextual_checks'])


def test_regions_share_one_extra_budget_and_deduplicate_identical_parent_bytes(tmp_path,monkeypatch):
    source = tmp_path/'input.kmx'
    second = [line.replace('q4','q5') for line in BANK]
    source.write_text('\n'.join(['X q8',*BANK,'Z q8',*second,'Z q8']))
    regions = [(1,4),(1,4),(5,8)]
    monkeypatch.setattr(episode_search,'select_episodes',lambda *a,**kw: [
        dict(start=start,stop=stop,retrieval='test') for start,stop in regions])
    result = restructure_file(source,tmp_path/'out',**_options(max_finalists=2,clean_inputs=[4,5]))
    ledger = result['discovery_yield']
    assert ledger['extra_parent_finalists_checked'] == 2
    assert ledger['extra_finalists_charged'] == 2 and ledger['remaining_extra_finalists'] == 0
    assert ledger['unique_verified_cheaper_native_designs'] == 3
    children = [json.loads(Path(row['report']).read_text())['discovery_yield'] for row in result['attempts']]
    assert [c['max_extra_finalists'] for c in children] == [1,1,0]
    assert set(children[0]['designs']) == set(children[1]['designs'])
    assert len(list((tmp_path/'out'/'parents').glob('*.kmx'))) == 3


def test_parent_rejects_corrupt_local_bytes_and_undeclared_private_parent_bit(tmp_path,monkeypatch):
    import strata.restructure as restructure
    real = restructure.restructure_file
    source = tmp_path/'input.kmx'
    source.write_text('\n'.join(['X q8 if b0',*BANK,'Z q8']))
    def mutate(*args,**kwargs):
        run = real(*args,**kwargs)
        states = {state['id']:state for state in run['states']}
        siblings = [states[design['id']] for design in run['discovery_yield']['designs']
                    if design['id'] != run['best']['id']]
        siblings[0]['private_bits'] = [0]
        Path(siblings[1]['path']).write_text('NEG')
        return run
    monkeypatch.setattr(restructure,'restructure_file',mutate)
    result = real(source,tmp_path/'out',scan_range=(1,4),**_options(max_finalists=2))
    ledger = result['discovery_yield']
    assert ledger['unique_verified_cheaper_native_designs'] == 1
    failures = {row['reason'] for row in ledger['contextual_checks'] if row['status']=='rejected'}
    assert failures == {'local finalist bytes differ from the checked state',
                        'local finalist would conceal an undeclared parent bit'}


def test_cancellation_after_local_search_leaves_unchecked_candidates_out_of_parent_yield(tmp_path,monkeypatch):
    import strata.restructure as restructure
    real = restructure.restructure_file
    source = tmp_path/'input.kmx'
    source.write_text('\n'.join(['X q8',*BANK,'Z q8']))
    cancel = tmp_path/'stop'
    def stop_after_search(*args,**kwargs):
        run = real(*args,**kwargs)
        cancel.touch()
        return run
    monkeypatch.setattr(restructure,'restructure_file',stop_after_search)
    result = real(source,tmp_path/'out',scan_range=(1,4),cancel_file=cancel,**_options(max_finalists=2))
    assert result['discovery_yield']['unique_verified_cheaper_native_designs'] == 0
    assert result['discovery_yield']['contextual_checks'] == []
    assert result['search_control']['stop_reason'] == 'cancel_requested'
    assert Path(result['best']['path']).read_text() == str(km.Circuit.from_file(source))
