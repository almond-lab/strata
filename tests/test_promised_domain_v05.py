import itertools

import kickmix as km
import pytest

from strata.domains import load_domain
from strata.model import StrataError
from strata.promised_domain import (check_promised_candidate, load_search_domain,
                                    parse_address_promise, promised_addresses,
                                    promised_domain_synthesis)


def promise(address, arbitrary=(), clean=(), **values):
    return dict(version=1, type='address_promise', address_wires=list(address),
                arbitrary_wires=list(arbitrary), clean_wires=list(clean), **values)


def test_finite_relation_synthesizes_paid_product_away_with_exact_phase_and_dirty_targets():
    original = km.Circuit('CCX q0 q1 q2\nCZ q0 q1\nBIT_INVERT b0')
    rows = [dict(q0=x,q1=x,q2=y,b0=b) for x,y,b in itertools.product(range(2), repeat=3)]
    domain = dict(type='finite', assignments=rows)
    result = promised_domain_synthesis(original, domain)
    assert result['status'] == 'constructed'
    assert result['resource_bill']['before']['toffoli'] == 1
    assert result['resources']['toffoli'] == 0
    assert result['structural_witness']['promise_specialization']
    assert result['structural_witness']['off_domain_witness']
    assert result['check']['checked_cases'] == 8
    assert result['check']['all_input_equivalence'] is False
    other = km.Circuit(result['candidate'] + '\nZ q2')
    assert check_promised_candidate(original, other, domain)['equivalent'] is False


def test_11_bit_lookup_replays_each_valid_address_and_proves_all_dirty_word_values():
    # Legal addresses include both values of every address bit. The high pair
    # can only be simultaneously set in the padding above 1219.
    layout = 'APPEND_TO_REGISTER q51 r0\nAPPEND_TO_REGISTER b4 r0\nREGISTER r0 "through"'
    main = ([f'CCX q10 q9 q{q}' for q in range(11,51)] +
            [f'CX q2 q{q}' for q in range(11,51)] +
            ['CCX q0 q1 q11','CCZ q10 q9 q2','CZ q1 q3','BIT_STORE1 b4'])
    original = km.Circuit('\n'.join([layout,*main]))
    domain = promise(range(11), range(11,51), [51], interval=[0,1219])
    result = promised_domain_synthesis(original, domain, clean_inputs=[51])
    assert result['status'] == 'constructed'
    assert result['source_rows_replayed'] == 1220
    assert result['check']['assignment_count'] == 1220 << 40
    assert result['check']['clean_outputs_restored'] and result['check']['exact_phase']
    assert result['check']['address_count'] == 1220
    assert result['resources']['toffoli'] == 1
    assert result['resource_bill']['before']['toffoli'] == 42
    assert result['structural_witness']['off_domain_witness']
    candidate = km.Circuit(result['candidate'])
    assert original.register_data['through'] == candidate.register_data['through']
    for bad, expected in [('CCX q11 q0 q12','q12'), ('Z q17','phase'),
                          ('X q51','q51'), ('BIT_INVERT b4','b4')]:
        check = check_promised_candidate(original, km.Circuit(str(candidate)+'\n'+bad),
                                        domain, clean_inputs=[51])
        assert check['equivalent'] is False and check['output'] == expected
    assert check_promised_candidate(original, candidate, dict(domain,interval=[0,2047]), clean_inputs=[51])['equivalent'] is False


def test_symbolic_range_with_moved_measurements_covers_every_outcome_without_expanding_range():
    layout = 'APPEND_TO_REGISTER q31 r0\nREGISTER r0 "workspace"'
    left = km.Circuit(layout+'\nCCX q0 q1 q30\nHMR q30 b0\nCZ q0 q1 if b0\nCCZ q21 q20 q19')
    right = km.Circuit(layout+'\nHMR q31 b0')
    compact = promise(range(22), range(22,30), [30,31], interval=[1,3133452])
    opts = dict(clean_inputs=[30,31],measurement_mode='independent-map',internal_bits=[0])
    bound = parse_address_promise(compact, left, clean_inputs=opts['clean_inputs'])
    assert isinstance(promised_addresses(bound), range)
    assert len(promised_addresses(bound)) == 3133452 and 'assignments' not in bound
    check = check_promised_candidate(left, right, bound, **opts)
    assert check['equivalent'] and check['measurement_independent']
    assert check['assignment_count'] == 3133452 << 8
    assert check['off_domain_witness'] and check['original_outcomes'] == 1
    debt = km.Circuit(str(right)+'\nZ q1 if b0')
    assert check_promised_candidate(left, debt, bound, **opts)['reason'] == 'public_value_or_phase_depends_on_outcomes'
    assert check_promised_candidate(left,right,bound,clean_inputs=[30,31])['reason'] == 'measurement_event_alignment_unproved'
    assert check_promised_candidate(left,right,bound,measurement_mode='independent-map',
                                   clean_inputs=[30,31])['reason'] == 'public_value_or_phase_depends_on_outcomes'
    assert promised_domain_synthesis(left,bound,clean_inputs=[30,31])['status'] == 'unknown'


@pytest.mark.parametrize('legal', [[1],[2],[0,3],[1,2],[0,1,2]])
def test_sparse_lsb_order_matches_explicit_exhaustion(legal):
    native = km.Circuit('CCX q0 q2 q1\nCZ q0 q1')
    other = km.Circuit('CX q2 q1\nCZ q0 q1')
    compact = promise([2,0],[1],addresses=legal)
    rows = [[a >> 1,y,a & 1] for a in legal for y in range(2)]
    assert check_promised_candidate(native,other,compact)['equivalent'] == check_promised_candidate(
        native,other,dict(assignments=rows))['equivalent']


def test_incomplete_domains_observations_and_false_bindings_do_not_authorize_search():
    native = km.Circuit('CCX q0 q1 q2')
    valid = promise([2,0],[1],addresses=[2,1,1])
    bound = load_search_domain(valid,native)
    assert list(promised_addresses(bound)) == [1,2]
    assert load_search_domain(bound,native) == bound
    invalid = [dict(valid, arbitrary_wires=[]), dict(valid,clean_wires=[1]),
               dict(valid,domain_hash='x'), dict(valid,interval=[0,2]), dict(valid,addresses=[4]),
               dict(valid,addresses=[True]), dict(valid,version=True), dict(valid,address_count=3),
               dict(type='boundary_observations',rows=[{'q0': 0}],coverage='exhaustive_initial_cube'),
               dict(assignments=[dict(q0=0,q1=0)])]
    for spec in invalid:
        with pytest.raises(StrataError):
            promised_domain_synthesis(native, spec)
    with pytest.raises(StrataError):
        parse_address_promise(dict(valid, clean_wires=[1], arbitrary_wires=[]),native)
    assert check_promised_candidate(native,native,bound,max_terms=2)['equivalent'] is None


def test_classical_register_alias_cannot_supply_a_missing_quantum_input():
    native = km.Circuit('APPEND_TO_REGISTER q0 r0\nAPPEND_TO_REGISTER b1 r0\n'
                        'REGISTER r0 "mixed"\nCX q0 q1')
    with pytest.raises(StrataError):
        load_domain(dict(assignments=[{'mixed': 3}]),native)


def test_independent_check_binds_public_register_order_and_classical_input():
    left = km.Circuit('APPEND_TO_REGISTER q0 r0\nAPPEND_TO_REGISTER b0 r0\nREGISTER r0 "out"\nX q1 if b0')
    switched = km.Circuit('APPEND_TO_REGISTER b0 r0\nAPPEND_TO_REGISTER q0 r0\nREGISTER r0 "out"\nX q1 if b0')
    domain = promise([0],[1],interval=[0,1])
    assert check_promised_candidate(left,switched,domain)['reason'] == 'public_named_register_binding_changed'
    no_control = km.Circuit(str(left)+'\nX q1 if b0')
    assert check_promised_candidate(left,no_control,domain)['equivalent'] is False
    assert check_promised_candidate(left,no_control,domain)['counterexample']['b0'] == 1


def test_compact_generation_wont_accept_zero_dirty_probe_as_proof():
    native = km.Circuit('CCX q2 q0 q1')
    domain = promise([0],[1,2],interval=[0,1])
    result = promised_domain_synthesis(native,domain)
    assert result['status'] == 'not_applicable'
    assert any(t['status'] == 'different' for t in result['trials'])
    assert 'candidate' not in result


def test_cofactor_completion_lowers_a_degree_four_lookup_on_an_incomplete_cube():
    gates = ['CCX q0 q1 q6','CCX q6 q2 q7','CCX q7 q3 q5',
             'CCX q6 q2 q7','CCX q0 q1 q6','CCX q3 q4 q5']
    original = km.Circuit('\n'.join(gates))
    domain = promise(range(5),[5],[6,7],addresses=[x for x in range(32) if x & 24 != 24])
    result = promised_domain_synthesis(original,domain,clean_inputs=[6,7])
    assert result['status'] == 'constructed'
    witness = result['structural_witness']
    assert witness['promise_specialization'] and witness['selected_completion'] == 'compatible_cofactor_completion'
    assert result['resources']['toffoli'] == 5
    assert result['check']['clean_outputs_restored']
    assert result['check']['assignment_count'] == 48


def test_restructure_entry_keeps_scoped_gain_out_of_learned_all_input_templates(tmp_path):
    import json
    from strata.restructure import restructure_file

    source = tmp_path/'input.kmx'
    source.write_text('CCX q0 q1 q2\n')
    domain = tmp_path/'promise.json'
    domain.write_text(json.dumps(promise([0,1],[2],addresses=[0,3])))
    scoped = restructure_file(source, tmp_path/'scoped', domain=domain, deep_only=True, fast=True)
    assert scoped['status'] == 'verified_improvement'
    check = scoped['attempts'][0]['check']
    assert not check['all_input_equivalence'] and check['domain_hash'] == scoped['contract']['domain_hash']
    assert scoped['best']['id'] == check['candidate_hash']
    assert scoped['baseline']['id'] == check['original_hash']
    assert scoped['learned_template_count'] == 0
    assert scoped['structural_gains'][0]['qualification'] == 'promise_specialization'


@pytest.mark.parametrize('compact', [True,False])
def test_private_outcome_is_hidden_only_at_its_own_output(compact):
    layout = ('APPEND_TO_REGISTER b1 r0\nAPPEND_TO_REGISTER q2 r0\nREGISTER r0 "result"\n'
              'BIT_INVERT b1\nBIT_INVERT b1\n')
    original = km.Circuit(layout+'HMR q2 b0')
    moved = km.Circuit(layout+'HMR q1 b0')
    rows = [dict(q0=x,q1=0,q2=0,b1=b) for x,b in itertools.product(range(2), repeat=2)]
    domain = promise([0],[],[1,2],interval=[0,1]) if compact else dict(assignments=rows)
    opts = dict(clean_inputs=[1,2],measurement_mode='independent-map')
    assert check_promised_candidate(original,moved,domain,internal_bits=[0],**opts)['equivalent'] is True
    assert check_promised_candidate(original,moved,domain,**opts)['equivalent'] is None
    for leak in ('X q0 if b0', 'Z q0 if b0', 'BIT_INVERT b1 if b0'):
        bad = km.Circuit(str(moved)+'\n'+leak)
        assert check_promised_candidate(original,bad,domain,internal_bits=[0],**opts)['equivalent'] is not True
