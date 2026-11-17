"""Release blockers found by independent proof/probability auditing."""
from dataclasses import replace

import pytest

from strata.floors import exact_minimum
from strata.model import StrataError
from strata.probability import ProbabilityPolicy
from strata.project import Design
from test_probability_project import check, make_project


@pytest.mark.parametrize('nested', ['polynomial', 'monomial'])
def test_exact_floor_does_not_consume_an_and_into_an_affine_function(nested):
    outputs = [iter([[0, 1]])] if nested == 'polynomial' else [[iter([0, 1])]]
    # Rejecting unsupported input containers is safe; silently certifying zero
    # ANDs after validation consumes their generators is not.
    try:
        result = exact_minimum(outputs, 2)
    except (StrataError, TypeError):
        return
    assert result['kind'] == 'exact_small' and result['value'] == 1
    assert result['proven_lower_bound'] == 1


def draws(failure, cost=0.1):
    return dict(seeds=[17, 29], draws=[dict(seed=seed, resources={
        'cost': cost, 'fail_rate': failure}) for seed in (17, 29)])


@pytest.mark.parametrize('weight', [1e308, 5e-324])
def test_finite_extreme_weights_cannot_erase_failure_or_cost(weight):
    policy = ProbabilityPolicy((17, 29), max_failure=0.1, weights=(weight, weight))
    with pytest.raises(StrataError):
        policy.evaluate(draws(0.5), 'cost')
    safe = ProbabilityPolicy((17, 29), weights=(weight, weight)).evaluate(draws(0), 'cost')
    assert safe['cost'] == pytest.approx(0.1)


def lucky_oracle_only(request):
    receipt = check(request)
    if request.stage == 'oracle_full':
        return replace(receipt, details={**receipt.details, 'selected_seed': 11})
    return receipt


def test_lucky_draw_in_oracle_gate_cannot_be_ingested(tmp_path):
    project = make_project(tmp_path, lucky_oracle_only)
    with pytest.raises(StrataError):
        project.init()
    assert not project.store.all('design')


def test_saved_certificate_cannot_add_lucky_oracle_selection(tmp_path):
    project = make_project(tmp_path)
    design = Design(project, project.init()['numeric'])
    certificate = project.store.get(design.record['certificate'])
    certificate['full']['oracle']['selected_seed'] = 11
    changed_certificate = project.store.put('certificate', certificate)
    changed_design = project.store.put('design', {**design.record, 'certificate': changed_certificate})
    with pytest.raises(StrataError):
        Design(project, changed_design).validate()
