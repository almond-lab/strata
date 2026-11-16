"""Independent adversarial regression cases from the bounded alpha review."""
from dataclasses import replace
from pathlib import Path
import time

import kickmix as km
import pytest

from strata import open as open_project
from strata.demo import AndFusion
from strata.lab import exhaustive
from strata.model import Contract, Register, StrataError


@pytest.fixture
def review_project(tmp_path):
    example = Path(__file__).parents[1] / 'examples' / 'and_fusion'
    for name in ('strata.toml', 'base.kmx'):
        (tmp_path / name).write_bytes((example / name).read_bytes())
    project = open_project(tmp_path / 'strata.toml')
    project.init()
    return project


def test_mutating_patch_after_build_cannot_certify_unbuilt_implementation(review_project):
    project = review_project
    line = project.open_line(project.task('and').base_design(), 'replace-stages', ['root'],
                             implementation={'fused': True})
    assert line.lab(bits=1).passed
    line.build()
    line.patch.implementation['fused'] = False
    with pytest.raises(StrataError):
        line.verify()


@pytest.mark.parametrize('consumer', ['best', 'replay'])
def test_design_cost_must_match_its_certificate(review_project, consumer):
    project = review_project
    base = project.task('and').base_design()
    forged = project.store.put('design', {**base.record, 'cost': 1})
    with pytest.raises(StrataError):
        project.best() if consumer == 'best' else project.replay(forged)


@pytest.mark.parametrize('consumer', ['best', 'replay'])
def test_failed_lab_receipt_cannot_be_attached_to_passing_design(review_project, consumer):
    project = review_project
    base = project.task('and').base_design()
    certificate = project.store.get(base.record['certificate'], 'certificate')
    certificate['lab'] = {**certificate['lab'], 'passed': False, 'exhaustive': False}
    cert_hash = project.store.put('certificate', certificate)
    forged = project.store.put('design', {**base.record, 'certificate': cert_hash})
    with pytest.raises(StrataError):
        project.best() if consumer == 'best' else project.replay(forged)


class OutsideRegionAdapter(AndFusion):
    def build(self, parameters, implementation):
        circuit, tree = super().build(parameters, {})
        if implementation.get('outside'):
            circuit = km.Circuit(str(circuit) + '\nX q0\nX q0')
            tree = replace(tree, stop=tree.stop + 2)
        return circuit, tree


def test_new_operations_outside_declared_region_are_rejected(tmp_path):
    example = Path(__file__).parents[1] / 'examples' / 'and_fusion'
    for name in ('strata.toml', 'base.kmx'):
        (tmp_path / name).write_bytes((example / name).read_bytes())
    project = open_project(tmp_path / 'strata.toml', adapters={'and_fusion': OutsideRegionAdapter()})
    project.init()
    line = project.open_line(project.task('and').base_design(), 're-synthesise', ['lane-0'],
                             implementation={'outside': True})
    with pytest.raises(StrataError):
        line.lab(bits=1)
        line.build()


def test_lab_timeout_covers_reference_evaluation():
    contract = Contract('identity', (Register('x', (0,)),), 'identity')

    def slow_reference(values, scale):
        time.sleep(0.02)
        return {'x': values['x']}, 0

    with pytest.raises(StrataError, match='timed out'):
        exhaustive(km.Circuit('X q0\nX q0'), contract, slow_reference, timeout=0.01)


def test_replay_requires_existing_certified_parent_chain(review_project):
    project = review_project
    line = project.open_line(project.task('and').base_design(), 'replace-stages', ['root'],
                             implementation={'fused': True})
    line.lab(bits=1)
    line.build()
    child = line.verify()
    forged = project.store.put('design', {**child.record, 'parent': 'f' * 64})
    with pytest.raises(StrataError):
        project.replay(forged)


class ChangedLocalContractAdapter(AndFusion):
    def build(self, parameters, implementation):
        circuit, tree = super().build(parameters, {})
        if implementation.get('changed_local_contract'):
            first = tree.children[0]
            first = replace(first, contract=replace(first.contract, reference='unrelated-function'))
            tree = replace(tree, children=(first, *tree.children[1:]))
        return circuit, tree


def test_l2_redesign_preserves_target_contract(tmp_path):
    example = Path(__file__).parents[1] / 'examples' / 'and_fusion'
    for name in ('strata.toml', 'base.kmx'):
        (tmp_path / name).write_bytes((example / name).read_bytes())
    project = open_project(tmp_path / 'strata.toml', adapters={'and_fusion': ChangedLocalContractAdapter()})
    project.init()
    line = project.open_line(project.task('and').base_design(), 're-synthesise', ['lane-0'],
                             implementation={'changed_local_contract': True})
    with pytest.raises(StrataError):
        line.lab(bits=1)
        line.build()


@pytest.mark.parametrize('gate', ['Z_POW q0 0.25', 'Z_POW q0 -0.25'])
def test_fractional_phase_error_is_not_truncated_to_zero(gate):
    contract = Contract('identity', (Register('x', (0,)),), 'identity')
    result = exhaustive(km.Circuit(gate), contract, lambda values, scale: ({'x': values['x']}, 0))
    assert not result.passed
    assert result.counterexample['input']['x'] == 1
    assert not result.counterexample['values']


def test_exact_fractional_phase_reference_uses_native_pi_units():
    from fractions import Fraction
    contract = Contract('quarter-turn', (Register('x', (0,)),), 'quarter-turn')
    result = exhaustive(km.Circuit('Z_POW q0 0.25'), contract,
                        lambda values, scale: ({'x': values['x']}, Fraction(values['x'], 4)))
    assert result.passed and result.exhaustive


def test_phase_comparison_is_periodic_modulo_two_pi():
    contract = Contract('identity', (Register('x', (0,)),), 'identity')
    result = exhaustive(km.Circuit('X q0\nX q0'), contract,
                        lambda values, scale: ({'x': values['x']}, 2))
    assert result.passed and result.exhaustive


def test_conditional_hmr_and_reused_measurement_bits_keep_every_branch():
    contract = Contract('clean-and', (Register('a', (0,)), Register('b', (1,)),
                                    Register('scratch', (2,), role='clean')), 'clean-and')
    circuit = km.Circuit('CCX q0 q1 q2\nHMR q2 b0\nCZ q0 q1 if b0\n'
                         'CCX q0 q1 q2 if b0\nHMR q2 b1 if b0\nCZ q0 q1 if b1\n'
                         'CCX q0 q1 q2\nHMR q2 b0\nCZ q0 q1 if b0')
    result = exhaustive(circuit, contract,
                        lambda values, scale: ({'a': values['a'], 'b': values['b']}, 0))
    assert result.passed and result.exhaustive
    assert result.cases == 4 and result.branches == 8 and result.tested == 32


class RenamedFusionAdapter(AndFusion):
    def build(self, parameters, implementation):
        from strata.model import Region
        circuit, tree = super().build(parameters, {'fused': implementation.get('merge', False)})
        if implementation.get('merge'):
            merged = Region('merged', tree.start, tree.stop, tree.contract, kind='and-xor')
            tree = replace(tree, children=(merged,))
        return circuit, tree


def test_l3_fusion_can_replace_selected_regions_with_new_region(tmp_path):
    example = Path(__file__).parents[1] / 'examples' / 'and_fusion'
    for name in ('strata.toml', 'base.kmx'):
        (tmp_path / name).write_bytes((example / name).read_bytes())
    project = open_project(tmp_path / 'strata.toml', adapters={'and_fusion': RenamedFusionAdapter()})
    project.init()
    line = project.open_line(project.task('and').base_design(), 'fuse',
                             ['lane-0', 'lane-1', 'lane-2'], implementation={'merge': True},
                             replacement_regions=['merged'])
    assert line.lab(bits=1).passed
    line.build()
    design = line.verify()
    assert design.record['depth'] == 'L3'
    assert [child.id for child in design.tree.children] == ['merged']
    assert project.replay(design.id)['passed']
