"""Native benchmark counters represent distinct executed programs, not shots."""
import importlib.util
from math import comb
from pathlib import Path

import pytest


def benchmark_module():
    path=Path(__file__).resolve().parents[1]/'scripts'/'benchmark_artifact_pipeline.py'
    spec=importlib.util.spec_from_file_location('native_candidate_benchmark',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_pipeline_counts_programs_and_shots_separately_and_reproduces_specs():
    benchmark=benchmark_module()
    first=benchmark.run(candidates=32,seed=192)
    repeat=benchmark.run(candidates=32,seed=192)
    counts=first['counts']
    assert counts['unique_specs']==counts['unique_native_canonical_programs']==counts['native_simulator_calls']==32
    assert counts['native_canonical_duplicates']==0
    assert counts['native_input_trajectories']==32*256
    assert counts['measurement_events']==counts['forced_measurement_outcomes']==0
    assert counts['independently_spec_checked_candidates']==32
    assert first['candidate_family']['complete_family_size']==comb(147,4)
    assert first['artifacts']['native_program_sequence_sha256']==repeat['artifacts']['native_program_sequence_sha256']
    assert first['verification']['every_candidate_ran_in_native_sdk']
    assert first['verification']['python_prefilter_rejections']==0


@pytest.mark.parametrize('mutation,error',[('\nZ q0','relative sign'),('\nX q8','tensor oracle')])
def test_native_pipeline_rejects_phase_and_helper_corruption(monkeypatch,mutation,error):
    benchmark=benchmark_module()
    original=benchmark.emit
    monkeypatch.setattr(benchmark,'emit',lambda spec,encoded:original(spec,encoded)+mutation)
    with pytest.raises(AssertionError,match=error):benchmark.run(candidates=1)


def test_screening_and_native_pipeline_use_the_same_atom_order():
    from strata.candidate_search import atom_catalog
    benchmark=benchmark_module()
    atoms,codes=atom_catalog((3,3,2))
    assert [(atom['left_mask'],atom['right_mask'],atom['output_mask']) for atom in atoms]==list(benchmark.ATOMS)
    assert all(benchmark.coefficients([i])==int(code) for i,code in enumerate(codes))
