#!/usr/bin/env python3
"""Opt-in measured algebraic candidate search; native finalist counts stay explicit."""
import argparse
import json
import sys
import platform
import resource
import hashlib
from importlib.metadata import version
from pathlib import Path
from strata.candidate_search import generated_contract,search_candidates


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidates',type=int,default=1_000_000)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--native-limit',type=int,default=8)
    parser.add_argument('--materialization-limit',type=int,default=128)
    args=parser.parse_args()
    runs=[]
    for name,shape,seed in [('development',(3,3,2),8021),('held_out_partition',(2,4,2),19331)]:
        tensor,instance=generated_contract(shape,seed=seed)
        for strategy in ('random','guided'):
            report=search_candidates(tensor,strategy=strategy,candidate_budget=args.candidates,
                native_limit=args.native_limit,materialization_limit=args.materialization_limit,seed=seed+91)
            runs.append(dict(case=name,instance=instance,**report))
            print(json.dumps(dict(case=name,strategy=strategy,counts=report['counts'],
                seconds=report['timings_seconds']['total'],accepted_gain=report['accepted_gain'])),flush=True)
    import strata.candidate_search as implementation
    code_path=Path(implementation.__file__)
    result=dict(experiment='bounded four-product bilinear candidate search',runs=runs,
        totals=dict(algebraic_specification_evaluations=sum(r['counts']['unique_complete_specifications_screened'] for r in runs),
            random_unique_specifications_screened=sum(r['counts']['unique_complete_specifications_screened'] for r in runs if r['strategy']=='random'),
            native_candidate_replays=sum(r['counts']['native_candidates_checked'] for r in runs),
            distinct_final_native_artifacts=len({c['artifact_sha256'] for r in runs for c in r['verified_candidates']}),
            native_input_shots=sum(r['counts']['native_input_shots'] for r in runs),
            final_exact_proof_runs=sum(r['counts']['final_exact_proofs'] for r in runs)),
        environment=dict(python=sys.version,numpy=version('numpy'),kickmix=version('kickmix'),
            platform=platform.platform(),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
        implementation_sha256=hashlib.sha256(code_path.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        comparison_scope='same candidate family, candidate cap, materialization cap, native cap and objective; actual partial work/time differ',
        held_out_scope='different input partition and independently fixed instance seed; no training or general discovery claim',
        warning='Millions count distinct algebraic decomposition specifications. Native circuits, shots and final exact proofs have separate counters.')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
