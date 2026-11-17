"""Strata: contract-driven restructuring on kickmix."""
from .project import Project, Design, Line
from .model import Contract, Region, Register, Debt, Floor, StrataError
from .engine import Patch
from .queries import Queries
from .research import Hypothesis, Dependency, ResearchPolicy, ResearchRound
from .file_workflow import inspect_file, optimize_file
from .discovery import discovery_brief, evaluate_candidate, record_research_note


def open(path='strata.toml', adapters=None):
    if adapters is None:
        from .demo import AndFusion
        adapters = {'and_fusion': AndFusion()}
    return Project(path, adapters)
