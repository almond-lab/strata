"""Strata: contract-driven restructuring on kickmix."""
from .project import Project, Design, Line
from .model import Contract, Region, Register, Debt, Floor, StrataError
from .engine import Patch
from .queries import Queries


def open(path='strata.toml', adapters=None):
    if adapters is None:
        from .demo import AndFusion
        adapters = {'and_fusion': AndFusion()}
    return Project(path, adapters)
