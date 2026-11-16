"""Extensible semantic block registry: matching requires a checker, not a label."""
from dataclasses import dataclass
from typing import Callable
from .model import StrataError


@dataclass
class Block:
    name: str
    match: Callable
    certify: Callable


class BlockLibrary:
    def __init__(self):self.blocks={}
    def register(self,block):
        if block.name in self.blocks:raise StrataError('duplicate block')
        self.blocks[block.name]=block
    def recognize(self,operations):
        for b in self.blocks.values():
            match=b.match(operations)
            if match is not None:
                certificate=b.certify(operations,match)
                if certificate.get('passed') is True and certificate.get('kind') in {'proof','exhaustive'}:
                    return dict(kind=b.name,binding=match,certificate=certificate)
        return dict(kind='opaque',certificate=None)


def product_xor_block():
    # Public algebra: t ^= a*b; y ^= t; t ^= a*b gives y ^= a*b
    # only when t is initially zero. The promise is retained in the binding.
    def match(ops):
        if len(ops)!=3:return None
        a,b,c=ops
        if a[0]!='CCX' or b[0]!='CX' or c!=a:return None
        controls=a[1]
        if len(controls)!=3 or len(b[1])!=2 or b[1][0]!=controls[2]:return None
        if len(set((*controls,b[1][1])))!=4:return None
        return dict(a=controls[0],b=controls[1],scratch=controls[2],target=b[1][1],promise='scratch=0')
    def certify(ops,binding):
        for a in (0,1):
            for b in (0,1):
                for y in (0,1):
                    t=a&b;out=y^t;t^=a&b
                    if t or out!=(y^(a&b)):return dict(passed=False)
        return dict(passed=True,kind='exhaustive',cases=8,scope='four-wire schema; clean scratch; all dirty-target values')
    return Block('materialized_product_xor',match,certify)
