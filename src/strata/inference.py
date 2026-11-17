"""Exact, scoped precondition queries from recovered Boolean functions."""
from itertools import combinations
from math import ceil, log2
from time import monotonic
from .model import StrataError, digest
from .semantic import analyze_window


def _poly(row):
    return {frozenset(t) for t in row['monomials']}


def _multiply(a,b,limit):
    if len(a)*len(b)>limit:raise StrataError('functional query product budget exceeded')
    result=set()
    for x in a:
        for y in b:
            t=x|y
            if t in result:result.remove(t)
            else:result.add(t)
    if len(result)>limit:raise StrataError('functional query term budget exceeded')
    return result


def functions_at(circuit,at=None,*,clean_inputs=(),max_variables=128,max_terms=65536,max_operations=10000):
    lines=str(circuit).splitlines()
    at=len(lines) if at is None else at
    if type(at) is not int or not 0<=at<=len(lines):raise StrataError('query boundary outside circuit')
    clean=set(clean_inputs)
    if any(type(q) is not int or not 0<=q<circuit.num_qubits for q in clean):raise StrataError('invalid clean input')
    sem=analyze_window(lines[:at],max_variables=max_variables,max_terms=max_terms,
                       max_operations=max_operations,max_product_terms=max_terms*16)
    if sem['status']!='exact':return dict(status='unknown',reason=sem.get('reason'),at=at)
    funcs={q:_poly(sem['outputs'].get(str(q),{'monomials':[[q]]})) for q in range(circuit.num_qubits)}
    funcs={q:{t for t in p if not t&clean} for q,p in funcs.items()}
    return dict(status='exact',functions=funcs,at=at,clean_inputs=sorted(clean),
                scope='all arbitrary circuit inputs except explicitly declared zero inputs')


def is_function_of(circuit,wire,at,support,*,degree=2,max_forms=4096,timeout=5,clean_inputs=(),**budgets):
    support=tuple(support)
    if len(set(support))!=len(support) or wire in support:raise StrataError('support must be distinct and exclude target')
    if any(type(q) is not int or not 0<=q<circuit.num_qubits for q in (wire,*support)):raise StrataError('query wire out of range')
    if type(degree) is not int or not 0<=degree<=3 or max_forms<1 or timeout<=0:raise StrataError('invalid functional-search budget')
    start=monotonic();family=dict(kind='Boolean polynomial in current support values',max_degree=degree,support=list(support))
    result=functions_at(circuit,at,clean_inputs=clean_inputs,**budgets)
    if result['status']!='exact':return dict(result,searched_family=family)
    funcs=result['functions'];target=funcs[wire]
    columns=[];forms=[]
    try:
        for size in range(degree+1):
            for form in combinations(support,size):
                if len(forms)>=max_forms or monotonic()-start>timeout:
                    return dict(status='unknown',reason='functional search budget',searched_family=family,forms=len(forms))
                p={frozenset()}
                for q in form:p=_multiply(p,funcs[q],budgets.get('max_terms',65536))
                columns.append(p);forms.append(form)
        basis={}
        def reduce(p,coeff):
            p=set(p)
            while p:
                pivot=max(p,key=lambda t:(len(t),tuple(sorted(t))))
                if pivot not in basis:break
                row,mask=basis[pivot];p^=row;coeff^=mask
            return p,coeff
        for index,p in enumerate(columns):
            if monotonic()-start>timeout:return dict(status='unknown',reason='linear solve budget',searched_family=family)
            row,mask=reduce(p,1<<index)
            if row:basis[max(row,key=lambda t:(len(t),tuple(sorted(t))))]=(row,mask)
        remainder,solution=reduce(target,0)
        evidence=dict(kind='symbolic',at=result['at'],clean_inputs=result['clean_inputs'],
                      forms_searched=len(forms),linear_rank=len(basis),scope=result['scope'])
        if remainder:return dict(status='no',searched_family=family,evidence=evidence,
                                  reason='target polynomial is outside the exact span of the searched family')
        selected=[list(form) for i,form in enumerate(forms) if solution>>i&1]
        check=set()
        for i in range(len(forms)):
            if solution>>i&1:check^=columns[i]
        if check!=target:raise StrataError('functional witness failed replay')
        return dict(status='yes',wire=wire,at=result['at'],form=dict(monomials=selected,
                    expression=' ^ '.join('*'.join(f'q{q}' for q in t) or '1' for t in selected) or '0'),
                    searched_family=family,evidence=evidence,witness_hash=digest(selected))
    except StrataError as exc:
        return dict(status='unknown',reason=str(exc),searched_family=family)


def information_slack(circuit,at=None,*,wires=None,clean_inputs=(),max_inputs=16,**budgets):
    selected=list(range(circuit.num_qubits)) if wires is None else list(wires)
    if len(set(selected))!=len(selected) or any(type(q) is not int or not 0<=q<circuit.num_qubits for q in selected):raise StrataError('invalid slack wire set')
    result=functions_at(circuit,at,clean_inputs=clean_inputs,**budgets)
    if result['status']!='exact':return result
    funcs=[result['functions'][q] for q in selected]
    inputs=sorted({q for p in funcs for t in p for q in t})
    if all(len(t)<=1 for p in funcs for t in p):
        basis={}
        for p in funcs:
            row={next(iter(t)) for t in p if t}
            while row and max(row) in basis:row^=basis[max(row)]
            if row:basis[max(row)]=row
        bits=len(basis);kind='exact_affine_rank';distinct=1<<bits
    else:
        if len(inputs)>max_inputs:return dict(status='unknown',reason='distinct-state enumeration budget',inputs=len(inputs))
        patterns=set()
        for assignment in range(1<<len(inputs)):
            on={q for i,q in enumerate(inputs) if assignment>>i&1}
            patterns.add(tuple(sum(t<=on for t in p)%2 for p in funcs))
        distinct=len(patterns);bits=(distinct-1).bit_length();kind='exact_full'
    return dict(status='exact',kind=kind,wires=selected,distinct_states=distinct,
                information_bits=bits,slack_bits=len(selected)-bits,
                clean_inputs=result['clean_inputs'],scope=result['scope'],
                note='Information compression bound only; does not establish a free clean scratch wire or cheap encoder.')
