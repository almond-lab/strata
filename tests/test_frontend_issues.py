from fractions import Fraction
import kickmix as km
import pytest
from strata.frontends import OperationStream,load_circuit
from strata.model import Contract,Register,StrataError
from strata.lab import exhaustive


def test_two_frontends_share_analysis_and_lab_without_semantic_change(tmp_path):
    c=km.Circuit('CCX q0 q1 q2\nHMR q2 b0\nCZ q0 q1 if b0')
    frontend=OperationStream();p=tmp_path/'c.ops';p.write_text(frontend.serialize(c))
    loaded=load_circuit(p)
    assert str(loaded)==str(c)
    assert frontend.analyze(loaded,tmp_path/'raw')['metrics']==frontend.analyze(c,tmp_path/'native')['metrics']
    contract=Contract('clean product',(Register('a',(0,)),Register('b',(1,)),Register('t',(2,),role='clean')),'identity')
    reference=lambda values,scale:({'a':values['a'],'b':values['b']},Fraction(0))
    a=exhaustive(c,contract,reference);b=exhaustive(loaded,contract,reference)
    assert a.passed and b.passed and a.tested==b.tested


def test_raw_frontend_rejects_unknown_or_lossy_operands():
    f=OperationStream()
    for text in ('{"op":"EXEC","text":"bad"}','{"op":"CCX","qubits":[0,1]}','{"op":"X","qubits":[0],"bit":0}'):
        with pytest.raises(StrataError):f.parse(text)
    c=km.Circuit('INFO_TAG "a b"\nX q0\nBIT_STORE1 b0\nZ q0 if b0')
    assert str(f.parse(f.serialize(c)))==str(c)
def test_nested_condition_stack_and_literal_if_tag_round_trip():
    import kickmix as km
    from strata.frontends import OperationStream
    native=km.Circuit('INFO_TAG "if"\nBIT_STORE1 b0\nPUSH_CONDITION if b0\nX q0\nPOP_CONDITION\nZ q0 if b0')
    codec=OperationStream()
    encoded=codec.serialize(native)
    assert str(codec.parse(encoded))==str(native)
