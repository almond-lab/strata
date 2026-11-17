"""Input front ends share the native execution backend through explicit codecs."""
from pathlib import Path
import json,re,shlex
import kickmix as km
from .kickmix import Kickmix
from .model import StrataError


class OperationStream:
    """JSONL reversible-operation records; no SDK object in the file format.

    Each record has op, qubits, and optionally bit/condition/name/text. Qubit and
    classical-bit identities are nonnegative integers. Native execution is the
    shared backend; this is a second front end, not a second simulator.
    """
    quantum={'X':1,'CX':2,'CCX':3,'SWAP':2,'CSWAP':3,'Z':1,'CZ':2,'CCZ':3,'R':1,'HMR':1,'NEG':0}
    bits={'BIT_STORE0','BIT_STORE1','BIT_INVERT'}

    def parse(self,text):
        lines=[]
        for number,line in enumerate(text.splitlines(),1):
            if not line.strip():continue
            try:r=json.loads(line)
            except ValueError as exc:raise StrataError(f'operation record {number} is not JSON') from exc
            if not isinstance(r,dict) or set(r)-{'op','qubits','bit','condition','name','text'}:raise StrataError('unknown operation fields')
            op=r.get('op');qs=r.get('qubits',[])
            if not isinstance(qs,list) or any(type(q)is not int or q<0 for q in qs):raise StrataError('operation qubits must be nonnegative integers')
            if len(set(qs))!=len(qs):raise StrataError('duplicate qubits in operation')
            def bit(key):
                b=r.get(key)
                if type(b)is not int or b<0:raise StrataError('operation bit must be a nonnegative integer')
                return f'b{b}'
            if op in self.quantum:
                if len(qs)!=self.quantum[op]:raise StrataError('operation arity mismatch')
                words=[op,*[f'q{q}' for q in qs]]
                if op=='HMR':words.append(bit('bit'))
            elif op in self.bits:
                if qs:raise StrataError('classical operation has qubits')
                words=[op,bit('bit')]
            elif op=='PUSH_CONDITION':
                if qs or 'condition' in r:raise StrataError('push takes one bit operand and no inline condition')
                words=[op,'if',bit('bit')]
            elif op=='POP_CONDITION':
                if qs or 'condition' in r:raise StrataError('pop cannot have qubits or an inline condition')
                words=[op]
            elif op in {'REGISTER','APPEND_TO_REGISTER'}:
                if not isinstance(r.get('name'),str):raise StrataError('register name required')
                words=[op,json.dumps(r['name']),*[f'q{q}' for q in qs]]
            elif op=='INFO_TAG':
                if not isinstance(r.get('text'),str):raise StrataError('tag text required')
                words=[op,json.dumps(r['text'])]
            else:raise StrataError('unsupported raw reversible operation')
            if 'condition'in r:words+=['if',bit('condition')]
            if 'bit'in r and op not in {'HMR','PUSH_CONDITION'} and op not in self.bits:raise StrataError('unexpected bit operand')
            lines.append(' '.join(words))
        return km.Circuit('\n'.join(lines))

    def serialize(self,circuit):
        records=[]
        for line in str(circuit).splitlines():
            w=shlex.split(line)
            if not w:continue
            op=w[0];r={'op':op}
            if op=='PUSH_CONDITION':
                if len(w)!=3 or w[1]!='if' or not re.fullmatch(r'b\d+',w[2]):raise StrataError('unsupported native push condition')
                r['bit']=int(w[2][1:])
            elif len(w)>=3 and w[-2]=='if' and re.fullmatch(r'b\d+',w[-1]):
                r['condition']=int(w[-1][1:]);w=w[:-2]
            if op in self.quantum:
                r['qubits']=[int(x[1:]) for x in w[1:] if re.fullmatch(r'q\d+',x)]
                if op=='HMR':r['bit']=int(w[-1][1:])
            elif op in self.bits:r['bit']=int(w[1][1:])
            elif op in {'PUSH_CONDITION','POP_CONDITION'}:pass
            elif op in {'REGISTER','APPEND_TO_REGISTER'}:
                r.update(name=w[1],qubits=[int(x[1:]) for x in w[2:]])
            elif op=='INFO_TAG':r['text']=w[1]
            else:raise StrataError('native operation cannot be represented in raw front end')
            records.append(json.dumps(r,sort_keys=True))
        text='\n'.join(records)
        if str(self.parse(text))!=str(circuit):raise StrataError('operation codec failed lossless round trip')
        return text

    def load(self,path):return self.parse(Path(path).read_text())
    def measured(self,circuit):return Kickmix().measured(circuit)
    def analyze(self,circuit,out,**kwargs):return Kickmix().analyze(circuit,out,**kwargs)
    def lift(self,circuit):return Kickmix().lift(circuit)


def frontend_for(name):
    if name in {'operations','raw','reversible'}:return OperationStream()
    if name=='kickmix':return Kickmix()
    raise StrataError('unknown circuit front end: '+str(name))


def load_circuit(path):
    return (OperationStream() if Path(path).suffix in {'.ops','.rev','.jsonl'} else Kickmix()).load(path)
