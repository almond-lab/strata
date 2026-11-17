"""Hash-consed Boolean expression DAGs for native values and effects.

These preserve products of sums without distributing them.  Canonical ANF is
still the equality oracle: DAG shape is evidence of construction, not a claim
that differently shaped expressions differ.
"""


class BooleanDAG:
    def __init__(self):
        self.nodes = []
        self._intern = {}
        self.zero = self._node('constant', value=0)
        self.one = self._node('constant', value=1)

    def _node(self, op, **fields):
        key = (op, tuple(fields.items()))
        if key not in self._intern:
            self._intern[key] = len(self.nodes)
            self.nodes.append(dict(id=len(self.nodes), op=op, **fields))
        return self._intern[key]

    def variable(self, name):
        return self._node('variable', variable=name,
                          namespace='quantum_input' if isinstance(name, int)
                          else 'measurement' if name.startswith('m') else 'classical_input')

    def xor(self, *args):
        terms = set()
        for arg in args:
            children = self.nodes[arg].get('arguments', (arg,)) if self.nodes[arg]['op'] == 'xor' else (arg,)
            for child in children:
                if child == self.zero:
                    continue
                if child in terms:
                    terms.remove(child)
                else:
                    terms.add(child)
        if not terms:
            return self.zero
        if len(terms) == 1:
            return next(iter(terms))
        return self._node('xor', arguments=tuple(sorted(terms)))

    def product(self, *args):
        terms = set()
        for arg in args:
            if arg == self.zero:
                return self.zero
            children = self.nodes[arg].get('arguments', (arg,)) if self.nodes[arg]['op'] == 'and' else (arg,)
            terms.update(child for child in children if child != self.one)
        if not terms:
            return self.one
        if len(terms) == 1:
            return next(iter(terms))
        return self._node('and', arguments=tuple(sorted(terms)))

    def summary(self, outputs, classical_outputs, phase):
        return dict(representation='hash_consed_boolean_dag', nodes=self.nodes,
                    outputs=outputs, classical_outputs=classical_outputs,
                    phase=phase, node_count=len(self.nodes),
                    equality='canonical ANF; DAG syntax is not an inequality proof')
