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


class DecisionBudget(Exception):
    pass


class DecisionDAG:
    """Reduced ordered Boolean decisions, canonical for a fixed variable order.

    Unlike expression syntax, equal roots prove equal Boolean functions. Both
    allocated nodes and uncached binary operations have explicit global limits.
    """
    def __init__(self, variables, *, max_nodes=262144, max_steps=1048576):
        self.variables = list(variables)
        self.order = {v: i for i, v in enumerate(self.variables)}
        self.nodes = [None, None]
        self.unique = {}
        self.cache = {}
        self.free = []
        self.pins = set()
        self.max_nodes, self.max_steps, self.steps = max_nodes, max_steps, 0

    def node(self, level, low, high):
        if low == high:
            return low
        key = (level, low, high)
        if key not in self.unique:
            if not self.free and len(self.nodes) >= self.max_nodes:
                raise DecisionBudget('decision_node_budget')
            if self.free:
                index = self.free.pop()
                self.nodes[index] = key
            else:
                index = len(self.nodes)
                self.nodes.append(key)
            self.unique[key] = index
        return self.unique[key]

    def variable(self, name):
        return self.node(self.order[name], 0, 1)

    def apply(self, op, a, b):
        if a > b:
            a, b = b, a
        if op == 'xor':
            if a == 0: return b
            if a == b: return 0
        else:
            if a == 0: return 0
            if a == 1: return b
            if a == b: return a
        key = (op, a, b)
        if key in self.cache:
            return self.cache[key]
        self.steps += 1
        if self.steps > self.max_steps:
            raise DecisionBudget('decision_apply_budget')
        level = min(self.nodes[n][0] for n in (a, b) if n > 1)
        al, ah = self.nodes[a][1:] if a > 1 and self.nodes[a][0] == level else (a, a)
        bl, bh = self.nodes[b][1:] if b > 1 and self.nodes[b][0] == level else (b, b)
        result = self.node(level, self.apply(op, al, bl), self.apply(op, ah, bh))
        self.cache[key] = result
        return result

    def xor(self, a, b):
        return self.apply('xor', a, b)

    def product(self, a, b):
        return self.apply('and', a, b)

    def support(self, root):
        seen, pending, result = set(), [root], set()
        while pending:
            n = pending.pop()
            if n < 2 or n in seen: continue
            seen.add(n)
            level, low, high = self.nodes[n]
            result.add(self.variables[level]); pending.extend((low, high))
        return result

    def collect(self, roots):
        """Discard unreachable intermediates; live root IDs remain stable."""
        live, pending = {0,1}, list(roots) + list(self.pins)
        while pending:
            root = pending.pop()
            if root in live: continue
            live.add(root); pending.extend(self.nodes[root][1:])
        self.cache.clear()
        for root, node in enumerate(self.nodes):
            if root > 1 and node is not None and root not in live:
                del self.unique[node]; self.nodes[root] = None; self.free.append(root)

    def evaluate(self, root, assignment):
        while root > 1:
            level, low, high = self.nodes[root]
            root = high if assignment.get(self.variables[level], 0) else low
        return root

    def witness(self, root):
        assignment = {v: 0 for v in self.variables}
        while root > 1:
            level, low, high = self.nodes[root]
            if low: root = low
            else:
                assignment[self.variables[level]] = 1
                root = high
        return assignment

    def export(self, roots):
        """Reachable nodes only, deterministically numbered in child-first order."""
        mapping, rows = {0: 0, 1: 1}, []
        def visit(root):
            if root not in mapping:
                level, low, high = self.nodes[root]
                low, high = visit(low), visit(high)
                mapping[root] = len(rows) + 2
                rows.append([self.variables[level], low, high])
            return mapping[root]
        output = {name: visit(root) for name, root in roots.items()}
        return dict(representation='reduced_ordered_boolean_decisions',
                    variable_order=self.variables, terminals=[0, 1], nodes=rows,
                    roots=output, equality='canonical reduced ordered decisions',
                    allocated_nodes=len(self.nodes), apply_steps=self.steps)
