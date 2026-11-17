"""Disk-backed canonical tapes for bounded-memory candidate alignment."""
from array import array
from contextlib import contextmanager
from itertools import islice
from pathlib import Path
import hashlib
import tempfile


class CircuitTape:
    """A line sequence with sparse offsets and one bounded cached block.

    The native parser still owns the circuit. Python stores eight bytes per
    1024 lines plus one block, rather than a Python string object for every gate.
    Slices are intended only for budgeted proof windows; spans stream directly.
    """
    block_lines = 1024

    def __init__(self, path):
        self.path = Path(path)
        self.offsets = array('Q')
        self.count = 0
        self._cache_at, self._cache = None, []
        hasher = hashlib.sha256()
        with self.path.open('rb') as source:
            position = 0
            for line in source:
                if self.count % self.block_lines == 0:
                    self.offsets.append(position)
                hasher.update(line)
                position += len(line)
                self.count += 1
        self.sha256 = hasher.hexdigest()

    def __len__(self):
        return self.count

    def iter_span(self, start=0, stop=None):
        stop = self.count if stop is None else stop
        if start >= stop:
            return
        block = start // self.block_lines
        with self.path.open('rb') as source:
            source.seek(self.offsets[block])
            for raw in islice(source, start % self.block_lines, stop - block*self.block_lines):
                yield raw.decode().rstrip('\r\n')

    def __iter__(self):
        return self.iter_span()

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(self.count)
            if step != 1:
                return [self[i] for i in range(start, stop, step)]
            return list(self.iter_span(start, stop))
        key = key if key >= 0 else key + self.count
        if not 0 <= key < self.count:
            raise IndexError(key)
        block = key // self.block_lines
        if block != self._cache_at:
            self._cache = list(self.iter_span(block*self.block_lines,
                                             min(self.count, (block+1)*self.block_lines)))
            self._cache_at = block
        return self._cache[key % self.block_lines]


@contextmanager
def canonical_tape(circuit):
    with tempfile.TemporaryDirectory(prefix='strata-tape-') as directory:
        path = Path(directory) / 'canonical.kmx'
        circuit.to_file(path, format='kmx')
        # SDK files end in a newline; str(circuit), existing parent bindings,
        # and stored canonical blobs omit it. Preserve that exact identity.
        with path.open('r+b') as output:
            output.seek(0, 2)
            size = output.tell()
            if size:
                output.seek(-1, 2)
                if output.read(1) == b'\n':
                    output.truncate(size - 1)
        yield CircuitTape(path)


def same_span(a, start, stop, b, other_start, other_stop):
    if stop-start != other_stop-other_start:
        return False
    left = a.iter_span(start, stop) if isinstance(a, CircuitTape) else islice(a, start, stop)
    right = b.iter_span(other_start, other_stop) if isinstance(b, CircuitTape) else islice(b, other_start, other_stop)
    return all(x == y for x, y in zip(left, right))
