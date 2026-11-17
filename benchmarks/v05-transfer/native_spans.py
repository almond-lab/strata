"""Read bounded KMB v2 spans through the SDK without decoding a full KMX tape."""
import mmap
import struct

import kickmix as km
import numpy as np

MAGIC = bytes.fromhex('d750c7d5c329d326e3cc9f6834f2b8bf')


class NativeSpans:
    def __init__(self, file):
        self.data = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
        if self.data[:16] != MAGIC:
            raise ValueError('Expected native Kickmix binary file')
        version, self.width, self.bits, registers = struct.unpack_from('<4I', self.data, 16)
        if version != 2:
            raise ValueError('Expected KMB version 2')
        counts = struct.unpack_from('<9Q', self.data, 32)
        self.ops = counts[0]
        self.packets = {}
        self.register_qubits = set()
        self.register_names = []
        self.register_qubit_order = {}
        pos = 104
        while pos < len(self.data):
            kind, size = struct.unpack_from('<2Q', self.data, pos)
            pos += 16
            if pos + size > len(self.data):
                raise ValueError('Truncated native packet')
            if kind == 0:
                namesize, wires = struct.unpack_from('<2I', self.data, pos)
                name = self.data[pos+8:pos+8+namesize].decode()
                self.register_names.append(name)
                ids = struct.unpack_from(f'<{wires}I', self.data, pos+8+namesize)
                self.register_qubits.update(i & 0x7fffffff for i in ids if i >> 31)
                if all(i >> 31 for i in ids):
                    self.register_qubit_order[name] = [i & 0x7fffffff for i in ids]
            else:
                self.packets[kind] = (pos, size)
            pos += size
        if len(self.register_names) != registers or self.packets[1][1] != self.ops:
            raise ValueError('Invalid KMB topology')
        self.checkpoints = {0: np.zeros(256, dtype=np.int64)}

    def close(self):
        self.data.close()

    def histogram(self, stop):
        start = max(i for i in self.checkpoints if i <= stop)
        counts = self.checkpoints[start].copy()
        for at in range(start, stop, 262144):
            types = np.frombuffer(self.data, dtype=np.uint8, count=min(stop-at, 262144),
                                  offset=self.packets[1][0]+at)
            counts += np.bincount(types, minlength=256)
        self.checkpoints[stop] = counts.copy()
        return counts

    def depth(self, at):
        h = self.histogram(at)
        return int(h[0x81] - h[0x01])

    @staticmethod
    def cursors(counts):
        op = np.arange(256)
        q = op & 0x30
        return [int(counts[mask].sum()) for mask in
                (q == 0x30, q == 0x20, q == 0x10, (op & 0x40) != 0,
                 (op & 0x80) != 0, (op & 0x7f) == 0x13)]

    def span(self, start, stop):
        if not 0 <= start < stop <= self.ops or stop-start > 65536:
            raise ValueError('Expected bounded operation coordinates')
        previous = self.histogram(start)
        counts = self.histogram(stop) - previous
        if counts[6]:
            raise ValueError('Tagged window unsupported')
        starts = self.cursors(previous)
        lengths = self.cursors(counts)
        types, _ = self.packets[1]
        chunks = {1: self.data[types+start:types+stop]}
        nq = nb = 0
        for kind, which in enumerate((0, 0, 0, 1, 1, 2, 3, 4, 5), 2):
            itemsize = 16 if kind == 10 else 4
            begin, size = self.packets[kind]
            lo = starts[which]*itemsize
            hi = lo + lengths[which]*itemsize
            if hi > size:
                raise ValueError('Operand cursor outside packet')
            chunk = self.data[begin+lo:begin+hi]
            chunks[kind] = chunk
            if chunk and kind < 10:
                used = 1 + int(np.frombuffer(chunk, dtype='<u4').max())
                if kind <= 7:
                    nq = max(nq, used)
                else:
                    nb = max(nb, used)
        chunks[11] = bytes(8)
        header = MAGIC + struct.pack('<4I9Q', 2, nq, nb, 0, stop-start, *lengths, 0, 0)
        raw = header + b''.join(struct.pack('<2Q', kind, len(data))+data for kind, data in chunks.items())
        # Kickmix validates operand counts, widths and native encoding and
        # performs all opcode/argument decoding of the extracted circuit.
        import tempfile
        with tempfile.TemporaryFile() as compact:
            compact.write(raw)
            compact.seek(0)
            return km.Circuit.from_file(compact, format='kmb')
