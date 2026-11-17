"""Content-addressed file APIs stay bounded and never expose partial copies."""
import hashlib
import os
from pathlib import Path
import tracemalloc

import pytest
from strata.model import StrataError
from strata.store import Store


def test_large_file_roundtrip_uses_bounded_memory(tmp_path, monkeypatch):
    source = tmp_path / 'large.circuit'
    block = b'a circuit row with numeric data\n' * 32768
    expected = hashlib.sha256()
    with source.open('wb') as output:
        for _ in range(8):
            output.write(block)
            expected.update(block)
    store = Store(tmp_path / 'store')
    restored = tmp_path / 'restored' / 'large.circuit'
    monkeypatch.setattr(Path, 'read_bytes', lambda *a, **k: pytest.fail('streaming API read the whole file'))
    tracemalloc.start()
    try:
        handle = store.blob_file(source)
        assert handle == expected.hexdigest()
        assert store.verify_blob(handle) is True
        assert store.copy_blob(handle, restored) == restored
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 4 * 1024**2  # The source is over 7 MiB; one block is 1 MiB.
    assert restored.stat().st_size == source.stat().st_size
    actual = hashlib.sha256()
    with restored.open('rb') as output:
        for chunk in iter(lambda: output.read(65536), b''):
            actual.update(chunk)
    assert actual.hexdigest() == handle
    # Avoid retaining several large fixture copies across unrelated test runs.
    source.unlink()
    restored.unlink()
    (store.path / 'objects' / handle).unlink()


def test_empty_and_existing_blob_are_compatible_with_bytes_api(tmp_path):
    store = Store(tmp_path / 'store')
    source = tmp_path / 'empty'
    source.touch()
    handle = store.blob_file(source)
    assert handle == store.blob(b'')
    assert store.blob_file(source) == handle
    assert store.read_blob(handle) == b''
    target = tmp_path / 'target'
    target.write_bytes(b'previous data')
    store.copy_blob(handle, target)
    assert target.read_bytes() == b''
    assert len(list((store.path / 'objects').iterdir())) == 1


def test_corrupt_blob_never_replaces_destination(tmp_path):
    store = Store(tmp_path / 'store')
    handle = store.blob(b'checked artifact')
    (store.path / 'objects' / handle).write_bytes(b'corrupted')
    target = tmp_path / 'restore' / 'candidate'
    target.parent.mkdir()
    target.write_bytes(b'retained safe candidate')
    with pytest.raises(StrataError, match='integrity'):
        store.verify_blob(handle)
    with pytest.raises(StrataError, match='integrity'):
        store.copy_blob(handle, target)
    assert target.read_bytes() == b'retained safe candidate'
    assert list(target.parent.iterdir()) == [target]
    source = tmp_path / 'original'
    source.write_bytes(b'checked artifact')
    with pytest.raises(StrataError, match='integrity'):
        store.blob_file(source)
    assert [p.name for p in (store.path / 'objects').iterdir()] == [handle]


@pytest.mark.parametrize('handle', ['', '../outside', 'A' * 64, '0' * 63, 1, None])
def test_digest_names_are_validated_before_restore(tmp_path, handle):
    store = Store(tmp_path / 'store')
    target = tmp_path / 'not-created'
    with pytest.raises(StrataError, match='invalid object hash'):
        store.verify_blob(handle)
    with pytest.raises(StrataError, match='invalid object hash'):
        store.copy_blob(handle, target)
    assert not target.exists()


def test_fsync_failure_does_not_publish_or_leave_partial_files(tmp_path, monkeypatch):
    store = Store(tmp_path / 'store')
    handle = store.blob(b'complete')
    source = tmp_path / 'source'
    source.write_bytes(b'new source')
    target = tmp_path / 'target'
    target.write_bytes(b'previous safe candidate')

    def disk_failure(_):
        raise OSError('simulated fsync failure')

    monkeypatch.setattr(os, 'fsync', disk_failure)
    with pytest.raises(OSError, match='fsync'):
        store.copy_blob(handle, target)
    with pytest.raises(OSError, match='fsync'):
        store.blob_file(source)
    assert target.read_bytes() == b'previous safe candidate'
    assert sorted(p.name for p in (store.path / 'objects').iterdir()) == [handle]
    assert sorted(p.name for p in tmp_path.iterdir()) == ['source', 'store', 'target']


def test_special_sources_are_rejected_without_blocking(tmp_path):
    store = Store(tmp_path / 'store')
    fifo = tmp_path / 'pipe'
    os.mkfifo(fifo)
    with pytest.raises(StrataError, match='regular file'):
        store.blob_file(fifo)
    with pytest.raises(StrataError, match='regular file'):
        store.blob_file(tmp_path)
    assert not list((store.path / 'objects').iterdir())


def test_source_change_during_read_is_not_published(tmp_path, monkeypatch):
    store = Store(tmp_path / 'store')
    source = tmp_path / 'moving-source'
    source.write_bytes(b'a' * (2 * 1024**2))
    real_sha256 = hashlib.sha256

    class ChangingHash:
        def __init__(self):
            self.value = real_sha256()
            self.changed = False

        def update(self, data):
            self.value.update(data)
            if not self.changed:
                self.changed = True
                with source.open('ab') as stream:
                    stream.write(b'changed during copy')

        def hexdigest(self):
            return self.value.hexdigest()

    monkeypatch.setattr(hashlib, 'sha256', ChangingHash)
    with pytest.raises(StrataError, match='changed while'):
        store.blob_file(source)
    assert not list((store.path / 'objects').iterdir())


def test_concurrent_byte_and_file_publishers_never_replace_object(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import time

    data = b'concurrent immutable artifact\n' * 32768
    handle = hashlib.sha256(data).hexdigest()
    source = tmp_path / 'source'
    source.write_bytes(data)
    stores = [Store(tmp_path / 'store') for _ in range(12)]
    target = stores[0].path / 'objects' / handle
    start = threading.Barrier(len(stores) + 1)
    published = threading.Event()
    finished = threading.Event()
    publications = []
    replacements_lock = threading.Lock()
    real_replace = os.replace

    def slow_publish(source_path, destination):
        assert Path(destination) == target
        with replacements_lock:
            publications.append(str(source_path))
        # Widen the missing-target race that used to replace a reader's inode.
        time.sleep(.025)
        real_replace(source_path, destination)
        published.set()

    monkeypatch.setattr(os, 'replace', slow_publish)

    def writer(index):
        start.wait(timeout=10)
        store = stores[index]
        return store.blob(data) if index % 2 else store.blob_file(source)

    def observer():
        assert published.wait(timeout=10)
        identity = target.stat()
        checks = 0
        while not finished.is_set() or checks < 4:
            assert stores[0].verify_blob(handle)
            current = target.stat()
            assert (current.st_ino, current.st_ctime_ns) == (identity.st_ino, identity.st_ctime_ns)
            checks += 1
        return checks

    try:
        with ThreadPoolExecutor(max_workers=len(stores) + 1) as pool:
            reader = pool.submit(observer)
            writers = [pool.submit(writer, index) for index in range(len(stores))]
            start.wait(timeout=10)
            try:
                assert [future.result(timeout=20) for future in writers] == [handle] * len(stores)
            finally:
                finished.set()
            assert reader.result(timeout=10) >= 4
        assert len(publications) == 1
        assert stores[0].read_blob(handle) == data
    finally:
        finished.set()
        for store in stores:
            store.db.close()
