"""Exact Boolean tests on a supplied finite table, packed in NumPy big bit order."""
from hashlib import file_digest
from pathlib import Path

import numpy as np

from .floors import _basis
from .model import StrataError


def _packed(value, row_count):
    if type(row_count) is not int or row_count < 1:
        raise StrataError('row_count must be a positive integer')
    a = np.asarray(value)
    if a.dtype != np.uint8 or a.shape != ((row_count + 7) // 8,):
        raise StrataError('each column must contain ceil(row_count/8) uint8 bytes')
    return int.from_bytes(a.tobytes(), 'big') >> (-row_count % 8)


def _matrix(columns, row_count):
    a = np.asarray(columns)
    if a.ndim != 2 or a.dtype != np.uint8 or a.shape[1] != (row_count + 7) // 8:
        raise StrataError('column matrix must be uint8 with one packed row per function')
    return a


def screen_care_table(required_one, required_zero, candidates, *, row_count):
    """Classify each candidate on forced rows. Other rows are unconstrained."""
    one = _packed(required_one, row_count)
    zero = _packed(required_zero, row_count)
    if one & zero:
        raise StrataError('a care-table row requires both output values')
    results = []
    for index, column in enumerate(_matrix(candidates, row_count)):
        value = _packed(column, row_count)
        conflict = (one & ~value) | (zero & value)
        result = {'candidate': index, 'matches': not conflict}
        if conflict:
            row = row_count - conflict.bit_length()
            result.update(counterexample_row=row, expected=(one >> (row_count - row - 1)) & 1)
        results.append(result)
    return {'row_count': row_count, 'required_one_count': one.bit_count(),
            'required_zero_count': zero.bit_count(), 'results': results,
            'scope': 'Supplied care rows only. Reachability, phase, workspace and circuit price are separate obligations.'}


def fit_care_table(required_one, required_zero, features, *, row_count):
    """Fit an affine combination, or exhibit care rows that contradict the entire span."""
    one = _packed(required_one, row_count)
    zero = _packed(required_zero, row_count)
    if one & zero:
        raise StrataError('a care-table row requires both output values')
    care = one | zero
    columns = [care, *(_packed(value, row_count) & care
                       for value in _matrix(features, row_count))]
    pivots = {}

    def reduce_column(value, coefficients):
        while value:
            key = value.bit_length() - 1
            if key not in pivots:
                break
            other, selection = pivots[key]
            value ^= other
            coefficients ^= selection
        return value, coefficients

    for index, column in enumerate(columns):
        residual, selection = reduce_column(column, 1 << index)
        if residual:
            pivots[residual.bit_length() - 1] = residual, selection

    residual, selection = reduce_column(one, 0)
    result = {'row_count': row_count, 'care_rows': care.bit_count(),
              'affine_rank': len(pivots), 'fits': residual == 0,
              'scope': 'Affine combinations of the supplied features on the supplied care rows.'}
    if not residual:
        result['constant'] = selection & 1
        result['feature_indices'] = [j for j in range(len(columns) - 1)
                                     if selection & (2 << j)]
        return result

    # This functional is 1 on the first missing pivot and zero on the span.
    # Solve for its other entries from the smaller pivot positions upwards.
    witness = 1 << (residual.bit_length() - 1)
    for key in sorted(pivots):
        if (pivots[key][0] & witness).bit_count() % 2:
            witness ^= 1 << key
    if any((column & witness).bit_count() % 2 for column in columns):
        raise StrataError('affine exclusion witness failed feature replay')
    if (one & witness).bit_count() % 2 != 1:
        raise StrataError('affine exclusion witness failed target replay')
    rows = []
    while witness:
        pivot = witness.bit_length() - 1
        rows.append(row_count - pivot - 1)
        witness ^= 1 << pivot
    result['contradictory_rows'] = rows
    return result




def affine_repair_floor(available, targets, *, row_count):
    """Minimum extra ANDs required by dimension, allowing all reversible affine edits.

    This is a lower bound on exact straight-line repair. It does not establish
    achievability or an average executed gate price for adaptive programs.
    """
    if type(row_count) is not int or row_count < 1:
        raise StrataError('row_count must be a positive integer')
    sources = _matrix(available, row_count)
    wanted = _matrix(targets, row_count)
    before = _basis([(1 << row_count) - 1, *(_packed(v, row_count) for v in sources)])
    after = _basis([*before, *(_packed(v, row_count) for v in wanted)])
    return {'row_count': row_count, 'available_rank_including_constant': len(before),
            'augmented_rank': len(after), 'additional_and_lower_bound': len(after) - len(before),
            'scope': 'Fixed supplied signals and addresses. Each extra binary AND introduces at most one independent Boolean function; affine operations are free.'}


def run_table_screen(path, mode):
    """Run a portable packed NPZ fixture. Hash the same open input that was read."""
    with Path(path).open('rb') as source:
        fingerprint = file_digest(source, 'sha256').hexdigest()
        source.seek(0)
        with np.load(source, allow_pickle=False) as data:
            n = data['row_count']
            if n.shape != () or n.dtype.kind not in 'ui':
                raise StrataError('row_count must be an integer scalar')
            if mode == 'care':
                result = screen_care_table(data['required_one'], data['required_zero'],
                                           data['candidates'], row_count=int(n))
            elif mode == 'fit':
                result = fit_care_table(data['required_one'], data['required_zero'],
                                        data['features'], row_count=int(n))
            elif mode == 'repair':
                result = affine_repair_floor(data['available'], data['targets'], row_count=int(n))
            else:
                raise StrataError('unknown table screen mode')
    return {'kind': 'finite_table_screen', 'mode': mode, 'input_sha256': fingerprint, **result}

