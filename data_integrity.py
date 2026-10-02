"""Stable training-data fingerprints across Windows and Linux checkouts."""
import hashlib
from pathlib import Path


def data_digest(path):
    data = Path(path).read_bytes().replace(b'\r\n', b'\n')
    return hashlib.sha256(data).hexdigest()


def matches_training_data(path, expected_digest):
    data = Path(path).read_bytes().replace(b'\r\n', b'\n')
    if hashlib.sha256(data).hexdigest() == expected_digest:
        return True
    # Existing model packages fingerprinted the Windows CSV bytes before Git
    # normalized line endings. Accept that representation of identical data.
    return hashlib.sha256(data.replace(b'\n', b'\r\n')).hexdigest() == expected_digest
