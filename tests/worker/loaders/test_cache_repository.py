import hashlib

from src.worker.loaders.repositories.cache_repository import get_file_hash
from src.worker.loaders.repositories.cache_repository import get_file_size
from src.worker.loaders.repositories.cache_repository import verify_file_integrity
from src.worker.loaders.repositories.cache_repository import verify_or_delete


class TestCacheRepository:
    def test_get_file_hash(self, tmp_path):
        p = tmp_path / "test.txt"
        content = b"hello world"
        p.write_bytes(content)

        expected = hashlib.sha3_256(content).hexdigest()
        assert get_file_hash(p) == expected

    def test_verify_file_integrity_success(self, tmp_path):
        p = tmp_path / "test.txt"
        content = b"hello world"
        p.write_bytes(content)
        expected = hashlib.sha3_256(content).hexdigest()

        assert verify_file_integrity(p, expected) is True

    def test_verify_file_integrity_failure(self, tmp_path):
        p = tmp_path / "test.txt"
        p.write_bytes(b"bad")
        assert verify_file_integrity(p, "wrong") is False

    def test_verify_or_delete_success(self, tmp_path):
        p = tmp_path / "test.txt"
        content = b"ok"
        p.write_bytes(content)
        expected = hashlib.sha3_256(content).hexdigest()

        assert verify_or_delete(p, expected) is True
        assert p.exists()

    def test_verify_or_delete_failure(self, tmp_path):
        p = tmp_path / "test.txt"
        p.write_bytes(b"corrupt")

        assert verify_or_delete(p, "wrong") is False
        assert not p.exists()

    def test_get_file_size(self, tmp_path):
        p = tmp_path / "test.txt"
        p.write_bytes(b"12345")
        assert get_file_size(p) == 5
        assert get_file_size(tmp_path / "none") == 0
