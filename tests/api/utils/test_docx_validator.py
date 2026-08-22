import io
import zipfile

import pytest
from fixtures.docx_builder import simple_docx
from src.api.exceptions import ValidationError
from src.api.utils.docx_validator import DOCXValidator


def zip_with(names: dict[str, bytes | str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, payload in names.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


class TestValidateDocxBytes:
    def test_valid_docx_returns_metadata(self):
        content = simple_docx()
        returned, metadata = DOCXValidator.validate_docx_bytes(content, "report.docx")

        assert returned == content
        assert metadata["filename"] == "report.docx"
        assert metadata["size_bytes"] == len(content)
        assert len(metadata["checksum"]) == 64
        assert metadata["page_count"] is None
        assert "wordprocessingml" in metadata["content_type"]

    def test_checksum_is_stable(self):
        content = simple_docx()
        first = DOCXValidator.validate_docx_bytes(content, "a.docx")[1]["checksum"]
        second = DOCXValidator.validate_docx_bytes(content, "b.docx")[1]["checksum"]
        assert first == second

    def test_not_a_zip_rejected(self):
        with pytest.raises(ValidationError, match="Invalid or corrupted Word document"):
            DOCXValidator.validate_docx_bytes(b"plain text, not a docx", "bad.docx")

    def test_password_protected_rejected(self):
        content = zip_with({"EncryptedPackage": b"\x00", "EncryptionInfo": b"\x00"})
        with pytest.raises(ValidationError, match="Password-protected"):
            DOCXValidator.validate_docx_bytes(content, "locked.docx")

    def test_legacy_doc_rejected(self):
        content = zip_with({"word/document.bin": b"\x00"})
        with pytest.raises(ValidationError, match=r"\(\.doc\) are not supported"):
            DOCXValidator.validate_docx_bytes(content, "legacy.docx")

    def test_missing_document_part_rejected(self):
        content = zip_with({"word/styles.xml": "<styles/>"})
        with pytest.raises(ValidationError, match="Not a Word document"):
            DOCXValidator.validate_docx_bytes(content, "empty.docx")

    def test_oversized_file_rejected(self, monkeypatch):
        monkeypatch.setattr(
            "src.api.utils.docx_validator.settings.MAX_FILE_SIZE", 10, raising=False
        )
        with pytest.raises(ValidationError, match="File size exceeds"):
            DOCXValidator.validate_docx_bytes(simple_docx(), "big.docx")
