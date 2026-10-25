import io
import zipfile

from src.data.download import EXPECTED_FILES, extract_cmapss, verify


def _zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_extract_handles_nested_zip(tmp_path):
    inner = _zip({name: b"1 1\n" for name in EXPECTED_FILES} | {"paper.pdf": b"%PDF"})
    outer = _zip({"Turbofan Data Set/": b"", "Turbofan Data Set/CMAPSSData.zip": inner})

    extract_cmapss(outer, tmp_path)

    assert verify(tmp_path) == []
    assert not (tmp_path / "paper.pdf").exists()


def test_verify_reports_missing_files(tmp_path):
    (tmp_path / "train_FD001.txt").write_text("1 1\n")
    missing = verify(tmp_path)
    assert "train_FD001.txt" not in missing
    assert "RUL_FD004.txt" in missing
