import pytest

from src.config import load_config, resolve_path
from src.data.load import COLUMNS, load_rul, load_split, load_subset


def _row(unit: int, cycle: int) -> str:
    values = [unit, cycle] + [0.5] * (len(COLUMNS) - 2)
    return " ".join(str(v) for v in values) + "  \n"  # CMAPSS rows end with trailing spaces


@pytest.fixture
def raw_dir(tmp_path):
    """Two train units, two test units, in the CMAPSS text format."""
    (tmp_path / "train_FD001.txt").write_text(
        "".join(_row(u, c) for u in (2, 1) for c in (1, 2, 3))
    )
    (tmp_path / "test_FD001.txt").write_text("".join(_row(u, c) for u in (1, 2) for c in (1, 2)))
    (tmp_path / "RUL_FD001.txt").write_text("112\n98\n")
    return tmp_path


def test_load_split_columns_and_order(raw_dir):
    df = load_split("FD001", "train", raw_dir)
    assert list(df.columns) == COLUMNS
    assert len(df) == 6
    assert df["unit"].dtype == int
    assert df[["unit", "cycle"]].values.tolist()[:3] == [[1, 1], [1, 2], [1, 3]]


def test_load_rul_assigns_unit_ids(raw_dir):
    rul = load_rul("FD001", raw_dir)
    assert rul.to_dict("list") == {"unit": [1, 2], "rul": [112, 98]}


def test_load_subset_rejects_rul_mismatch(raw_dir):
    (raw_dir / "RUL_FD001.txt").write_text("112\n")
    with pytest.raises(ValueError, match="RUL values"):
        load_subset("FD001", raw_dir)


def test_load_split_rejects_wrong_column_count(raw_dir):
    (raw_dir / "train_FD001.txt").write_text("1 1 0.1 0.2\n")
    with pytest.raises(ValueError, match="expected 26 columns"):
        load_split("FD001", "train", raw_dir)


@pytest.mark.parametrize("subset, split", [("FD005", "train"), ("FD001", "valid")])
def test_load_split_rejects_bad_arguments(raw_dir, subset, split):
    with pytest.raises(ValueError):
        load_split(subset, split, raw_dir)


REAL_DATA = resolve_path(load_config()["data"]["raw_dir"])


@pytest.mark.skipif(not (REAL_DATA / "train_FD001.txt").exists(), reason="CMAPSS not downloaded")
@pytest.mark.parametrize(
    "subset, train_rows, test_rows, train_units, test_units",
    [("FD001", 20631, 13096, 100, 100), ("FD002", 53759, 33991, 260, 259)],
)
def test_real_cmapss_shapes(subset, train_rows, test_rows, train_units, test_units):
    train, test, rul = load_subset(subset)
    assert (len(train), len(test)) == (train_rows, test_rows)
    assert (train["unit"].nunique(), test["unit"].nunique()) == (train_units, test_units)
    assert len(rul) == test_units
