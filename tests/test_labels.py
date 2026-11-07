import pandas as pd
import pytest

from src.config import load_config, resolve_path
from src.data.load import load_subset
from src.features.labels import add_test_rul, add_train_rul


@pytest.fixture
def trajectories():
    """Unit 1 runs 5 cycles, unit 2 runs 3 cycles."""
    return pd.DataFrame({"unit": [1] * 5 + [2] * 3, "cycle": [1, 2, 3, 4, 5, 1, 2, 3]})


def test_train_rul_counts_down_to_zero(trajectories):
    out = add_train_rul(trajectories, cap=None)
    assert out["rul"].tolist() == [4, 3, 2, 1, 0, 2, 1, 0]


def test_train_rul_is_capped(trajectories):
    out = add_train_rul(trajectories, cap=2)
    assert out["rul"].tolist() == [2, 2, 2, 1, 0, 2, 1, 0]


def test_input_is_not_modified(trajectories):
    add_train_rul(trajectories, cap=2)
    assert "rul" not in trajectories.columns


def test_test_rul_offsets_from_ground_truth(trajectories):
    rul = pd.DataFrame({"unit": [1, 2], "rul": [10, 0]})
    out = add_test_rul(trajectories, rul, cap=None)
    assert out["rul"].tolist() == [14, 13, 12, 11, 10, 2, 1, 0]


def test_test_rul_is_capped(trajectories):
    rul = pd.DataFrame({"unit": [1, 2], "rul": [10, 0]})
    out = add_test_rul(trajectories, rul, cap=12)
    assert out["rul"].tolist() == [12, 12, 12, 11, 10, 2, 1, 0]


def test_test_rul_requires_every_unit(trajectories):
    rul = pd.DataFrame({"unit": [1], "rul": [10]})
    with pytest.raises(ValueError, match=r"\[2\]"):
        add_test_rul(trajectories, rul, cap=None)


@pytest.mark.parametrize("cap", [0, -5])
def test_non_positive_cap_rejected(trajectories, cap):
    with pytest.raises(ValueError, match="cap must be positive"):
        add_train_rul(trajectories, cap=cap)


REAL_DATA = resolve_path(load_config()["data"]["raw_dir"])


@pytest.mark.skipif(not (REAL_DATA / "train_FD001.txt").exists(), reason="CMAPSS not downloaded")
def test_real_fd001_labels():
    cap = load_config()["preprocessing"]["rul_cap"]
    train, test, rul = load_subset("FD001")

    train = add_train_rul(train, cap=cap)
    test = add_test_rul(test, rul, cap=None)

    assert train["rul"].between(0, cap).all()
    assert (train.groupby("unit")["rul"].min() == 0).all()
    last = test.sort_values("cycle").groupby("unit").tail(1).sort_values("unit")
    assert last["rul"].tolist() == rul["rul"].tolist()
