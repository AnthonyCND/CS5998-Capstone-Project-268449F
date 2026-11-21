import math

import pytest

from src.evaluation.metrics import nasa_score, rmse


def test_rmse():
    assert rmse([10, 20, 30], [13, 16, 30]) == pytest.approx(math.sqrt((9 + 16) / 3))


def test_perfect_predictions_score_zero():
    assert rmse([5, 50], [5, 50]) == 0.0
    assert nasa_score([5, 50], [5, 50]) == 0.0


def test_nasa_score_values():
    assert nasa_score([100], [87]) == pytest.approx(math.e - 1)  # 13 cycles early
    assert nasa_score([100], [110]) == pytest.approx(math.e - 1)  # 10 cycles late


def test_late_costs_more_than_early():
    assert nasa_score([100], [120]) > nasa_score([100], [80])


def test_nasa_score_sums_over_engines():
    assert nasa_score([100, 100], [110, 110]) == pytest.approx(2 * nasa_score([100], [110]))


@pytest.mark.parametrize("y_true, y_pred", [([1, 2], [1]), ([], []), ([[1]], [[1]])])
def test_mismatched_inputs_are_rejected(y_true, y_pred):
    with pytest.raises(ValueError):
        rmse(y_true, y_pred)
