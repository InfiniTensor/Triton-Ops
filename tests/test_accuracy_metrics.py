import pytest
import torch

from diagnostics.accuracy_metrics import accuracy_metrics


def test_legacy_allclose_can_hide_100pct_error():
    actual, reference = torch.tensor([.02]), torch.tensor([.01])
    assert torch.allclose(actual, reference, atol=.01, rtol=.01)
    metrics = accuracy_metrics(actual, reference)
    assert metrics["nonzero_relative_error_gt_1pct"] == 1
    assert metrics["max_nonzero_relative_error"] == pytest.approx(1)


def test_zero_reference_and_nonfinite_values_are_not_hidden():
    actual = torch.tensor([.001, 0, float('nan'), float('inf'), 1])
    reference = torch.tensor([0, 0, float('nan'), float('-inf'), 1])
    metrics = accuracy_metrics(actual, reference)
    assert metrics["zero_reference_nonzero_outputs"] == 1
    assert metrics["nonfinite_mismatches"] == 1
    assert metrics["nonzero_relative_error_gt_1pct"] == 0


def test_all_zero_inputs_have_no_relative_denominator():
    metrics = accuracy_metrics(torch.zeros(2), torch.zeros(2))
    assert metrics["relative_l2"] is None
    assert metrics["max_nonzero_relative_error"] is None
