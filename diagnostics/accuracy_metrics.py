"""Numerical evidence, without inventing an acceptance rule for near-zero values."""
import torch


def accuracy_metrics(actual, reference):
    if actual.shape != reference.shape:
        raise ValueError("actual and reference shapes differ")
    if not reference.is_floating_point():
        return {"elements": reference.numel(), "exact": torch.equal(actual, reference)}
    actual, reference = actual.float(), reference.float()
    finite = torch.isfinite(actual) & torch.isfinite(reference)
    matching_special = (torch.isnan(actual) & torch.isnan(reference)) | (actual == reference)
    delta = (actual[finite] - reference[finite]).abs()
    finite_ref = reference[finite]
    nonzero = finite_ref != 0
    relative = delta[nonzero] / finite_ref[nonzero].abs()
    reference_norm = finite_ref.norm().item()
    return {
        "elements": reference.numel(),
        "nonfinite_mismatches": ((~finite) & (~matching_special)).sum().item(),
        "max_absolute_error": delta.max().item() if delta.numel() else None,
        "relative_l2": delta.norm().item() / reference_norm if reference_norm else None,
        "nonzero_reference_elements": nonzero.sum().item(),
        "nonzero_relative_error_gt_1pct": (relative > .01).sum().item(),
        "max_nonzero_relative_error": relative.max().item() if relative.numel() else None,
        "zero_reference_elements": (~nonzero).sum().item(),
        "zero_reference_nonzero_outputs": (delta[~nonzero] != 0).sum().item(),
        "note": "Strict elementwise nonzero relative error is reported, not an agreed contract acceptance metric; zeros and nonfinite values are separate.",
    }
