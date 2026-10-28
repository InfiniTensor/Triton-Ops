"""Explicit reference precision for comparison with input_precision='ieee'."""
import os

import torch


def configure_reference(device=None, mode=None):
    device = device or os.environ.get("TB_DEVICE", "cuda")
    mode = mode or os.environ.get("TB_REFERENCE_PRECISION", "ieee")
    if mode not in ("ieee", "vendor-default"):
        raise ValueError("TB_REFERENCE_PRECISION must be ieee or vendor-default")
    if device == "mlu":
        import torch_mlu  # registers the backend
        matmul, convolution = torch.backends.mlu.matmul, torch.backends.cnnl
    else:
        matmul, convolution = torch.backends.cuda.matmul, torch.backends.cudnn
    before = {"matmul_allow_tf32": matmul.allow_tf32, "conv_allow_tf32": convolution.allow_tf32}
    if mode == "ieee":
        matmul.allow_tf32 = False
        convolution.allow_tf32 = False
    return {"policy": mode, "before": before,
            "after": {"matmul_allow_tf32": matmul.allow_tf32, "conv_allow_tf32": convolution.allow_tf32}}
