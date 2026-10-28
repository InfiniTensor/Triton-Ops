# -*- coding: utf-8 -*-
"""NineToothed operator implementations.

Compare with triton_impl.py: for the same vector addition, NineToothed needs no
program_id / arange / mask —— tiling (arrangement) and computation (application)
are written separately, and the compiler generates all the parallel details.

Note: import torch must happen before ninetoothed is used
(Triton's device query has an encoding bug on Chinese Windows; loading torch first works around it).
"""
import torch  # noqa: F401  order-sensitive, do not remove
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


# ---------------------------------------------------------------- add
def _add_arrangement(x, y, z, BLOCK_SIZE=ninetoothed.block_size()):
    return x.tile((BLOCK_SIZE,)), y.tile((BLOCK_SIZE,)), z.tile((BLOCK_SIZE,))


def _add_application(x, y, z):
    z = x + y  # noqa: F841


_add_kernel = ninetoothed.make(
    _add_arrangement, _add_application, (Tensor(1), Tensor(1), Tensor(1))
)


def add(x, y, z):
    _add_kernel(x, y, z)
    return z


# ---------------------------------------------------------------- abs
# Unary op: y = |x|. Math functions go through ninetoothed.language (ntl.xxx),
# consistent with the official operator library ntops (see ntops/kernels/abs.py).
def _abs_arrangement(x, y, BLOCK_SIZE=ninetoothed.block_size()):
    return x.tile((BLOCK_SIZE,)), y.tile((BLOCK_SIZE,))


def _abs_application(x, y):
    y = ntl.abs(x)  # noqa: F841


_abs_kernel = ninetoothed.make(
    _abs_arrangement, _abs_application, (Tensor(1), Tensor(1))
)


def abs(x, y):
    _abs_kernel(x, y)
    return y
