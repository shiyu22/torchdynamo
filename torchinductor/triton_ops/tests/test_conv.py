import itertools
import torch
import triton
import pytest

import torchinductor.triton_ops

# The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
torch.backends.cuda.matmul.allow_tf32 = False
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = False
from torchdynamo.testing import same


@pytest.mark.parametrize(
    "BATCH, IN_C, IN_H, IN_W, KERNEL_N, KERNEL_H, KERNEL_W, stride, padding, dilation, groups, dtype,\
        BLOCK_M, BLOCK_N, BLOCK_K, NSTAGE, NWARP, layout",
    itertools.chain(
        *[
            [
                # 1 warp
                (16, 16, 8, 8, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 16, 16, BLOCK_K, NSTAGE, 1, layout),
                (16, 16, 8, 8, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 32, 16, BLOCK_K, NSTAGE, 1, layout),
                (16, 32, 8, 8, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 16, 16, BLOCK_K, NSTAGE, 1, layout),
                (16, 64, 8, 8, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 32, 16, BLOCK_K, NSTAGE, 1, layout),
                (16, 64, 8, 8, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 64, 16, BLOCK_K, NSTAGE, 1, layout),
                (16, 16, 16, 16, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 32, 16, BLOCK_K, NSTAGE, 1, layout),
                (16, 16, 16, 16, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 64, 16, BLOCK_K, NSTAGE, 1, layout),
                (16, 32, 16, 16, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 32, 16, BLOCK_K, NSTAGE, 1, layout),
                (16, 64, 16, 16, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 64, 16, BLOCK_K, NSTAGE, 1, layout),
                (16, 64, 16, 16, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 128, 16, BLOCK_K, NSTAGE, 1, layout),

                # 2 wrap
                (16, 128, 8, 8, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 32, 16, BLOCK_K, NSTAGE, 2, layout),
                (16, 128, 8, 8, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 64, 16, BLOCK_K, NSTAGE, 2, layout),
                (16, 256, 8, 8, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 64, 16, BLOCK_K, NSTAGE, 2, layout),
                (16, 256, 8, 8, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 128, 16, BLOCK_K, NSTAGE, 2, layout),
                (16, 64, 16, 16, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 64, 16, BLOCK_K, NSTAGE, 2, layout),
                (16, 64, 16, 16, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 128, 16, BLOCK_K, NSTAGE, 2, layout),
                (16, 64, 16, 16, 32, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 128, 16, BLOCK_K, NSTAGE, 2, layout),
                (16, 64, 16, 16, 32, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 128, 32, BLOCK_K, NSTAGE, 2, layout),
                (16, 16, 32, 32, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 32, 16, BLOCK_K, NSTAGE, 2, layout),
                (16, 16, 32, 32, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 64, 16, BLOCK_K, NSTAGE, 2, layout),

                # 4 wrap
                (16, 128, 16, 16, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 64, 16, BLOCK_K, NSTAGE, 4, layout),
                (16, 128, 16, 16, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 128, 16, BLOCK_K, NSTAGE, 4, layout),
                (16, 256, 16, 16, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 64, 16, BLOCK_K, NSTAGE, 4, layout),
                (16, 256, 16, 16, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 128, 16, BLOCK_K, NSTAGE, 4, layout),
                (16, 128, 32, 32, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 64, 16, BLOCK_K, NSTAGE, 4, layout),
                (16, 128, 32, 32, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 128, 16, BLOCK_K, NSTAGE, 4, layout),
                (16, 128, 32, 32, 32, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 128, 16, BLOCK_K, NSTAGE, 4, layout),
                (16, 128, 32, 32, 32, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 128, 32, BLOCK_K, NSTAGE, 4, layout),
                (16, 128, 32, 32, 64, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 128, 16, BLOCK_K, NSTAGE, 4, layout),
                (16, 128, 32, 32, 64, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 128, 32, BLOCK_K, NSTAGE, 4, layout),
                (16, 128, 32, 32, 64, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 128, 64, BLOCK_K, NSTAGE, 4, layout),
                (16, 16, 64, 64, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 64, 16, BLOCK_K, NSTAGE, 4, layout),
                (16, 16, 64, 64, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 128, 16, BLOCK_K, NSTAGE, 4, layout),
                (16, 16, 64, 64, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 256, 16, BLOCK_K, NSTAGE, 4, layout),
                (16, 8, 128, 128, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 128, 16, BLOCK_K, NSTAGE, 4, layout),
                (16, 8, 128, 128, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 256, 16, BLOCK_K, NSTAGE, 4, layout),

                # 8 wrap
                (16, 256, 32, 32, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 128, 16, BLOCK_K, NSTAGE, 8, layout),
                (16, 256, 32, 32, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 256, 16, BLOCK_K, NSTAGE, 8, layout),
                (16, 128, 64, 64, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 128, 16, BLOCK_K, NSTAGE, 8, layout),
                (16, 128, 64, 64, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 256, 16, BLOCK_K, NSTAGE, 8, layout),
                (16, 128, 128, 128, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 128, 16, BLOCK_K, NSTAGE, 8, layout),
                (16, 128, 128, 128, 16, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 256, 16, BLOCK_K, NSTAGE, 8, layout),
                (16, 128, 32, 32, 128, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 256, 64, BLOCK_K, NSTAGE, 8, layout),
                (16, 128, 32, 32, 128, 1, 1, (1, 1), (0, 0), (1, 1), 1, DTYPE, 128, 128, BLOCK_K, NSTAGE, 8, layout),

            ] for DTYPE in [torch.float32]
            for BLOCK_K in [32]
            for NSTAGE in [2, 3, 4]
            for layout in ["nchw", "nhwc"]
        ],
        # kernel, stride, padding, dilation
        *[
            [
                # kernel > 1
                (16, 128, 16, 16, 16, 2, 2, (1, 1), (0, 0), (1, 1), 1, DTYPE, 64, 16, BLOCK_K, NSTAGE, 4, layout),
                (16, 128, 16, 16, 16, 3, 3, (1, 1), (0, 0), (1, 1), 1, DTYPE, 64, 16, BLOCK_K, NSTAGE, 4, layout),
                (16, 128, 16, 16, 16, 4, 4, (1, 1), (0, 0), (1, 1), 1, DTYPE, 64, 16, BLOCK_K, NSTAGE, 4, layout),
                (16, 128, 16, 16, 16, 5, 5, (1, 1), (0, 0), (1, 1), 1, DTYPE, 64, 16, BLOCK_K, NSTAGE, 4, layout),
                (16, 128, 16, 16, 16, 7, 7, (1, 1), (0, 0), (1, 1), 1, DTYPE, 64, 16, BLOCK_K, NSTAGE, 4, layout),

                # stride > 1
                (16, 128, 16, 16, 16, 1, 1, (2, 2), (0, 0), (1, 1), 1, DTYPE, 64, 16, BLOCK_K, NSTAGE, 4, layout),
                (16, 128, 16, 16, 16, 1, 1, (3, 3), (0, 0), (1, 1), 1, DTYPE, 64, 16, BLOCK_K, NSTAGE, 4, layout),
                (16, 128, 16, 16, 16, 1, 1, (4, 4), (0, 0), (1, 1), 1, DTYPE, 64, 16, BLOCK_K, NSTAGE, 4, layout),
                (16, 128, 16, 16, 16, 1, 1, (5, 5), (0, 0), (1, 1), 1, DTYPE, 64, 16, BLOCK_K, NSTAGE, 4, layout),

                # padding > 0
                (16, 128, 16, 16, 16, 1, 1, (1, 1), (1, 1), (1, 1), 1, DTYPE, 64, 16, BLOCK_K, NSTAGE, 4, layout),
                (16, 128, 16, 16, 16, 1, 1, (1, 1), (2, 2), (1, 1), 1, DTYPE, 64, 16, BLOCK_K, NSTAGE, 4, layout),
                (16, 128, 16, 16, 16, 1, 1, (1, 1), (3, 3), (1, 1), 1, DTYPE, 64, 16, BLOCK_K, NSTAGE, 4, layout),

                # dilation > 1
                (16, 128, 16, 16, 16, 1, 1, (1, 1), (1, 1), (2, 2), 1, DTYPE, 64, 16, BLOCK_K, NSTAGE, 4, layout),
                (16, 128, 16, 16, 16, 1, 1, (1, 1), (1, 1), (3, 3), 1, DTYPE, 64, 16, BLOCK_K, NSTAGE, 4, layout),

            ] for DTYPE in [torch.float32]
            for BLOCK_K in [32]
            for NSTAGE in [2, 3, 4]
            for layout in ["nchw", "nhwc"]
        ],
    ),
)
def test_conv(
        # Tensor dimensions
        BATCH, IN_C, IN_H, IN_W,
        KERNEL_N, KERNEL_H, KERNEL_W,
        # parameters of conv
        stride, padding,
        dilation, groups,
        # others,
        dtype,
        # MATA
        BLOCK_M, BLOCK_N, BLOCK_K,
        NSTAGE, NWARP,
        layout,
    ):

    torch.manual_seed(0)
    # nuke kernel decorators -- will set meta-parameters manually
    kwargs = {'BLOCK_M': BLOCK_M, 'BLOCK_N': BLOCK_N, 'BLOCK_K': BLOCK_K}
    configs = [triton.Config(kwargs=kwargs, num_warps=NWARP, num_stages=NSTAGE)]
    kernel = torchinductor.triton_ops._conv.kernel
    decorators = kernel.kernel_decorators
    kernel.kernel_decorators = []
    triton.autotune(configs, [])(kernel)
    kernel.kernel_decorators += decorators[1:]

    # allocate inputs, nchw
    x = torch.randn((BATCH, IN_C, IN_H, IN_W), dtype=dtype, device='cuda')
    w = torch.randn((KERNEL_N, IN_C // groups, KERNEL_H, KERNEL_W),
                    dtype=dtype, device='cuda')
    bias = torch.randn((KERNEL_N), dtype=dtype, device='cuda')
    if layout == "nhwc":
        x = x.to(memory_format=torch.channels_last)
        w = w.to(memory_format=torch.channels_last)
    y = torchinductor.triton_ops.conv(
        x, w, bias, stride, padding, dilation, False, (0, 0), groups
    )

    y_correct = torch.conv2d(x, w, bias, stride, padding, dilation, groups)
    # print("y", y[0])
    # print("y_correct", y_correct[0])
    assert(same(y, y_correct, cos_similarity=True))
    print("passed")

BATCH, IN_C, IN_H, IN_W, KERNEL_N, KERNEL_H, KERNEL_W, stride, padding, dilation, groups, dtype, BLOCK_M, BLOCK_N, BLOCK_K, NSTAGE, NWARP = \
    128, 3, 224, 224, 64, 3, 3, (1, 1), (2, 2), (1, 1), 1, torch.float32, 128, 16, 128, 2, 4
# 32, 3, 224, 224, 64, 3, 3, (2, 2), (0, 0), (1, 1), 1, torch.float16, 128, 16, 32, 2, 4
# 32, 3, 224, 224, 32, 3, 3, (1, 1), (1, 1), (1, 1), 1, torch.float16, 128, 16, 32, 2, 4
# 
# 32, 128, 32, 32,32, 3, 3, (1, 1), (0, 0), (1, 1), 1, torch.float16, 128, 16, 32, 2, 4
    
test_conv(
        # Tensor dimensions
        BATCH, IN_C, IN_H, IN_W,
        KERNEL_N, KERNEL_H, KERNEL_W,
        # parameters of conv
        stride, padding,
        dilation, groups,
        # others,
        dtype,
        # MATA
        BLOCK_M, BLOCK_N, BLOCK_K,
        NSTAGE, NWARP,
        layout="nchw")
