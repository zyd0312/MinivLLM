"""Triton kernels for the validated single-device GPTQ INT8 layout."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gptq_prefill_kernel(
    x_ptr,
    qweight_ptr,
    scales_ptr,
    output_ptr,
    num_rows,
    input_size,
    output_size,
    stride_x_row,
    stride_x_col,
    stride_q_row,
    stride_q_col,
    stride_scale_row,
    stride_scale_col,
    NUM_K_BLOCKS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute X @ dequant(Q) without materializing dequantized weights.

    ``qweight`` stores four input values per int32, low byte first. The kernel
    uses FP16 dequantized values to match the reference implementation and
    accumulates the dot product in Triton's default FP32 accumulator.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = row_offsets < num_rows
    col_mask = col_offsets < output_size
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    zero_point = tl.full([1], 128.0, tl.float16)

    for block_idx in range(NUM_K_BLOCKS):
        input_offsets = block_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        input_mask = input_offsets < input_size
        x_ptrs = x_ptr + row_offsets[:, None] * stride_x_row + input_offsets[None, :] * stride_x_col
        x = tl.load(x_ptrs, mask=row_mask[:, None] & input_mask[None, :], other=0.0)

        packed_rows = input_offsets // 4
        byte_shifts = (input_offsets % 4) * 8
        q_ptrs = qweight_ptr + packed_rows[:, None] * stride_q_row + col_offsets[None, :] * stride_q_col
        packed = tl.load(q_ptrs, mask=input_mask[:, None] & col_mask[None, :], other=0)
        quantized = ((packed >> byte_shifts[:, None]) & 255).to(tl.float16)

        group_rows = input_offsets // GROUP_SIZE
        scale_ptrs = scales_ptr + group_rows[:, None] * stride_scale_row + col_offsets[None, :] * stride_scale_col
        scales = tl.load(scale_ptrs, mask=input_mask[:, None] & col_mask[None, :], other=0.0)
        weight = (quantized - zero_point) * scales
        accumulator += tl.dot(x, weight)

    output_ptrs = output_ptr + row_offsets[:, None] * output_size + col_offsets[None, :]
    tl.store(output_ptrs, accumulator.to(tl.float16), mask=row_mask[:, None] & col_mask[None, :])


@triton.jit
def _gptq_decode_kernel(
    x_ptr,
    qweight_ptr,
    scales_ptr,
    output_ptr,
    num_rows,
    input_size,
    output_size,
    stride_x_row,
    stride_x_col,
    stride_q_row,
    stride_q_col,
    stride_scale_row,
    stride_scale_col,
    NUM_K_BLOCKS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Decode GEMV kernel; one program handles one input row and N tile."""
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    col_offsets = col_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = col_offsets < output_size
    accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)
    zero_point = tl.full([1], 128.0, tl.float16)

    for block_idx in range(NUM_K_BLOCKS):
        input_offsets = block_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        input_mask = input_offsets < input_size
        x_ptrs = x_ptr + row_idx * stride_x_row + input_offsets * stride_x_col
        x = tl.load(x_ptrs, mask=input_mask, other=0.0)

        packed_rows = input_offsets // 4
        byte_shifts = (input_offsets % 4) * 8
        q_ptrs = qweight_ptr + packed_rows[:, None] * stride_q_row + col_offsets[None, :] * stride_q_col
        packed = tl.load(q_ptrs, mask=input_mask[:, None] & col_mask[None, :], other=0)
        quantized = ((packed >> byte_shifts[:, None]) & 255).to(tl.float16)

        group_rows = input_offsets // GROUP_SIZE
        scale_ptrs = scales_ptr + group_rows[:, None] * stride_scale_row + col_offsets[None, :] * stride_scale_col
        scales = tl.load(scale_ptrs, mask=input_mask[:, None] & col_mask[None, :], other=0.0)
        weight = (quantized - zero_point) * scales
        accumulator += tl.sum(x[:, None] * weight, axis=0)

    output_ptrs = output_ptr + row_idx * output_size + col_offsets
    tl.store(output_ptrs, accumulator.to(tl.float16), mask=(row_idx < num_rows) & col_mask)


def _validate_inputs(x: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    if not (x.is_cuda and qweight.is_cuda and scales.is_cuda):
        raise RuntimeError("Fused GPTQ kernels require CUDA tensors")
    if x.dtype != torch.float16 or qweight.dtype != torch.int32 or scales.dtype != torch.float16:
        raise ValueError("Fused GPTQ kernels require FP16 activations/scales and INT32 qweight")
    if x.shape[-1] <= 0 or x.shape[-1] % 128:
        raise ValueError("Fused GPTQ kernels require input_size divisible by group size 128")
    if qweight.ndim != 2 or scales.ndim != 2:
        raise ValueError("Fused GPTQ weights must be 2D")
    input_size = x.shape[-1]
    output_size = qweight.shape[1]
    if output_size <= 0:
        raise ValueError("Fused GPTQ weights must have a non-empty output dimension")
    if qweight.shape != (input_size // 4, output_size):
        raise ValueError("qweight shape does not match activation input size")
    if scales.shape != (input_size // 128, output_size):
        raise ValueError("scales shape does not match activation input size")
    flat_x = x.reshape(-1, input_size).contiguous()
    return flat_x, input_size, output_size


def _launch(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    *,
    block_m: int,
    block_n: int,
    num_warps: int,
) -> torch.Tensor:
    flat_x, input_size, output_size = _validate_inputs(x, qweight, scales)
    flat_x = flat_x.contiguous()
    qweight = qweight.contiguous()
    scales = scales.contiguous()
    num_rows = flat_x.shape[0]
    # Smaller K tiles keep the prefill dot-product kernel within the
    # shared-memory limit on consumer GPUs.
    block_k = 32
    grid = (triton.cdiv(num_rows, block_m), triton.cdiv(output_size, block_n))
    output = torch.empty((num_rows, output_size), dtype=flat_x.dtype, device=flat_x.device)
    _gptq_prefill_kernel[grid](
        flat_x,
        qweight,
        scales,
        output,
        num_rows,
        input_size,
        output_size,
        flat_x.stride(0),
        flat_x.stride(1),
        qweight.stride(0),
        qweight.stride(1),
        scales.stride(0),
        scales.stride(1),
        NUM_K_BLOCKS=input_size // block_k,
        GROUP_SIZE=128,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=1,
    )
    return output.reshape(*x.shape[:-1], output_size)


def _launch_decode(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    *,
    block_n: int,
    num_warps: int,
) -> torch.Tensor:
    flat_x, input_size, output_size = _validate_inputs(x, qweight, scales)
    qweight = qweight.contiguous()
    scales = scales.contiguous()
    num_rows = flat_x.shape[0]
    block_k = 128
    grid = (num_rows, triton.cdiv(output_size, block_n))
    output = torch.empty((num_rows, output_size), dtype=flat_x.dtype, device=flat_x.device)
    _gptq_decode_kernel[grid](
        flat_x,
        qweight,
        scales,
        output,
        num_rows,
        input_size,
        output_size,
        flat_x.stride(0),
        flat_x.stride(1),
        qweight.stride(0),
        qweight.stride(1),
        scales.stride(0),
        scales.stride(1),
        NUM_K_BLOCKS=input_size // block_k,
        GROUP_SIZE=128,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )
    return output.reshape(*x.shape[:-1], output_size)


def fused_gptq_decode(x: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Run the small-M GEMV-oriented GPTQ kernel used by decode."""
    _, _, output_size = _validate_inputs(x, qweight, scales)
    block_n = 64 if output_size <= 1024 else 128
    num_warps = 2 if block_n == 64 else 4
    return _launch_decode(x, qweight, scales, block_n=block_n, num_warps=num_warps)


def fused_gptq_prefill(x: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Run the multi-row GEMM-oriented GPTQ kernel used by prefill."""
    _, _, output_size = _validate_inputs(x, qweight, scales)
    rows = x.numel() // x.shape[-1]
    block_m = 16
    block_n = 64
    return _launch(x, qweight, scales, block_m=block_m, block_n=block_n, num_warps=4)
